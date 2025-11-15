import time
import pandas as pd
import math
import numpy as np

class system:
    def __init__(self, F1, F2, L1, costperkw, weather=False):
        """
        High-level system controller.

        F1: current inflow to tunnel      [m3/h]
        F2: current outflow to WWTP      [m3/h]
        L1: current tunnel level         [m]
        costperkw: current electricity price [€/kWh]
        weather: False = dry, True = rain
        """
        # --- PUMP INFO ---
        self.pumps = []  # 0,1 small; rest big
        for _ in range(2):
            self.pumps.append(Pump(is_small=True))
        for _ in range(6):
            self.pumps.append(Pump(is_small=False))

        # --- SIM / STATE ---
        self.sim_time = 0.0            # simulation time [s]
        self.realtime = time.time()
        self.weather = weather

        self.F1 = F1                   # inflow [m3/h]
        self.F2 = F2                   # outflow [m3/h]
        self.F2_target = F2            # commanded outflow [m3/h]
        self.L1 = L1                   # level [m]
        self.costperkw = costperkw     # price

        # --- LEVEL LIMITS ---
        self.L1_min = 0.0
        self.L1_max = 8.0
        self.flush_threshold = 0.5     # "flushed" if we get below this
        self.L1_safe_min = 0.2         # we really don't want to go below this

        # --- FLOW LIMITS (HIGH-LEVEL) ---
        self.F2_min = 0.0              # low-level ensures at least 1 pump ON
        # from data: F2 max ~10.8k → cap a bit above that
        self.F2_max = 11000.0          # ### CHANGED: more realistic cap
        self.max_F2_rate = self.F2_max

        # --- NOMINAL INFLOW MODEL (FROM DATA) ---
        # corrected F1 (m3/h) statistics from your dataset:
        # mean ≈ 6250, p10 ≈ 3200, p90 ≈ 10000
        # Use a slow filtered F1 to represent "typical load".
        self.F1_slow = F1              # ### CHANGED: smoothed inflow state

        # --- PID AROUND L1 ---
        # From data: mean L1 ≈ 2.25 m; target slightly above 2 m.
        self.L1p_base = 2.2            # ### CHANGED
        self.L1p = self.L1p_base

        # PID gains tuned down a bit (we let feed-forward do the heavy work)
        self.Kp = 300.0                # ### CHANGED
        self.Ki = 20.0                 # ### CHANGED
        self.Kd = 0.0

        self.error_prev = 0.0
        self.error_int = 0.0
        self.integral_limit = 6.0      # ### CHANGED: tighter anti-windup

        # --- ENERGY / PRICE NORMALIZATION ---
        # From data: median price ~4.6
        self.price_ref = 4.6           # ### CHANGED: fixed ref tied to data

        # --- FLUSH LOGIC ---
        self.flushed_today = False
        self.F1_low_threshold = 500.0
        self.seconds_per_day = 24 * 3600
        self.current_day_index = 0

        # --- STATS FOR SWITCHING ---
        self.last_num_switches = 0
        self.last_switch_mask = 0

    def step(self, timestep):
        """
        High-level + low-level update for one simulation step.

        - Updates F2_target using PID around L1 with price & flush logic.
        - Dispatches pumps with a heuristic that:
            * matches F2_target,
            * respects min on/off times,
            * spreads runtime across pumps (low-runtime & big pumps preferred),
            * limits switches per step.
        """
        dt = timestep
        dt_hours = dt / 3600.0

        # advance sim clock
        self.sim_time += dt
        self.realtime = time.time()

        # --- SMOOTH INFLOW MODEL (F1_slow) ---
        tau_hours = 6.0
        alpha_slow = min(1.0, dt_hours / tau_hours) if tau_hours > 0 else 1.0
        self.F1_slow = (1.0 - alpha_slow) * self.F1_slow + alpha_slow * self.F1
        F1_for_base = max(10.0, self.F1_slow)

        # --- DAILY FLUSH TRACKING ---
        day_index = int(self.sim_time // self.seconds_per_day)
        if day_index != self.current_day_index:
            self.current_day_index = day_index
            self.flushed_today = False

        if self.L1 <= self.flush_threshold:
            self.flushed_today = True

        # --- TIME-OF-DAY INFO ---
        seconds_into_day = self.sim_time % self.seconds_per_day
        hour_of_day = seconds_into_day / 3600.0

        # fixed daily flush schedule:
        #   prep window: 01:00–03:00
        #   flush window: 03:00–06:00
        flush_hour = 3.0
        flush_window_hours = 3.0
        prep_hours = 2.0

        flush_start = flush_hour
        flush_end = flush_hour + flush_window_hours
        prep_start = flush_start - prep_hours

        in_flush_window = (hour_of_day >= flush_start) and (hour_of_day < flush_end)
        in_prep_window = (hour_of_day >= prep_start) and (hour_of_day < flush_start)

        # --- PRICE RATIO / BIAS ---
        if self.price_ref > 0.0:
            price_ratio_raw = self.costperkw / self.price_ref
        else:
            price_ratio_raw = 1.0

        price_ratio_L1 = max(0.5, min(price_ratio_raw, 2.0))
        price_bias = max(-1.0, min(price_ratio_raw - 1.0, 2.0))  # -1..2

        # --- DYNAMIC SETPOINT L1p (LEVEL TARGET) ---
        L1_target = self.L1p_base

        k_price_L1 = 0.5  # m per 2x price change
        L1_target = self.L1p_base + k_price_L1 * (price_ratio_L1 - 1.0)

        if (not self.flushed_today) and in_prep_window:
            L1_target = min(L1_target, self.L1p_base - 0.3)

        if (not self.flushed_today) and in_flush_window:
            L1_target = min(L1_target, self.flush_threshold + 0.05)

        margin = 0.3
        L1_target = max(self.L1_min + margin,
                        min(L1_target, self.L1_max - margin))

        self.L1p = L1_target

        # --- PID ON LEVEL L1 ---
        error = self.L1 - self.L1p

        self.error_int += error * dt_hours
        self.error_int = max(-self.integral_limit,
                             min(self.error_int, self.integral_limit))

        if dt_hours > 0.0:
            derror_dt = (error - self.error_prev) / dt_hours
        else:
            derror_dt = 0.0
        self.error_prev = error

        u_pid = (self.Kp * error
                 + self.Ki * self.error_int
                 + self.Kd * derror_dt)    # [m3/h]

        # --- FEED-FORWARD FLOW BASED ON F1 & PRICE ---
        k_price_flow = 0.3
        F2_base = F1_for_base * (1.0 - k_price_flow * price_bias)
        F2_base = max(0.3 * F1_for_base, min(1.7 * F1_for_base, F2_base))

        F2_raw = F2_base + u_pid

        # --- EXTRA SAFETY NEAR HIGH LEVEL ---
        high_band = self.L1_max - 0.2
        if self.L1 > high_band:
            F2_raw = max(F2_raw, self.F1 + 2000.0)

        # --- FLUSH ENFORCEMENT (SOFT, NOT FULL SEND) ---  ### CHANGED
        if (not self.flushed_today) and in_flush_window and (self.L1 > self.flush_threshold + 0.05):
            # ensure we are definitely draining, but cap aggressiveness:
            flush_boost = 3000.0  # m3/h above inflow; tune if needed
            F2_raw = max(F2_raw, self.F1 + flush_boost)

        # --- SAFE-MIN HANDLING (REACTIVE) ---
        below_safe = (self.L1 <= self.L1_safe_min)
        if below_safe:
            # when already below safe min, really back off
            F2_raw = min(F2_raw, 0.8 * self.F1)
            if self.L1 <= 0.1:
                F2_raw = min(F2_raw, 0.2 * self.F1)

        # --- FLOW LIMITS ---
        F2_raw = max(self.F2_min, min(F2_raw, self.F2_max))

        # --- PREDICTIVE SAFE-MIN CHECK USING GEOMETRY ---  ### NEW, IMPORTANT
        try:
            V_current = height_to_volume(self.L1)
            V_safe = height_to_volume(self.L1_safe_min)
            V_pred_raw = V_current + dt_hours * (self.F1 - F2_raw)

            if V_pred_raw < V_safe and dt_hours > 0.0:
                # solve for F2 such that we land exactly at L1_safe_min
                F2_safe = self.F1 + (V_current - V_safe) / dt_hours
                F2_safe = max(self.F2_min, min(F2_safe, self.F2_max))
                F2_raw = min(F2_raw, F2_safe)
        except Exception:
            # if anything weird happens with the geometry table, just fall back
            pass

        # --- SMOOTHING OF F2_TARGET ---
        if below_safe or ((not self.flushed_today) and in_flush_window):
            self.F2_target = F2_raw
        else:
            delta_max = self.max_F2_rate * dt_hours   # [m3/h]
            delta = F2_raw - self.F2_target
            if delta > delta_max:
                delta = delta_max
            elif delta < -delta_max:
                delta = -delta_max
            self.F2_target = self.F2_target + delta

        # ---------- LOW-LEVEL HEURISTIC DISPATCHER ----------
        n = len(self.pumps)
        full_freq = 50.0          # Hz

        min_switch_hours = 0.25
        soft_switch_hours = 1.0
        max_switches_per_step = 2

        w_flow   = 15.0
        w_sw     = 0.5
        w_soft   = 2.0
        w_wear   = 2.0
        w_small  = 1.0

        self.last_num_switches = 0
        self.last_switch_mask = 0

        # --- 1. update per-pump timers & runtime ---
        for p in self.pumps:
            if not hasattr(p, "time_since_on"):
                p.time_since_on = 1e9 if p.on else 0.0
            if not hasattr(p, "time_since_off"):
                p.time_since_off = 1e9 if not p.on else 0.0
            if not hasattr(p, "runtime"):
                p.runtime = 0.0
            if not hasattr(p, "switch_count"):
                p.switch_count = 0

            if p.on:
                p.time_since_on += dt_hours
                p.time_since_off = 0.0
                p.runtime += dt_hours
            else:
                p.time_since_off += dt_hours
                p.time_since_on = 0.0

        full_flows = [p.conv_to_flow(full_freq) for p in self.pumps]

        max_runtime = max((p.runtime for p in self.pumps), default=0.0)
        if max_runtime <= 0.0:
            max_runtime = 1.0

        n_small = sum(1 for p in self.pumps if p.is_small) or 1

        best_score = math.inf
        best_mask = None
        best_switches = 0

        for mask in range(1, 1 << n):
            num_switches = 0
            soft_penalty = 0.0
            valid = True

            for i in range(n):
                new_on = bool((mask >> i) & 1)
                p = self.pumps[i]

                if new_on != p.on:
                    num_switches += 1

                    if new_on and p.time_since_off < min_switch_hours:
                        valid = False
                        break
                    if (not new_on) and p.time_since_on < min_switch_hours:
                        valid = False
                        break

                    if new_on and p.time_since_off < soft_switch_hours:
                        soft_penalty += (soft_switch_hours - p.time_since_off)
                    if (not new_on) and p.time_since_on < soft_switch_hours:
                        soft_penalty += (soft_switch_hours - p.time_since_on)

            if (not valid) or (num_switches > max_switches_per_step):
                continue

            F2_candidate = 0.0
            runtime_penalty = 0.0
            small_on_count = 0

            for i in range(n):
                new_on = bool((mask >> i) & 1)
                p = self.pumps[i]
                if new_on:
                    F2_candidate += full_flows[i]
                    small_on_count += int(p.is_small)
                    runtime_norm = p.runtime / max_runtime
                    size_weight = 2.0 if p.is_small else 1.0
                    runtime_penalty += size_weight * runtime_norm

            if F2_candidate > self.F2_max:
                continue

            if self.F2_target > 0.0:
                flow_err = abs(F2_candidate - self.F2_target) / self.F2_target
            else:
                flow_err = 0.0

            small_usage_term = small_on_count / n_small

            score = (
                w_flow * flow_err
                + w_sw * num_switches
                + w_soft * soft_penalty
                + w_wear * runtime_penalty
                + w_small * small_usage_term
            )

            if score < best_score:
                best_score = score
                best_mask = mask
                best_switches = num_switches

        if best_mask is not None:
            self.last_num_switches = best_switches
            self.last_switch_mask = best_mask
            for i in range(n):
                p = self.pumps[i]
                old_on = p.on
                new_on = bool((best_mask >> i) & 1)
                if new_on != old_on:
                    p.switch_count += 1
                p.on = new_on
                p.frequency = full_freq if new_on else 0.0
                p.flow = p.conv_to_flow(p.frequency)

        self.F2 = sum(p.flow for p in self.pumps)

    def get_weather(self):
        return self.weather


class Pump:
    def __init__(self, is_small):
        self.frequency = 0.0
        self.on = False
        self.flow = 0.0      # [m3/h]
        self.is_small = is_small

    def conv_to_flow(self, frequency):
        """
        Convert pump frequency [Hz] to flow [m3/h].

        Original curve ~ m3/s with 0.001 factor -> convert to m3/h.
        """
        if frequency <= 45.0:
            return 0.0

        if self.is_small:
            q_m3s = 0.001 * (36.46 * frequency - 1322.9)
        else:
            q_m3s = 0.001 * (63.81 * frequency - 2208.61)

        q_m3s = max(0.0, q_m3s)
        return q_m3s * 3600.0

    def conv_to_freq(self, flow):
        """
        Approx inverse: flow [m3/h] -> frequency [Hz].
        """
        if flow <= 0.0:
            return 0.0

        q_m3s = flow / 3600.0

        if self.is_small:
            freq = (q_m3s / 0.001 + 1322.9) / 36.46
        else:
            freq = (q_m3s / 0.001 + 2208.61) / 63.81

        return max(0.0, freq)


def system_from_row(row):
    """
    Create and initialize a `system` instance from a single pandas Series row.
    """
    L1 = float(row["Water level in tunnel L2"])
    F1 = float(row["Inflow to tunnel F1"]) * 4.0            # m3/15min → m3/h
    F2 = float(row["Sum of pumped flow to WWTP F2"])
    costperkw = float(row["Electricity price 2: normal"])
    weather = False

    sys = system(F1=F1, F2=F2, L1=L1, costperkw=costperkw, weather=weather)

    pump_ids_by_index = ["1.1", "2.1", "1.2", "1.4", "2.2", "2.3", "2.4"]
    small_ids = {"1.1", "2.1"}

    for i, pump in enumerate(sys.pumps):
        if i >= len(pump_ids_by_index):
            pump.frequency = 0.0
            pump.on = False
            pump.flow = 0.0
            continue

        pid = pump_ids_by_index[i]
        pump.is_small = (pid in small_ids)

        freq_col = f"Pump frequency {pid}"
        flow_col = f"Pump flow {pid}"

        if freq_col in row.index:
            freq = float(row[freq_col])
        else:
            freq = 0.0

        pump.frequency = freq
        pump.on = freq > 0.0

        if hasattr(pump, "conv_to_flow"):
            pump.flow = pump.conv_to_flow(freq)
        elif flow_col in row.index:
            pump.flow = float(row[flow_col] / 3600.0)
        else:
            pump.flow = 0.0

    return sys



def compute_baseline_cost(df, price_col="Electricity price 2: normal", max_days=None):
    df = df.sort_values("Time stamp").reset_index(drop=True)

    if max_days is not None:
        start_time = df["Time stamp"].iloc[0]
        end_time = start_time + pd.Timedelta(days=max_days)
        df = df[df["Time stamp"] < end_time].reset_index(drop=True)

    dt_seconds = df["Time stamp"].diff().dt.total_seconds().median()
    dt_hours = dt_seconds / 3600.0

    F2_raw = df["Sum of pumped flow to WWTP F2"].values  # m3/h
    price = df[price_col].values

    baseline_cost = float(np.sum(price * F2_raw * dt_hours))
    return baseline_cost




def simulate_controller(df, price_col="Electricity price 2: normal", max_days=None):
    """
    Run your controller over the cleaned_data time series.

    Returns:
      results_df, controller_cost, sys
    """
    df = df.sort_values("Time stamp").reset_index(drop=True)

    if max_days is not None:
        start_time = df["Time stamp"].iloc[0]
        end_time = start_time + pd.Timedelta(days=max_days)
        df = df[df["Time stamp"] < end_time].reset_index(drop=True)

    dt_seconds_series = df["Time stamp"].diff().dt.total_seconds()
    dt_seconds = float(dt_seconds_series.median())
    dt_hours = dt_seconds / 3600.0

    first_row = df.iloc[0]
    sys = system_from_row(first_row)

    L1_init = float(first_row["Water level in tunnel L2"])
    V = height_to_volume(L1_init)
    sys.L1 = L1_init

    sys.F1 = float(first_row["Inflow to tunnel F1"]) * 4.0
    sys.costperkw = float(first_row[price_col])
    sys.weather = False

    controller_cost = 0.0
    records = []

    F1_rain_threshold = 2000.0  # still optional

    for t in range(1, len(df)):
        row_prev = df.iloc[t - 1]
        row = df.iloc[t]

        dt = dt_seconds
        if pd.notna(row_prev["Time stamp"]) and pd.notna(row["Time stamp"]):
            dt = (row["Time stamp"] - row_prev["Time stamp"]).total_seconds() or dt_seconds
        dt_hours = dt / 3600.0

        sys.F1 = float(row_prev["Inflow to tunnel F1"]) * 4.0
        sys.costperkw = float(row_prev[price_col])
        sys.weather = bool(sys.F1 > F1_rain_threshold)

        # keep level consistent with volume
        sys.L1 = volume_to_height(V)

        # --- run controller ---
        sys.step(dt)

        # --- update physical state ---
        V = V + dt_hours * (sys.F1 - sys.F2)
        # still clamp V into geometry range so volume_to_height doesn't go crazy
        V = max(_VOLUMES[0], min(V, _VOLUMES[-1]))
        L1_new = volume_to_height(V)
        sys.L1 = L1_new

        controller_cost += sys.costperkw * sys.F2 * dt_hours

        # per-step pump state timeline  ### CHANGED
        pump_states = [int(p.on) for p in sys.pumps]
        pump_mask = 0
        for i, on in enumerate(pump_states):
            if on:
                pump_mask |= (1 << i)

        row_dict = {
            "Time stamp": row["Time stamp"],
            "F1": sys.F1,
            "price": sys.costperkw,
            "L1": sys.L1,
            "L1_target": sys.L1p,
            "F2_ctrl": sys.F2,
            "F2_target": sys.F2_target,
            "weather": sys.weather,
            "num_switches": getattr(sys, "last_num_switches", 0),
            "num_pumps_on": sum(pump_states),
            "switch_mask": pump_mask,
        }

        for i, on in enumerate(pump_states):
            row_dict[f"pump{i}_on"] = on

        records.append(row_dict)

    results_df = pd.DataFrame(records)
    return results_df, controller_cost, sys




def evaluate_results(results_df, baseline_cost, controller_cost,
                     L1_min=0.0, L1_max=8.0, flush_threshold=0.5,
                     F1_low_threshold=500.0):

    L1 = results_df["L1"].values
    L1_violations_low = np.sum(L1 < L1_min)
    L1_violations_high = np.sum(L1 > L1_max)
    L1_safe_min = 0.3
    L1_violations_safe = np.sum(L1 < L1_safe_min)
    L1_mean = float(np.mean(L1))

    results_df = results_df.copy()
    results_df["date"] = pd.to_datetime(results_df["Time stamp"]).dt.date

    flush_by_day = {}
    for day, group in results_df.groupby("date"):
        cond = (group["L1"] <= flush_threshold)
        flush_by_day[day] = bool(cond.any())

    num_days = len(flush_by_day)
    num_days_flushed = sum(flush_by_day.values())
    num_days_missed = num_days - num_days_flushed

    cost_diff = baseline_cost - controller_cost
    cost_diff_pct = 100.0 * cost_diff / baseline_cost if baseline_cost > 0 else np.nan

    print("=== L1 safety ===")
    print(f"  mean(L1) = {L1_mean:.3f}")
    print(f"  min(L1) = {L1.min():.3f}, max(L1) = {L1.max():.3f}")
    print(f"  violations below {L1_min}: {L1_violations_low}")
    print(f"  violations above {L1_max}: {L1_violations_high}")
    print(f"  violations below {L1_safe_min} (safe limit): {L1_violations_safe}")

    print("\n=== Daily flush ===")
    print(f"  days simulated: {num_days}")
    print(f"  days with successful flush: {num_days_flushed}")
    print(f"  days without flush: {num_days_missed}")

    if "num_switches" in results_df.columns:
        total_switches = int(results_df["num_switches"].sum())
        avg_switches_step = float(results_df["num_switches"].mean())
        switches_by_day = results_df.groupby("date")["num_switches"].sum()
        avg_switches_day = float(switches_by_day.mean())

        print("\n=== Pump switching ===")
        print(f"  total on/off events: {total_switches}")
        print(f"  avg switches per timestep: {avg_switches_step:.3f}")
        print(f"  avg switches per day: {avg_switches_day:.1f}")
    else:
        total_switches = avg_switches_step = avg_switches_day = None

    print("\n=== Cost comparison (proxy: price * flow) ===")
    print(f"  baseline_cost   ≈ {baseline_cost:,.2f}")
    print(f"  controller_cost ≈ {controller_cost:,.2f}")
    print(f"  absolute saving ≈ {cost_diff:,.2f}")
    print(f"  relative saving ≈ {cost_diff_pct:.2f}%")

    return {
        "L1_min": float(L1.min()),
        "L1_max": float(L1.max()),
        "L1_violations_low": int(L1_violations_low),
        "L1_violations_high": int(L1_violations_high),
        "days": int(num_days),
        "days_flushed": int(num_days_flushed),
        "days_missed": int(num_days_missed),
        "baseline_cost": float(baseline_cost),
        "controller_cost": float(controller_cost),
        "cost_diff": float(cost_diff),
        "cost_diff_pct": float(cost_diff_pct),
        "L1_mean": L1_mean,
        "L1_violations_safe": int(L1_violations_safe),
        "total_switches": total_switches,
        "avg_switches_per_step": avg_switches_step,
        "avg_switches_per_day": avg_switches_day,
    }



convtable_raw = pd.read_excel(
    "./raw_data/Volume of tunnel vs level Blominmäki.xlsx",
    sheet_name="Taul1"
)

convtable_raw = convtable_raw.iloc[:, :2].copy()
convtable_raw.columns = ["level", "volume"]
convtable_raw = convtable_raw.sort_values("level").reset_index(drop=True)

_LEVELS = convtable_raw["level"].to_numpy()
_VOLUMES = convtable_raw["volume"].to_numpy()


def height_to_volume(level):
    x = float(level)
    levels = _LEVELS
    vols = _VOLUMES

    if x <= levels[0]:
        x0, x1 = levels[0], levels[1]
        y0, y1 = vols[0], vols[1]
        dx = x1 - x0
        if abs(dx) < 1e-9:
            return float(y0)
        return y0 + (x - x0) * (y1 - y0) / dx

    if x >= levels[-1]:
        x0, x1 = levels[-2], levels[-1]
        y0, y1 = vols[-2], vols[-1]
        dx = x1 - x0
        if abs(dx) < 1e-9:
            return float(y1)
        return y0 + (x - x0) * (y1 - y0) / dx

    i = np.searchsorted(levels, x)
    x0, x1 = levels[i-1], levels[i]
    y0, y1 = vols[i-1], vols[i]
    dx = x1 - x0
    if abs(dx) < 1e-9:
        return float(0.5 * (y0 + y1))
    return y0 + (x - x0) * (y1 - y0) / dx


def volume_to_height(volume):
    y = float(volume)
    levels = _LEVELS
    vols = _VOLUMES

    if y <= vols[0]:
        y0, y1 = vols[0], vols[1]
        x0, x1 = levels[0], levels[1]
        dy = y1 - y0
        if abs(dy) < 1e-9:
            return float(x0)
        return x0 + (y - y0) * (x1 - x0) / dy

    if y >= vols[-1]:
        y0, y1 = vols[-2], vols[-1]
        x0, x1 = levels[-2], levels[-1]
        dy = y1 - y0
        if abs(dy) < 1e-9:
            return float(x1)
        return x0 + (y - y0) * (x1 - x0) / dy

    i = np.searchsorted(vols, y)
    y0, y1 = vols[i-1], vols[i]
    x0, x1 = levels[i-1], levels[i]
    dy = y1 - y0
    if abs(dy) < 1e-9:
        return float(0.5 * (x0 + x1))
    return x0 + (y - y0) * (x1 - x0) / dy





# 1. Load data
df = pd.read_csv("./output/cleaned_data.csv", parse_dates=["Time stamp"])

SIM_DAYS = 31  # ← change this to simulate any period you want

# 2. Baseline cost
baseline_cost = compute_baseline_cost(df, price_col="Electricity price 2: normal",
                                      max_days=SIM_DAYS)

# 3. Simulate controller
results_df, controller_cost, sys = simulate_controller(
    df, price_col="Electricity price 2: normal", max_days=SIM_DAYS
)

# 4. Evaluate
metrics = evaluate_results(results_df, baseline_cost, controller_cost)

# 5. Pump runtimes
print(f"\n=== Pump runtimes over {SIM_DAYS} days (hours) ===")
for i, p in enumerate(sys.pumps):
    print(
        f"Pump {i}:",
        "small" if p.is_small else "big",
        f"runtime = {getattr(p, 'runtime', 0.0):.2f} h"
    )

# 6. Per-pump switching intervals *table*
sim_start = results_df["Time stamp"].iloc[0]
sim_end   = results_df["Time stamp"].iloc[-1]
sim_hours = (sim_end - sim_start).total_seconds() / 3600.0

rows = []
for i, p in enumerate(sys.pumps):
    switches = getattr(p, "switch_count", 0)
    if switches > 0 and sim_hours > 0:
        avg_hours = sim_hours / switches
        avg_minutes = avg_hours * 60.0
    else:
        avg_minutes = float("nan")
    rows.append({
        "pump_index": i,
        "size": "small" if p.is_small else "big",
        "runtime_h": getattr(p, "runtime", 0.0),
        "switches": switches,
        "avg_minutes_between_switches": avg_minutes,
    })

switch_stats_df = pd.DataFrame(rows)
print(f"\n=== Pump switching intervals over {SIM_DAYS} days ===")
print(switch_stats_df.to_string(index=False))

