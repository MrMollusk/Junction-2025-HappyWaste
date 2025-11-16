import time
import math
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


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
        self.L1_safe_min = 0.2         # absolute safety floor
        self.L1_oper_min = 1.0         # normal operation should stay above this

        # --- FLOW LIMITS (HIGH-LEVEL) ---
        self.F2_min = 0.0
        self.F2_max = 16000.0          # cap a bit above real max
        self.max_F2_rate = self.F2_max

        # --- NOMINAL INFLOW MODEL ---
        self.F1_slow = F1

        # --- PID AROUND L1 ---
        # “Nominal” level is ~2.2 m; we sit a bit above 2 m.
        self.L1p_base = 2.2
        self.L1p = self.L1p_base

        # PID gains – softer; feed-forward does the heavy lifting.
        self.Kp = 80.0       # was 300
        self.Ki = 4.0        # was 20
        self.Kd = 60.0       # was 0 → add damping

        self.error_prev = 0.0
        self.error_int = 0.0
        self.integral_limit = 3.0   # tighter anti-windup

        # --- ENERGY / PRICE NORMALIZATION ---
        # note: you divide by 100 in the script part
        self.price_ref = 0.046

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
        """
        dt = timestep
        dt_hours = dt / 3600.0

        # VFD limits
        full_freq = 50.0
        freq_min = 47.6        # Hz – lowest allowed VFD speed

        # ---------------------------------------------------
        # TIME / INFLOW MODEL
        # ---------------------------------------------------
        self.sim_time += dt
        self.realtime = time.time()

        # Smooth nominal inflow
        tau_hours = 6.0
        alpha_slow = min(1.0, dt_hours / tau_hours) if tau_hours > 0 else 1.0
        self.F1_slow = (1.0 - alpha_slow) * self.F1_slow + alpha_slow * self.F1
        F1_for_base = max(10.0, self.F1_slow)

        # Day / flush tracking
        day_index = int(self.sim_time // self.seconds_per_day)
        if day_index != self.current_day_index:
            self.current_day_index = day_index
            self.flushed_today = False

        # if we ever actually get below the flush threshold, remember it
        if self.L1 <= self.flush_threshold:
            self.flushed_today = True

        seconds_into_day = self.sim_time % self.seconds_per_day
        hour_of_day = seconds_into_day / 3600.0

        flush_hour = 3.0
        flush_window_hours = 4.0
        prep_hours = 2.0

        flush_start = flush_hour
        flush_end = flush_hour + flush_window_hours
        prep_start = flush_start - prep_hours

        in_flush_window = (hour_of_day >= flush_start) and (hour_of_day < flush_end)
        in_prep_window = (hour_of_day >= prep_start) and (hour_of_day < flush_start)

        # ---------------------------------------------------
        # PRICE EFFECTS
        # ---------------------------------------------------
        if self.price_ref > 0.0:
            price_ratio_raw = self.costperkw / self.price_ref
        else:
            price_ratio_raw = 1.0

        price_ratio_L1 = max(0.5, min(price_ratio_raw, 2.0))
        price_bias = max(-0.5, min(price_ratio_raw - 1.0, 0.5))

        # ---------------------------------------------------
        # L1 SETPOINT (WITH FLUSH / PREP / PRICE)
        # ---------------------------------------------------
        L1_target = self.L1p_base
        k_price_L1 = 0.5
        L1_target = self.L1p_base + k_price_L1 * (price_ratio_L1 - 1.0)

        if (not self.flushed_today) and in_flush_window:
            # FLUSH MODE: we conceptually "aim" at the flush threshold
            L1_target = self.flush_threshold + 0.1

        elif (not self.flushed_today) and in_prep_window:
            # PREP MODE: ramp from nominal level down to just above 1.0 m
            prep_span = flush_start - prep_start  # = prep_hours
            if prep_span > 0.0:
                prog = (hour_of_day - prep_start) / prep_span
                prog = max(0.0, min(prog, 1.0))
            else:
                prog = 1.0

            # where we want to be at flush_start (≈1.0–1.1 m)
            L1_prep_target = max(self.L1_oper_min,
                                 self.flush_threshold)  # ~1.0 m

            # linear ramp: at prog=0 we sit at L1p_base, at prog=1 we sit at L1_prep_target
            L1_target = L1_prep_target + (1.0 - prog) * (self.L1p_base - L1_prep_target)

        else:
            # NORMAL MODE: never aim below operational minimum
            L1_target = max(self.L1_oper_min, L1_target)

        # Clamp inside [margin, L1_max - margin]
        margin = self.L1_safe_min
        L1_target = max(self.L1_min - margin,
                        min(L1_target, self.L1_max - margin))
        self.L1p = L1_target

        # ---------------------------------------------------
        # PID ON L1
        # ---------------------------------------------------
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

        # Fade out price influence when level is low
        if self.L1p_base > self.L1_oper_min:
            fill_frac = (self.L1 - self.L1_oper_min) / (self.L1p_base - self.L1_oper_min)
            fill_frac = max(0.0, min(fill_frac, 1.0))
        else:
            fill_frac = 1.0

        effective_price_bias = price_bias * fill_frac

        # ---------------------------------------------------
        # FEED-FORWARD F2 BASED ON F1 & PRICE
        # ---------------------------------------------------
        k_price_flow = 0.15
        F2_base = F1_for_base * (1.0 - k_price_flow * effective_price_bias)
        F2_base = max(0.3 * F1_for_base, min(1.7 * F1_for_base, F2_base))

        F2_raw = F2_base + u_pid

        # Extra safety near high level
        high_band = self.L1_max - 0.2
        if self.L1 > high_band:
            F2_raw = max(F2_raw, self.F1 + 2000.0)

        # When below operational minimum in NORMAL mode, don't drain faster than inflow
        if (not in_flush_window) and (self.L1 < self.L1_oper_min):
            F2_raw = min(F2_raw, self.F1)

        # ---------------------------------------------------
        # FLUSH ENFORCEMENT – AGGRESSIVE, LET SAFE-MIN CLIP
        # ---------------------------------------------------
        if (not self.flushed_today) and in_flush_window:
            try:
                V_current = height_to_volume(self.L1)
                V_goal = 0.0
                V_safe = height_to_volume(self.L1_safe_min - 0.1)

                # Remaining time in window
                hours_left = flush_end - hour_of_day
                if hours_left < dt_hours:
                    hours_left = dt_hours

                # Aim to finish well within the window
                flush_horizon = min(hours_left / 2.0, 0.6)  # <= ~0.6 h

                if V_current > V_goal:
                    net_needed = (V_current - V_goal) / flush_horizon
                else:
                    net_needed = 0.0

                k_aggr = 5.0
                net_needed *= k_aggr

                F2_flush = self.F1 + net_needed
                F2_flush = max(self.F2_min, min(F2_flush, self.F2_max))

                if F2_flush > F2_raw:
                    F2_raw = F2_flush

            except Exception:
                # fall back if geometry table fails
                pass

        # ---------------------------------------------------
        # SAFE-MIN FLAG (no clamping here – real clamp is outside)
        # ---------------------------------------------------
        below_safe = (self.L1 <= self.L1_safe_min)

        # Flow limits
        F2_raw = max(self.F2_min, min(F2_raw, self.F2_max))

        # Monotonicity guard vs level (but NOT during flush)
        if not in_flush_window:
            if (self.L1 < self.L1p_base or self.L1 > 4.5) and F2_raw > self.F2_target:
                F2_raw = self.F2_target

        # ---------------------------------------------------
        # SMOOTHING OF F2_TARGET
        # ---------------------------------------------------
        if below_safe or ((not self.flushed_today) and in_flush_window):
            self.F2_target = F2_raw
        else:
            delta_max = self.max_F2_rate * dt_hours
            delta = F2_raw - self.F2_target
            if delta > delta_max:
                delta = delta_max
            elif delta < -delta_max:
                delta = -delta_max
            self.F2_target = self.F2_target + delta

        # ---------------------------------------------------
        # LOW-LEVEL DISPATCHER WITH CONTINUOUS FREQUENCY
        # ---------------------------------------------------
        n = len(self.pumps)

        min_switch_hours = 0.25
        soft_switch_hours = 1.0
        max_switches_per_step = 2

        w_flow = 15.0
        w_sw = 0.5
        w_soft = 2.0
        w_wear = 2.0
        w_small = 0.0

        self.last_num_switches = 0
        self.last_switch_mask = 0

        # 1) Update per-pump timers & runtime
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

        # Linear models q = a * f + b for small & big pumps
        freq_hi = 50.0
        freq_lo = 49.0

        tmp_small = Pump(is_small=True)
        q_small_hi = tmp_small.conv_to_flow(freq_hi)
        q_small_lo = tmp_small.conv_to_flow(freq_lo)
        a_small = (q_small_hi - q_small_lo) / (freq_hi - freq_lo)
        b_small = q_small_hi - a_small * freq_hi

        tmp_big = Pump(is_small=False)
        q_big_hi = tmp_big.conv_to_flow(freq_hi)
        q_big_lo = tmp_big.conv_to_flow(freq_lo)
        a_big = (q_big_hi - q_big_lo) / (freq_hi - freq_lo)
        b_big = q_big_hi - a_big * freq_hi

        max_runtime = max((p.runtime for p in self.pumps), default=0.0)
        if max_runtime <= 0.0:
            max_runtime = 1.0

        n_small = sum(1 for p in self.pumps if p.is_small) or 1

        # local F2 cap – only global limit;
        local_F2_max = self.F2_max

        best_score = math.inf
        best_mask = None
        best_switches = 0
        best_freq = full_freq

        # 2) Search over on/off masks, but allow continuous frequency
        for mask in range(1, 1 << n):
            num_switches = 0
            soft_penalty = 0.0
            valid = True

            ns = 0   # number of small pumps on
            nb = 0   # number of big pumps on

            for i in range(n):
                new_on = bool((mask >> i) & 1)
                p = self.pumps[i]

                if new_on:
                    if p.is_small:
                        ns += 1
                    else:
                        nb += 1

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
            if ns + nb == 0:
                continue

            # Total flow model for this mask: F2 = a_total * f + b_total
            a_total = ns * a_small + nb * a_big
            b_total = ns * b_small + nb * b_big
            if a_total <= 0.0:
                continue

            # Ideal common frequency to hit F2_target
            f_ideal = (self.F2_target - b_total) / a_total

            if f_ideal < freq_min:
                f_cmd = freq_min
            elif f_ideal > full_freq:
                f_cmd = full_freq
            else:
                f_cmd = f_ideal

            F2_candidate = a_total * f_cmd + b_total

            if F2_candidate > local_F2_max:
                continue

            if self.F2_target > 0.0:
                flow_err = abs(F2_candidate - self.F2_target) / self.F2_target
            else:
                flow_err = 0.0

            runtime_penalty = 0.0
            small_on_count = 0
            for i in range(n):
                new_on = bool((mask >> i) & 1)
                p = self.pumps[i]
                if new_on:
                    small_on_count += int(p.is_small)
                    runtime_norm = p.runtime / max_runtime
                    runtime_penalty += runtime_norm

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
                best_freq = f_cmd

        # 3) Apply best mask and frequency
        if best_mask is not None:
            self.last_num_switches = best_switches
            self.last_switch_mask = best_mask
            for i in range(n):
                p = self.pumps[i]
                new_on = bool((best_mask >> i) & 1)
                old_on = p.on
                if new_on != old_on:
                    p.switch_count += 1
                p.on = new_on
                if new_on:
                    p.frequency = best_freq
                    p.flow = p.conv_to_flow(p.frequency)
                else:
                    p.frequency = 0.0
                    p.flow = 0.0

        # Final outflow
        self.F2 = sum(p.flow for p in self.pumps)

        # -----------------------------------------------
        # FINAL GEOMETRIC SAFE-MIN OVERRIDE (AFTER F2 SET)
        # -----------------------------------------------
        try:
            if dt_hours > 0.0 and self.F2 > 0.0:
                V_current = height_to_volume(self.L1)
                V_safe = height_to_volume(self.L1_safe_min)

                V_pred = V_current + dt_hours * (self.F1 - self.F2)

                if V_pred < V_safe:
                    F2_safe = self.F1 + (V_current - V_safe) / dt_hours
                    F2_safe = max(self.F2_min, min(F2_safe, self.F2_max))

                    if F2_safe < self.F2:
                        scale = F2_safe / self.F2
                        scale = max(0.0, min(scale, 1.0))

                        for p in self.pumps:
                            if p.on and p.frequency > 0.0:
                                new_freq = freq_min + (p.frequency - freq_min) * scale

                                if new_freq <= 45.0:
                                    p.on = False
                                    p.frequency = 0.0
                                    p.flow = 0.0
                                    p.switch_count = getattr(p, "switch_count", 0) + 1
                                else:
                                    p.frequency = new_freq
                                    p.flow = p.conv_to_flow(p.frequency)

                        self.F2 = sum(p.flow for p in self.pumps)
        except Exception:
            pass


class Pump:
    def __init__(self, is_small):
        self.frequency = 0.0
        self.on = False
        self.flow = 0.0      # [m3/h]
        self.is_small = is_small

    def conv_to_flow(self, frequency):
        """
        Convert pump frequency [Hz] to flow [m3/h].
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
    Run your controller over the cleaned_data time series with substeps.

    Returns:
      results_df, controller_cost, sys
    """
    df = df.sort_values("Time stamp").reset_index(drop=True)

    if max_days is not None:
        start_time = df["Time stamp"].iloc[0]
        end_time = start_time + pd.Timedelta(days=max_days)
        df = df[df["Time stamp"] < end_time].reset_index(drop=True)

    dt_seconds_series = df["Time stamp"].diff().dt.total_seconds()
    dt_seconds_nominal = float(dt_seconds_series.median())
    if not np.isfinite(dt_seconds_nominal) or dt_seconds_nominal <= 0:
        dt_seconds_nominal = 900.0  # assume 15 min

    first_row = df.iloc[0]
    sys = system_from_row(first_row)

    L1_init_raw = float(first_row["Water level in tunnel L2"])
    L1_init = max(L1_init_raw, sys.L1_safe_min)
    V = height_to_volume(L1_init)
    sys.L1 = L1_init

    sys.F1 = float(first_row["Inflow to tunnel F1"]) * 4.0
    sys.costperkw = float(first_row[price_col])
    sys.weather = False

    controller_cost = 0.0
    records = []

    F1_rain_threshold = 2000.0

    DT_SUB_SECONDS = 60.0  # 1-minute integration steps

    print(
        "timestamp, "
        "L1_m, "
        "F1_m3ph, "
        "F2_m3ph, "
        "price_EUR_per_kWh, "
        "pump_freqs_Hz"
    )

    for t in range(1, len(df)):
        row_prev = df.iloc[t - 1]
        row = df.iloc[t]

        total_dt = (row["Time stamp"] - row_prev["Time stamp"]).total_seconds()
        if not np.isfinite(total_dt) or total_dt <= 0:
            total_dt = dt_seconds_nominal

        n_sub = max(1, int(math.ceil(total_dt / DT_SUB_SECONDS)))
        dt_sub = total_dt / n_sub
        dt_sub_hours = dt_sub / 3600.0

        F1_prev = float(row_prev["Inflow to tunnel F1"]) * 4.0
        F1_next = float(row["Inflow to tunnel F1"]) * 4.0

        price_prev = float(row_prev[price_col])
        price_next = float(row[price_col])

        for k in range(n_sub):
            frac = float(k + 1) / float(n_sub)

            sys.F1 = F1_prev + frac * (F1_next - F1_prev)
            sys.costperkw = price_prev + frac * (price_next - price_prev)
            sys.weather = bool(sys.F1 > F1_rain_threshold)

            sys.L1 = volume_to_height(V)

            sys.step(dt_sub)

            V_current = V
            V_floor = height_to_volume(sys.L1_safe_min)
            V_trigger = height_to_volume(0.2)

            if dt_sub_hours > 0.0 and sys.F2 > 0.0:
                V_pred = V_current + dt_sub_hours * (sys.F1 - sys.F2)

                if V_pred < V_trigger:
                    pumps_on = [(i, p) for i, p in enumerate(sys.pumps)
                                if p.on and p.flow > 0.0]
                    pumps_on.sort(key=lambda ip: ip[1].flow, reverse=True)

                    for idx, p in pumps_on:
                        p.on = False
                        p.frequency = 0.0
                        p.flow = 0.0
                        p.switch_count = getattr(p, "switch_count", 0) + 1

                        F2_candidate = sum(pp.flow for pp in sys.pumps)
                        V_candidate = V_current + dt_sub_hours * (sys.F1 - F2_candidate)

                        sys.F2 = F2_candidate

                        if V_candidate >= V_floor:
                            break

                    sys.F2_target = min(sys.F2_target, sys.F2)

            V = V + dt_sub_hours * (sys.F1 - sys.F2)
            V = max(_VOLUMES[0], min(V, _VOLUMES[-1]))
            L1_new = volume_to_height(V)
            L1_new = max(sys.L1_min, min(L1_new, sys.L1_max))
            sys.L1 = L1_new

            controller_cost += sys.costperkw * sys.F2 * dt_sub_hours

        pump_states = [int(p.on) for p in sys.pumps]
        pump_mask = 0
        for i, on in enumerate(pump_states):
            if on:
                pump_mask |= (1 << i)

        pump_freqs_str = ", ".join(
            f"{i}:{p.frequency:.1f}"
            for i, p in enumerate(sys.pumps)
        )
        print(
            f"{row['Time stamp']}, "
            f"{sys.L1:.3f}, "
            f"{sys.F1:.1f}, "
            f"{sys.F2:.1f}, "
            f"{sys.costperkw:.4f}, "
            f"[{pump_freqs_str}]"
        )

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

        for i, p in enumerate(sys.pumps):
            row_dict[f"pump{i}_on"] = int(p.on)
            row_dict[f"pump{i}_freq"] = float(getattr(p, "frequency", 0.0))
            row_dict[f"pump{i}_flow"] = float(getattr(p, "flow", 0.0))
            row_dict[f"pump{i}_is_small"] = bool(p.is_small)

        records.append(row_dict)

    results_df = pd.DataFrame(records)
    return results_df, controller_cost, sys


def evaluate_results(results_df, baseline_cost, controller_cost,
                     L1_min=0.0, L1_max=8.0, flush_threshold=0.5,
                     F1_low_threshold=500.0):

    L1 = results_df["L1"].values
    L1_violations_low = np.sum(L1 < L1_min)
    L1_violations_high = np.sum(L1 > L1_max)
    L1_safe_min = 0.1
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


# Geometry tables – NOTE path now uses ROOT
convtable_raw = pd.read_excel(
    ROOT / "raw_data" / "Volume of tunnel vs level Blominmäki.xlsx",
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
    x0, x1 = levels[i - 1], levels[i]
    y0, y1 = vols[i - 1], vols[i]
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
    y0, y1 = vols[i - 1], vols[i]
    x0, x1 = levels[i - 1], levels[i]
    dy = y1 - y0
    if abs(dy) < 1e-9:
        return float(0.5 * (x0 + x1))
    return x0 + (y - y0) * (x1 - x0) / dy


if __name__ == "__main__":
    # 1. Load data from backend/output
    df = pd.read_csv(ROOT / "output" / "cleaned_data.csv", parse_dates=["Time stamp"])
    df["Electricity price 2: normal"] = df["Electricity price 2: normal"] / 100.0
    SIM_DAYS = 7

    # 2. Baseline cost
    baseline_cost = compute_baseline_cost(
        df, price_col="Electricity price 2: normal", max_days=SIM_DAYS
    )

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

    # 6. Switching stats
    sim_start = results_df["Time stamp"].iloc[0]
    sim_end = results_df["Time stamp"].iloc[-1]
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
