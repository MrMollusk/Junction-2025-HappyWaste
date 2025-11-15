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
        costperkw: current electricity price [€/kWh or whatever units]
        weather: False = dry, True = rain (you can override each step)
        """
        # --- PUMP INFO (for later low-level use) ---
        self.pumps = []  # 0,1 small; rest big
        for i in range(2):
            self.pumps.append(Pump(is_small=True))
        for i in range(6):
            self.pumps.append(Pump(is_small=False))

        # --- SIM / STATE ---
        self.sim_time = 0.0            # simulation time [s]
        self.realtime = time.time()    # wall-clock, only if you care
        self.weather = weather         # False=dry, True=rain

        self.F1 = F1                   # current inflow          [m3/h]
        self.F2 = F2                   # current actual outflow  [m3/h]
        self.F2_target = F2            # commanded total flow    [m3/h]
        self.L1 = L1                   # current level           [m]
        self.costperkw = costperkw     # current electricity price

        # --- LEVEL LIMITS ---
        self.L1_min = 0.0
        self.L1_max = 8.0
        self.flush_threshold = 0.5     # L1 < 0.5 m counts as "flushed"

        # --- FLOW LIMITS (HIGH-LEVEL) ---
        self.F2_min = 500.0            # don't ever go to 0 (keeps at least one pump on)
        self.F2_max = 16000.0          # max outflow (about 5–6 big pumps)

        # max *rate of change* of commanded flow [m3/h per hour]
        # (this keeps F2_target smooth)
        self.max_F2_rate = 4000.0

        # --- PID AROUND L1 ---
        self.L1p_base = 4.0            # "mid" setpoint [m]
        self.L1p = self.L1p_base       # current dynamic setpoint

        self.Kp = 500.0                # you will need to tune these
        self.Ki = 50.0
        self.Kd = 0.0

        self.error_prev = 0.0
        self.error_int = 0.0
        self.integral_limit = 10.0     # anti-windup clamp

        # --- ENERGY / PRICE NORMALIZATION ---
        self.price_ref = costperkw if costperkw > 0 else 1.0

        # --- FLUSH LOGIC ---
        self.flushed_today = False
        self.F1_low_threshold = 500.0  # if inflow below this, we treat as "low"
        self.seconds_per_day = 24 * 3600
        self.current_day_index = 0     # floor(sim_time / seconds_per_day)

    # timestep in seconds
    def step(self, timestep):
        """
        High-level update for one simulation step.

        - Updates F2_target (commanded outflow).
        - Uses L1, F1, costperkw, weather.
        - Keeps L1 within limits (via PID) and encourages cheap pumping + daily flush.
        """
        dt = timestep
        dt_hours = dt / 3600.0

        # advance sim clock
        self.sim_time += dt
        self.realtime = time.time()  # optional

        # --- DAILY FLUSH TRACKING ---
        day_index = int(self.sim_time // self.seconds_per_day)
        if day_index != self.current_day_index:
            # new simulated day
            self.current_day_index = day_index
            self.flushed_today = False

        is_rain = bool(self.weather)

        # mark flush achieved if conditions met
        if (not is_rain) and (self.L1 <= self.flush_threshold) and (self.F1 < self.F1_low_threshold):
            self.flushed_today = True

        # --- DYNAMIC SETPOINT L1p (LEVEL TARGET) ---

        # Start from a mid-level target
        L1_target = self.L1p_base

        # 1) Adjust for price: high price → allow L1 to be higher (pump less now),
        #    low price → prefer lower L1 (pump more now).
        if self.price_ref > 0.0:
            price_ratio = self.costperkw / self.price_ref
            # clamp to avoid crazy values
            price_ratio = max(0.5, min(price_ratio, 2.0))

            # k_price controls how much setpoint moves with price (in meters)
            k_price = 1.0
            # price_ratio=1 → no change; >1 → higher L1_target, <1 → lower
            L1_target = self.L1p_base + k_price * (price_ratio - 1.0)

        # 2) If we still haven't flushed today, it's dry, and inflow is low,
        #    push target down to encourage draining.
        if (not self.flushed_today) and (not is_rain) and (self.F1 < self.F1_low_threshold):
            L1_target = min(L1_target, self.flush_threshold)

        # 3) Clamp setpoint inside safe band
        margin = 0.3
        L1_target = max(self.L1_min + margin,
                        min(L1_target, self.L1_max - margin))

        self.L1p = L1_target

        # --- PID ON LEVEL L1 (with feed-forward from F1) ---

        # error: positive if level is too high
        error = self.L1 - self.L1p

        # integral term with anti-windup
        self.error_int += error * dt_hours
        self.error_int = max(-self.integral_limit,
                             min(self.error_int, self.integral_limit))

        # derivative term
        if dt_hours > 0.0:
            derror_dt = (error - self.error_prev) / dt_hours
        else:
            derror_dt = 0.0
        self.error_prev = error

        # PID correction
        u_pid = (self.Kp * error
                 + self.Ki * self.error_int
                 + self.Kd * derror_dt)

        # feed-forward: roughly match inflow by default
        F2_base = self.F1

        # raw command from high-level controller
        F2_raw = F2_base + u_pid

        # --- EXTRA SAFETY NEAR LIMITS ---

        # If we're very close to overflow, force strong pumping
        high_band = self.L1_max - 0.2
        if self.L1 > high_band:
            # ensure outflow at least inflow + some extra
            F2_raw = max(F2_raw, self.F1 + 2000.0)

        # If we're very close to empty, avoid over-draining
        low_band = self.L1_min + 0.2
        if self.L1 < low_band:
            # ensure we don't pump much more than inflow
            F2_raw = min(F2_raw, self.F1 + 500.0)

        # --- FLOW LIMITS (NO FULL STOP, NO CRAZY FLOW) ---
        F2_raw = max(self.F2_min, min(F2_raw, self.F2_max))

        # --- SMOOTHING: LIMIT RATE OF CHANGE IN F2 TARGET ---
        # max change allowed this step
        delta_max = self.max_F2_rate * dt_hours   # [m3/h]

        delta = F2_raw - self.F2_target
        if delta > delta_max:
            delta = delta_max
        elif delta < -delta_max:
            delta = -delta_max

        self.F2_target = self.F2_target + delta

        # NOTE:
        # self.F2_target is now updated.
        # The low-level pump controller + physics simulator will decide
        # what actual F2 (self.F2) and L1 next become.

        # If you want, you can return F2_target for convenience:
        
        # if you don't already have dt_hours defined above in step(), uncomment:
        # dt_hours = timestep / 3600.0

        n = len(self.pumps)
        full_freq = 50.0          # "full speed" frequency (Hz) – adjust if needed

        min_switch_hours = 0.25   # HARD limit: 15 min between on/off
        soft_switch_hours = 1.0   # SOFT preference: avoid switching more often than 1 h
        max_switches_per_step = 2 # at most this many pumps can change state in one step

        w_flow  = 1.0   # weight for matching F2_target
        w_sw    = 0.1   # weight for number of switches
        w_soft  = 1.0   # weight for switching too soon
        w_wear  = 0.1   # weight for runtime imbalance

        # --- 1. initialise per-pump timers and runtime if not present ---
        for p in self.pumps:
            if not hasattr(p, "time_since_on"):
                p.time_since_on = 1e9 if p.on else 0.0
            if not hasattr(p, "time_since_off"):
                p.time_since_off = 1e9 if not p.on else 0.0
            if not hasattr(p, "runtime"):
                p.runtime = 0.0

            if p.on:
                p.time_since_on += dt_hours
                p.time_since_off = 0.0
                p.runtime += dt_hours
            else:
                p.time_since_off += dt_hours
                p.time_since_on = 0.0

        # --- 2. precompute flow at full speed for each pump ---
        full_flows = []
        for p in self.pumps:
            full_flows.append(p.conv_to_flow(full_freq))

        # --- 3. search over all ON/OFF patterns (except all-off) ---
        best_score = math.inf
        best_mask = None

        for mask in range(1, 1 << n):  # 1..(2^n - 1), so at least one pump ON
            num_switches = 0
            soft_penalty = 0.0
            valid = True

            # 3a. enforce min switch time + switch count
            for i in range(n):
                new_on = bool((mask >> i) & 1)
                p = self.pumps[i]

                if new_on != p.on:
                    num_switches += 1

                    # HARD min on/off time
                    if new_on and p.time_since_off < min_switch_hours:
                        valid = False
                        break
                    if (not new_on) and p.time_since_on < min_switch_hours:
                        valid = False
                        break

                    # SOFT penalty for switching too often
                    if new_on and p.time_since_off < soft_switch_hours:
                        soft_penalty += (soft_switch_hours - p.time_since_off)
                    if (not new_on) and p.time_since_on < soft_switch_hours:
                        soft_penalty += (soft_switch_hours - p.time_since_on)

            if not valid or num_switches > max_switches_per_step:
                continue

            # 3b. compute candidate F2 and wear imbalance
            F2_candidate = 0.0
            runtimes_new = []
            for i in range(n):
                new_on = bool((mask >> i) & 1)
                p = self.pumps[i]

                if new_on:
                    F2_candidate += full_flows[i]
                    runtimes_new.append(p.runtime + dt_hours)
                else:
                    runtimes_new.append(p.runtime)

            # respect global F2_max from high-level
            if F2_candidate > self.F2_max:
                continue

            # runtime variance as a simple wear-imbalance metric
            mean_rt = sum(runtimes_new) / n
            var_rt = sum((rt - mean_rt) ** 2 for rt in runtimes_new) / n

            # 3c. objective: match target flow, avoid switches, spread wear
            flow_err = (F2_candidate - self.F2_target) ** 2

            score = (
                w_flow * flow_err
                + w_sw * num_switches
                + w_soft * soft_penalty
                + w_wear * var_rt
            )

            if score < best_score:
                best_score = score
                best_mask = mask

        # --- 4. apply best pattern (fallback: keep current pattern) ---
        if best_mask is not None:
            for i in range(n):
                new_on = bool((best_mask >> i) & 1)
                p = self.pumps[i]
                p.on = new_on
                p.frequency = full_freq if new_on else 0.0
                p.flow = p.conv_to_flow(p.frequency)

        # update actual total outflow from pumps
        self.F2 = sum(p.flow for p in self.pumps)
        

    def get_weather(self):
        # Placeholder for a real weather model or forecast.
        # For now, just return whatever was passed / set externally.
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

        Original curve looks like it's in m3/s with a 0.001 factor.
        Here we convert that to m3/h by multiplying by 3600.
        """
        if frequency <= 45.0:
            return 0.0

        if self.is_small:
            q_m3s = 0.001 * (36.46 * frequency - 1322.9)
        else:
            q_m3s = 0.001 * (63.81 * frequency - 2208.61)

        # Clamp negative flows to zero
        q_m3s = max(0.0, q_m3s)
        # Convert to m3/h
        return q_m3s * 3600.0

    def conv_to_freq(self, flow):
        """
        Approximate inverse: flow [m3/h] -> frequency [Hz].
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
    Create and initialize a `system` instance from a single pandas Series row
    of cleaned_data.csv.

    Assumes:
      - 'Water level in tunnel L2' is actually L1
      - Uses 'Inflow to tunnel F1' as F1
      - Uses 'Sum of pumped flow to WWTP F2' as F2
      - Uses 'Electricity price 2: normal' as costperkw (change if you want)
      - Initializes pump frequencies and flows from the row
      - Pumps 1.1 and 2.1 are small, all others big
    """

    # --- High-level scalars ---
    L1 = float(row["Water level in tunnel L2"])          # treat as L1
    F1 = float(row["Inflow to tunnel F1"])
    F2 = float(row["Sum of pumped flow to WWTP F2"])

    # choose your cost column; you can change this to "Electricity price 1: high"
    costperkw = float(row["Electricity price 2: normal"])

    # no explicit weather column → start with "dry"
    weather = False

    # --- Create system object ---
    sys = system(F1=F1, F2=F2, L1=L1, costperkw=costperkw, weather=weather)

    # --- Map dataframe pump columns to sys.pumps indices ---

    # sys.pumps: first 2 are small, rest big (by your design)
    # Data columns available: 1.1, 1.2, 1.4, 2.1, 2.2, 2.3, 2.4
    pump_ids_by_index = ["1.1", "2.1", "1.2", "1.4", "2.2", "2.3", "2.4"]
    small_ids = {"1.1", "2.1"}

    for i, pump in enumerate(sys.pumps):
        if i >= len(pump_ids_by_index):
            # No data column for this pump → leave it off by default
            pump.frequency = 0.0
            pump.on = False
            pump.flow = 0.0
            continue

        pid = pump_ids_by_index[i]
        pump.is_small = (pid in small_ids)

        freq_col = f"Pump frequency {pid}"
        flow_col = f"Pump flow {pid}"

        # frequency from data (0 means off)
        if freq_col in row.index:
            freq = float(row[freq_col])
        else:
            freq = 0.0

        pump.frequency = freq
        pump.on = freq > 0.0

        # flow: prefer your conversion function, fall back to the flow column
        if hasattr(pump, "conv_to_flow"):
            pump.flow = pump.conv_to_flow(freq)
        elif flow_col in row.index:
            pump.flow = float(row[flow_col] / 3600.0)
        else:
            pump.flow = 0.0

    return sys

def compute_baseline_cost(df, price_col="Electricity price 2: normal"):
    """
    Approximate baseline energy cost over the dataset, using raw F2.

    cost ≈ Σ price_t * F2_raw_t * Δt_hours
    where F2_raw is 'Sum of pumped flow to WWTP F2'.
    """
    df = df.sort_values("Time stamp").reset_index(drop=True)
    # dt per step in hours (assume regular sampling)
    dt_seconds = df["Time stamp"].diff().dt.total_seconds().median()
    dt_hours = dt_seconds / 3600.0

    F2_raw = df["Sum of pumped flow to WWTP F2"].values  # m3/h
    price = df[price_col].values

    baseline_cost = float(np.sum(price * F2_raw * dt_hours))
    return baseline_cost

def simulate_controller(df, price_col="Electricity price 2: normal"):
    """
    Run your `system` controller over the cleaned_data time series.

    Uses:
      - F1 from 'Inflow to tunnel F1'
      - price from `price_col`
      - L1 initial from 'Water level in tunnel L2'
      - tank mass-balance using height_to_volume / volume_to_height
    Returns:
      results_df, total_controller_cost
    """
    df = df.sort_values("Time stamp").reset_index(drop=True)

    # infer dt from data (assumes regular sampling, e.g. 15 min)
    dt_seconds_series = df["Time stamp"].diff().dt.total_seconds()
    dt_seconds = float(dt_seconds_series.median())
    dt_hours = dt_seconds / 3600.0

    # --- initialise system from first row ---
    first_row = df.iloc[0]
    sys = system_from_row(first_row)

    # make sure L1 matches first_row and initialise volume via helper
    L1_init = float(first_row["Water level in tunnel L2"])
    V = height_to_volume(L1_init)
    sys.L1 = L1_init

    sys.F1 = float(first_row["Inflow to tunnel F1"])
    sys.costperkw = float(first_row[price_col])
    sys.weather = False  # we’ll infer a simple rain/dry flag below

    controller_cost = 0.0
    records = []

    # simple threshold to detect "rain" from inflow
    F1_rain_threshold = 2000.0  # adjust this based on your data

    for t in range(1, len(df)):
        row_prev = df.iloc[t - 1]
        row = df.iloc[t]

        # use actual dt if it varies; else use median dt
        dt = dt_seconds
        if pd.notna(row_prev["Time stamp"]) and pd.notna(row["Time stamp"]):
            dt = (row["Time stamp"] - row_prev["Time stamp"]).total_seconds() or dt_seconds
        dt_hours = dt / 3600.0

        # --- update exogenous inputs for this interval (from prev row) ---
        sys.F1 = float(row_prev["Inflow to tunnel F1"])
        sys.costperkw = float(row_prev[price_col])

        # crude "weather" from inflow
        sys.weather = bool(sys.F1 > F1_rain_threshold)

        # keep system's L1 consistent with our volume before stepping
        sys.L1 = volume_to_height(V)

        # --- run controller (high-level + low-level) ---
        sys.step(dt)

        # --- update physical state using volume model ---
        # mass balance: V_{t+1} = V_t + dt * (F1 - F2)
        V = V + dt_hours * (sys.F1 - sys.F2)
        L1_new = volume_to_height(V)
        sys.L1 = L1_new

        # accumulate controller cost proxy (price * outflow)
        controller_cost += sys.costperkw * sys.F2 * dt_hours

        # record state for analysis
        records.append({
            "Time stamp": row["Time stamp"],
            "F1": sys.F1,
            "price": sys.costperkw,
            "L1": sys.L1,
            "L1_target": sys.L1p,
            "F2_ctrl": sys.F2,
            "F2_target": sys.F2_target,
            "weather": sys.weather,
        })

    results_df = pd.DataFrame(records)
    return results_df, controller_cost



def evaluate_results(results_df, baseline_cost, controller_cost,
                     L1_min=0.0, L1_max=8.0, flush_threshold=0.5,
                     F1_low_threshold=500.0):
    """
    Print / return metrics:
      - L1 violations
      - daily flush success
      - baseline vs controller cost
    """
    # --- 1. L1 safety ---
    L1 = results_df["L1"].values
    L1_violations_low = np.sum(L1 < L1_min)
    L1_violations_high = np.sum(L1 > L1_max)

    # --- 2. Flush requirement: per day, at least one timestep with
    #       L1 <= flush_threshold AND F1 < F1_low_threshold and weather==False
    results_df = results_df.copy()
    results_df["date"] = pd.to_datetime(results_df["Time stamp"]).dt.date

    flush_by_day = {}
    for day, group in results_df.groupby("date"):
        cond = (
            (group["L1"] <= flush_threshold) &
            (group["F1"] < F1_low_threshold) &
            (~group["weather"])
        )
        flush_by_day[day] = bool(cond.any())

    num_days = len(flush_by_day)
    num_days_flushed = sum(flush_by_day.values())
    num_days_missed = num_days - num_days_flushed

    # --- 3. Cost comparison ---
    cost_diff = baseline_cost - controller_cost
    cost_diff_pct = 100.0 * cost_diff / baseline_cost if baseline_cost > 0 else np.nan

    print("=== L1 safety ===")
    print(f"  min(L1) = {L1.min():.3f}, max(L1) = {L1.max():.3f}")
    print(f"  violations below {L1_min}: {L1_violations_low}")
    print(f"  violations above {L1_max}: {L1_violations_high}")

    print("\n=== Daily flush ===")
    print(f"  days simulated: {num_days}")
    print(f"  days with successful flush: {num_days_flushed}")
    print(f"  days without flush: {num_days_missed}")

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
    }


convtable_raw = pd.read_excel(
    "./raw_data/Volume of tunnel vs level Blominmäki.xlsx",
    sheet_name="Taul1"
)

# keep only the first two columns: level, volume
convtable_raw = convtable_raw.iloc[:, :2].copy()
convtable_raw.columns = ["level", "volume"]

# ensure sorted by level
convtable_raw = convtable_raw.sort_values("level").reset_index(drop=True)

# store as numpy arrays for fast interpolation
_LEVELS = convtable_raw["level"].to_numpy()   # [m]
_VOLUMES = convtable_raw["volume"].to_numpy() # [m3]


# --- Helper: level -> volume ---

def height_to_volume(level):
    """
    Convert water level in tunnel [m] to volume [m3] using linear interpolation
    on the lookup table.

    For levels outside the table range, clamps to min/max volume.
    Works with scalars or numpy arrays.
    """
    return np.interp(level, _LEVELS, _VOLUMES)


# --- Helper: volume -> level (inverse) ---

def volume_to_height(volume):
    """
    Convert tunnel volume [m3] to water level [m] using linear interpolation
    on the same lookup table.

    For volumes outside the table range, clamps to min/max level.
    Works with scalars or numpy arrays.
    """
    return np.interp(volume, _VOLUMES, _LEVELS)






# 1. Load data with timestamp parsed
df = pd.read_csv("./output/cleaned_data.csv", parse_dates=["Time stamp"])

# 2. Compute baseline cost
baseline_cost = compute_baseline_cost(df, price_col="Electricity price 2: normal")

# 3. Simulate controller
results_df, controller_cost = simulate_controller(df, price_col="Electricity price 2: normal")

# 4. Evaluate
metrics = evaluate_results(results_df, baseline_cost, controller_cost)
