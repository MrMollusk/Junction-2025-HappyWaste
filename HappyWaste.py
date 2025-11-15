import time
import pandas as pd

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
        return self.F2_target

    def get_weather(self):
        # Placeholder for a real weather model or forecast.
        # For now, just return whatever was passed / set externally.
        return self.weather


class Pump:
    def __init__(self, is_small):
        self.frequency = 0.0
        self.on = False
        self.flow = 0.0
        self.is_small = is_small

    def conv_to_flow(self, frequency):
        if frequency <= 45:
            return 0.0
        if self.is_small:
            return 0.001 * (36.46*frequency - 1322.9)
        else:
            return 0.001 * (63.81*frequency - 2208.61)

    def conv_to_freq(self, flow):
        if flow <= 200.0:
            return 0.0
        if self.is_small:
            return (1322.9 + flow/0.001) / 36.46
        else:
            return (2208.61 + flow/0.001) / 63.81

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


df = pd.read_csv("./output/cleaned_data.csv")

row = df.iloc[100]          # pick any row
sys = system_from_row(row)  # build a system from that snapshot

print("L1:", sys.L1)
print("F1:", sys.F1)
print("F2:", sys.F2)
print("Price:", sys.costperkw)
for i, p in enumerate(sys.pumps):
    print(i, "small" if p.is_small else "big",
          "freq:", p.frequency, "flow:", p.flow, "on:", p.on)