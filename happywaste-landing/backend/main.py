from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# --------------------------------------------------------------------
# Path setup so we can import HappyWaste.py which lives in backend/
# --------------------------------------------------------------------
BACKEND_ROOT = Path(__file__).resolve().parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from HappyWaste import height_to_volume, simulate_controller  # type: ignore

app = FastAPI(title="HappyWaste Demo Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Precomputed 1-minute states for 7 days
MINUTE_STATES: List[Dict[str, Any]] = []
DEMO_START_REAL: float | None = None

# Approximate specific energy: kWh per m³ pumped.
# Calibrated so that ~12 000 m³/h all day gives ~18–20 MWh/day.
SPECIFIC_KWH_PER_M3 = 0.065

# Level threshold used for generating "flush start" events
FLUSH_LEVEL_M = 3.0


def _build_minute_states(sim_days: int = 7) -> List[Dict[str, Any]]:
    """
    Build an array of 1-minute states for exactly `sim_days` days from
    backend/output/cleaned_data.csv and the HappyWaste controller.

    Derived metrics per pump:
      - runtimeHours: cumulative ON-time in hours
      - stepsSinceSwitch: number of 15-minute blocks since last state change
      - startsToday: ON-transitions since start of simulated day
      - totalStarts: ON-transitions over full simulated history

    Derived metrics per day:
      - energyTodayMWh: ∫ F2 * SPECIFIC_KWH_PER_M3 dt   (MWh)
      - costTodayEUR:  energyTodayMWh * 1000 * price(EUR/kWh)  [using CURRENT price]

    Also derives per-frame "events" for the UI.
    """
    cleaned_csv = BACKEND_ROOT / "output" / "cleaned_data.csv"
    if not cleaned_csv.exists():
        raise RuntimeError(f"cleaned_data.csv not found at {cleaned_csv}")

    df = pd.read_csv(cleaned_csv, parse_dates=["Time stamp"])

    # If the prices are in cents, convert to EUR/kWh
    if df["Electricity price 2: normal"].max() > 10:
        df["Electricity price 2: normal"] = df["Electricity price 2: normal"] / 100.0

    # Run controller for sim_days
    results_df, controller_cost, sys = simulate_controller(
        df, price_col="Electricity price 2: normal", max_days=sim_days
    )

    # Resample controller output to 1-minute resolution
    res = results_df.copy()
    res = res.set_index("Time stamp")
    minute_df = res.resample("1T").nearest()  # 1-minute grid
    minute_df = minute_df.sort_index()

    if minute_df.empty:
        raise RuntimeError("Resampled controller output is empty")

    # Map internal pump indices to the 8 tiles in your UI
    front_pump_ids = ["1.1", "2.1", "1.2", "2.2", "1.3", "2.3", "1.5", "2.4"]
    n_pumps = len(front_pump_ids)

    # --- Per-pump state trackers --------------------------------------
    runtime_hours: List[float] = [0.0 for _ in range(n_pumps)]
    last_state: List[bool] = [False for _ in range(n_pumps)]
    first_ts = minute_df.index[0]
    last_switch_ts: List[pd.Timestamp] = [first_ts for _ in range(n_pumps)]
    total_starts: List[int] = [0 for _ in range(n_pumps)]
    starts_today: List[int] = [0 for _ in range(n_pumps)]

    # initialize last_state using first row
    first_row = minute_df.iloc[0]
    for i in range(n_pumps):
        last_state[i] = bool(first_row.get(f"pump{i}_on", 0))

    # trackers for events
    last_level = float(first_row["L1"])
    first_price = float(first_row["price"])
    last_band = "Low" if first_price * 100.0 <= 10.0 else "High"
    events: List[Dict[str, Any]] = []
    event_counter = 0

    prev_ts = first_ts

    # --- Per-day energy tracker ---------------------------------------
    current_day = first_ts.date()
    energy_today_mwh: float = 0.0

    states: List[Dict[str, Any]] = []

    for ts, row in minute_df.iterrows():
        # elapsed time since previous frame (in minutes/hours of SIMULATION)
        delta_minutes = max(0.0, (ts - prev_ts).total_seconds() / 60.0)
        delta_hours = delta_minutes / 60.0
        prev_ts = ts

        # handle simulated "new day": reset daily totals and per-pump starts
        day = ts.date()
        if day != current_day:
            current_day = day
            energy_today_mwh = 0.0
            starts_today = [0 for _ in range(n_pumps)]

        level = float(row["L1"])
        F1 = float(row["F1"])
        F2 = float(row["F2_ctrl"])
        price = float(row["price"])  # EUR/kWh
        volume = float(height_to_volume(level))

        # flushedToday from HappyWaste (per simulated day)
        flushed_today_flag = bool(row.get("flushedToday", False))

        pumps: List[Dict[str, Any]] = []
        total_power_kw = 0.0

        # --- Events: level-based "flush start" -------------------------
        if last_level < FLUSH_LEVEL_M <= level:
            event_counter += 1
            events.append(
                {
                    "id": f"e{event_counter}",
                    "timestamp": ts.isoformat(),
                    "time": ts.strftime("%H:%M"),
                    "text": f"Level reached {FLUSH_LEVEL_M:.1f} m (flush start)",
                }
            )

        # --- Events: tariff band change --------------------------------
        price_cents = price * 100.0
        band = "Low" if price_cents <= 10.0 else "High"
        if band != last_band:
            event_counter += 1
            events.append(
                {
                    "id": f"e{event_counter}",
                    "timestamp": ts.isoformat(),
                    "time": ts.strftime("%H:%M"),
                    "text": f"Tariff band changed: {last_band} \u2192 {band}",
                }
            )
            last_band = band

        # --- Per-pump loop + pump events -------------------------------
        for i, pid in enumerate(front_pump_ids):
            is_on = bool(row.get(f"pump{i}_on", 0))
            prev_is_on = last_state[i]
            freq = float(row.get(f"pump{i}_freq", 0.0) or 0.0)
            flow = float(row.get(f"pump{i}_flow", 0.0) or 0.0)
            is_small = bool(row.get(f"pump{i}_is_small", i < 2))
            kind = "small" if is_small else "big"

            # Runtime accumulation: only add while ON
            if is_on:
                runtime_hours[i] += delta_hours

            # Detect state change (ON <-> OFF) and update last_switch_ts, starts
            if is_on != prev_is_on:
                last_state[i] = is_on
                last_switch_ts[i] = ts
                if is_on:
                    total_starts[i] += 1
                    starts_today[i] += 1
                    # Pump ON event
                    event_counter += 1
                    events.append(
                        {
                            "id": f"e{event_counter}",
                            "timestamp": ts.isoformat(),
                            "time": ts.strftime("%H:%M"),
                            "text": f"Pump {pid} ON (AI)",
                        }
                    )
                else:
                    # Pump OFF event (kept for context)
                    event_counter += 1
                    events.append(
                        {
                            "id": f"e{event_counter}",
                            "timestamp": ts.isoformat(),
                            "time": ts.strftime("%H:%M"),
                            "text": f"Pump {pid} OFF",
                        }
                    )

            # Minutes since last switch
            minutes_since_switch = max(
                0.0, (ts - last_switch_ts[i]).total_seconds() / 60.0
            )
            steps_since_switch = int(minutes_since_switch // 15.0)

            nominal_power = 180.0 if is_small else 360.0
            power_kw = (freq / 50.0) * nominal_power if is_on else 0.0
            total_power_kw += power_kw

            pumps.append(
                {
                    "id": pid,
                    "kind": kind,
                    "isOn": is_on,
                    "freqHz": round(freq, 1),
                    "flowM3h": round(flow, 1),
                    "powerKw": round(power_kw, 1),
                    "runtimeHours": round(runtime_hours[i], 2),
                    "stepsSinceSwitch": steps_since_switch,
                    "startsToday": starts_today[i],
                    "totalStarts": total_starts[i],
                    # Still demo placeholders for service:
                    "hoursToService": 100.0,
                    "healthStatus": "ok",
                }
            )

        # Per-day energy – based on pumped volume F2 from model
        # pumped_m3 = F2[m3/h] * dt[h]
        if delta_hours > 0.0 and F2 > 0.0:
            pumped_m3 = F2 * delta_hours
            energy_kwh_inc = pumped_m3 * SPECIFIC_KWH_PER_M3
            energy_mwh_inc = energy_kwh_inc / 1000.0
            energy_today_mwh += energy_mwh_inc

        # Cost today: EnergyTodayMWh * 1000 * current price (EUR/kWh)
        cost_today_eur = energy_today_mwh * 1000.0 * price

        # Last-level update for next iteration
        last_level = level

        # Take last 4 events as "recent" snapshot for this frame
        recent_events = events[-4:]

        state: Dict[str, Any] = {
            "timestamp": ts.isoformat(),
            "L1": level,
            "volume": volume,
            "F1": F1,
            "F2": F2,
            "price": price,
            "pumps": pumps,
            "energyTodayMWh": round(energy_today_mwh, 3),
            "costTodayEUR": round(cost_today_eur, 2),
            "flushedToday": flushed_today_flag,
            "events": [
                {"id": e["id"], "time": e["time"], "text": e["text"]}
                for e in recent_events
            ],
        }
        states.append(state)

    return states


@app.on_event("startup")
def _on_startup() -> None:
    """
    On server startup:
    - simulate 7 days
    - compress to 1-minute states
    - store in memory
    - start a demo clock where:
        1 real second = 10 simulated minutes (10 × 1-minute frames)
    """
    global MINUTE_STATES, DEMO_START_REAL
    MINUTE_STATES = _build_minute_states(sim_days=7)
    DEMO_START_REAL = time.time()
    print(f"[backend] Loaded {len(MINUTE_STATES)} 1-minute samples for 7 days.")


@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    """
    Demo endpoint:
    - 1 frame = 1 simulated minute
    - 10 frames per real second
    - loops over the precomputed 7 days forever
    """
    if not MINUTE_STATES:
        raise HTTPException(status_code=500, detail="No simulation data loaded")

    if DEMO_START_REAL is None:
        idx = 0
        frame = 0
    else:
        elapsed = time.time() - DEMO_START_REAL  # real seconds
        frames_per_second = 10.0
        frame = int(elapsed * frames_per_second)  # frame index in "minutes"
        idx = frame % len(MINUTE_STATES)

    state = MINUTE_STATES[idx]
    loop_length = len(MINUTE_STATES)

    # Build dynamic 1h plan using future simulated states
    offsets = [0, 15, 30, 45]  # minutes ahead
    labels = ["Now", "+15 min", "+30 min", "+45 min"]
    plan: List[Dict[str, Any]] = []

    for label, offset in zip(labels, offsets):
        plan_idx = (idx + offset) % loop_length
        s = MINUTE_STATES[plan_idx]

        pumps = s.get("pumps", [])
        big_on = sum(1 for p in pumps if p.get("kind") == "big" and p.get("isOn"))
        small_on = sum(1 for p in pumps if p.get("kind") == "small" and p.get("isOn"))

        F2 = float(s.get("F2", 0.0))
        price = float(s.get("price", 0.0))  # EUR/kWh

        # Approximate cost for that 15-minute slot
        slot_hours = 0.25  # 15 minutes
        pumped_m3 = max(0.0, F2) * slot_hours
        energy_kwh = pumped_m3 * SPECIFIC_KWH_PER_M3
        slot_cost_eur = energy_kwh * price

        plan.append(
            {
                "label": label,
                "offsetMinutes": offset,
                "flowM3h": round(F2, 1),
                "bigOn": big_on,
                "smallOn": small_on,
                "slotCostEUR": round(slot_cost_eur, 2),
                "current": offset == 0,
            }
        )

    return {
        "index": idx,
        "frame": frame,
        "loopLength": loop_length,
        "now": time.time(),
        "plan": plan,
        **state,
    }
