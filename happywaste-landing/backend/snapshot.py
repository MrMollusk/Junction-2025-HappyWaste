import json
from pathlib import Path
from datetime import timezone
import pandas as pd


def make_snapshot():
  """
  Read the latest row from cleaned_data.csv and emit a JSON snapshot
  compatible with the frontend dashboard.
  """
  root = Path(__file__).resolve().parents[1]
  csv_path = root / "output" / "cleaned_data.csv"

  df = pd.read_csv(csv_path, parse_dates=["Time stamp"])
  if df.empty:
    return {"error": "no_data"}

  df = df.sort_values("Time stamp")
  row = df.iloc[-1]

  ts = row["Time stamp"]
  if ts.tzinfo is None:
    ts = ts.replace(tzinfo=timezone.utc)

  # F1 is stored as m³ / 15 min in the cleaned data → convert to m³/h
  F1_15 = float(row.get("Inflow to tunnel F1", 0.0))
  F1_m3h = F1_15 * 4.0

  F2 = float(row.get("Sum of pumped flow to WWTP F2", 0.0))
  L1 = float(row.get("Water level in tunnel L2", 0.0))
  price = float(row.get("Electricity price 2: normal", 0.0))

  pump_ids = ["1.1", "2.1", "1.2", "2.2", "1.3", "2.3", "1.5", "2.4"]
  small_ids = {"1.1", "2.1"}

  pumps = []
  for pid in pump_ids:
    freq_col = f"Pump frequency {pid}"
    flow_col = f"Pump flow {pid}"
    power_col = f"Pump power {pid}"

    freq = float(row.get(freq_col, 0.0))
    flow = float(row.get(flow_col, 0.0))
    power = float(row.get(power_col, 0.0))

    is_on = freq > 0.1

    pump = {
      "id": pid,
      "kind": "small" if pid in small_ids else "big",
      "isOn": bool(is_on),
      "freqHz": freq,
      "flowM3h": flow,
      "powerKw": power,
      # no lifecycle tracking yet – keep UI happy with placeholders
      "runtimeHours": 0.0,
      "stepsSinceSwitch": 0,
      "startsToday": 0,
      "totalStarts": 0,
      "hoursToService": 0,
      "healthStatus": "ok",
    }
    pumps.append(pump)

  snapshot = {
    "timestamp": ts.isoformat(),
    "F1": F1_m3h,
    "F2": F2,
    "L1": L1,
    "price": price,
    "pumps": pumps,
    # optional fields for later
    "flushedToday": False,
  }
  return snapshot


def main():
  snap = make_snapshot()
  print(json.dumps(snap, default=float))


if __name__ == "__main__":
  main()
