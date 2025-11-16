# Junction-2025-HappyWaste
# Blominmäki Pumping Station Demo

End-to-end demo for experimenting with AI-assisted control of a wastewater pumping station.

This repo is suitable for a hackathon team to:

- Run a realistic control simulation on real data.
- Build UI features on top of a clean API.
- Prototype optimisation, alerting, or analytics ideas without touching real hardware.

---

## 1. High-Level Overview

### 1.1 What This Project Does

- Simulates a real pumping station (Blominmäki) over 7 days using historic data.
- Exposes the simulated state as an HTTP API via FastAPI.
- Renders a live dashboard with React showing:
  - Tunnel level and volume.
  - Inflow and outflow.
  - 8 pumps with states, runtimes, and starts.
  - Instant power and big/small pump split.
  - A dynamic “Next 1 hour” plan from the future simulated states.
  - Recent events: pump on/off, flush start, tariff band changes.

**Time scaling:**

- 1 real second = 10 simulated minutes.

### 1.2 Who This Is For

Hackathon teams working on:

- Industrial AI / optimisation.
- Energy efficiency.
- Real-time dashboards.
- Digital twins.

You do not need domain expertise in wastewater treatment to work with this code. The controller and geometry are already implemented.

---

## 2. Repository Structure

**Assumed layout:**

```text
backend/
  main.py                 # FastAPI app exposing /api/state
  HappyWaste.py           # Controller logic, geometry conversion, simulation
  output/
    cleaned_data.csv      # Historic input data (time series)
  raw_data/
    Volume of tunnel vs level Blominmäki.xlsx

frontend/
  src/
    index.jsx             # Main React dashboard
    global.css            # Styling
  public/
  package.json


3. Backend: FastAPI Simulation API
3.1 Entry Point: backend/main.py

Responsibilities:

Precompute a 7-day simulation at startup.

Downsample to 1-minute resolution.

Derive metrics per minute:

Tunnel state: level, volume, inflow, outflow, price.

Per-pump state: on/off, frequency, flow, power, runtimes, starts.

Daily energy and approximate daily cost.

Recent events.

A dynamic 1-hour plan from future frames.

Expose state via GET /api/state.

Key globals (conceptual values):

MINUTE_STATES: list of per-minute state dictionaries.

DEMO_START_REAL: real-world UNIX timestamp when the demo starts.

SPECIFIC_KWH_PER_M3 = 0.065: approximate kWh per cubic meter pumped.

FLUSH_LEVEL_M = 3.0: level threshold (meters) for flush-start events.

3.1.1 Startup Flow

On startup:

build_minute_states(sim_days=7) is called.

The resulting list (one state per simulated minute) is stored in MINUTE_STATES.

DEMO_START_REAL is set to the current real-world time.

3.1.2 Time Scaling

At request time:

Compute elapsed = now - DEMO_START_REAL in seconds.

Use frames_per_second = 10.0 (10 simulated minutes per real second).

Compute frame = int(elapsed * frames_per_second).

Compute idx = frame % len(MINUTE_STATES) so the 7-day loop repeats forever.

Result:

Each frame corresponds to one simulated minute.

10 frames per real second → 10 simulated minutes per real second.

3.2 Simulation Builder: build_minute_states(sim_days: int = 7)

This function constructs a per-minute state array for exactly sim_days of simulated time.

Pipeline:

3.2.1 Load CSV

File: backend/output/cleaned_data.csv.

Required columns (partial list; controller expects more):

Time stamp

Inflow to tunnel F1

Sum of pumped flow to WWTP F2

Electricity price 2: normal

Pump columns as used by HappyWaste.system_from_row(...).

Price handling:

If Electricity price 2: normal ever exceeds 10, the column is divided by 100 to convert from cents to EUR.

3.2.2 Run Controller

Call simulate_controller(...) with:

Input dataframe.

price_col = "Electricity price 2: normal".

max_days = sim_days.

Returns:

results_df — time series with controller outputs.

controller_cost — aggregate cost metric.

sys — final system object.

3.2.3 Resample to 1-Minute

Set time index to Time stamp.

Resample at 1-minute resolution using nearest neighbor.

Sort by timestamp.

3.2.4 Pump Mapping

Internal pump indices are mapped to UI IDs:

front_pump_ids = ["1.1", "2.1", "1.2", "2.2", "1.3", "2.3", "1.5", "2.4"]


For each pump index i, input data per minute includes:

pump{i}_on

pump{i}_freq

pump{i}_flow

pump{i}_is_small

Tracked over time:

runtime_hours[i] — increments when the pump is on.

last_switch_ts[i] — timestamp of last ON/OFF change.

total_starts[i] — total ON transitions over the simulation.

starts_today[i] — ON transitions within the current simulated day.

Computed power per pump:

Nominal power is about:

180 kW for small pumps at 50 Hz.

360 kW for big pumps at 50 Hz.

Instant power (kW) scales linearly with frequency when the pump is on; otherwise zero.

3.2.5 Daily Energy and Cost

For each minute:

Outflow from controller: F2_ctrl (m³/h).

Convert to pumped volume:

delta_hours = timestep in hours.

pumped_m3 = F2_ctrl * delta_hours.

Energy increment:

energy_kwh_inc = pumped_m3 * SPECIFIC_KWH_PER_M3.

Convert to MWh and accumulate into energy_today_mwh.

Cost:

price is in EUR/kWh.

cost_today_eur = energy_today_mwh * 1000 * price.

New simulated day:

When calendar day changes:

Reset energy_today_mwh to 0.

Reset starts_today for all pumps to 0.

These daily values feed backend metrics; the UI card shows only instant power.

3.2.6 Events Generation

Events are accumulated as we iterate through minute frames.

Event types:

Flush start

Triggered when level L1 crosses FLUSH_LEVEL_M from below.

Adds an event with text like:

Level reached 3.0 m (flush start).

Tariff band change

price_cents = price * 100.

Band:

≤ 10 c/kWh → "Low".

> 10 c/kWh → "High".

When band changes from the previous value, add an event:

Tariff band changed: High → Low or Low → High.

Pump ON

When a pump transitions from OFF to ON.

Text: Pump <id> ON (AI).

Pump OFF

When a pump transitions from ON to OFF.

Text: Pump <id> OFF.

For each frame, only the last 4 events are attached to the state:

events = [
  { "id": "...", "time": "...", "text": "..." },
  ...
]


(up to 4 items).

3.2.7 State Object Per Minute

Per-minute state includes at least:

timestamp — ISO string.

L1 — tunnel level.

volume — tunnel volume (via height_to_volume).

F1 — inflow.

F2 — outflow.

price — electricity price (EUR/kWh).

pumps — list of 8 pump dictionaries.

energyTodayMWh — rounded daily energy consumption so far.

costTodayEUR — rounded daily cost estimate.

flushedToday — flag (currently placeholder).

events — short list of recent events (id, time, text).

All of these states are stored in MINUTE_STATES.

3.3 API: GET /api/state

Handler get_state():

Computes the current index into MINUTE_STATES based on real time and the time-scaling logic.

Builds a dynamic “Next 1 h” plan by looking ahead into MINUTE_STATES.

Returns:

Base state (current frame).

Derived plan array.

Plan creation:

Offsets: 0, 15, 30, 45 minutes.

Labels: "Now", "+15 min", "+30 min", "+45 min".

For each offset:

Compute future_index = (current_index + offset) % loop_length.

Read that state:

Count big and small pumps that are on.

Read F2 (flow).

Read price.

Assume 0.25 hours (15 minutes) per slot:

pumped_m3 = max(F2, 0) * 0.25.

energy_kwh = pumped_m3 * SPECIFIC_KWH_PER_M3.

slot_cost_eur = energy_kwh * price.

Build plan entry with:

label

offsetMinutes

flowM3h

bigOn

smallOn

slotCostEUR

current flag (for offset = 0).

Response JSON (conceptually) includes:

index — index in MINUTE_STATES.

frame — global frame counter.

loopLength — total number of minute states (typically 10_080 for 7 days).

now — current real time (UNIX seconds).

plan — as described above.

Current state fields:

timestamp

L1

volume

F1

F2

price

pumps

energyTodayMWh

costTodayEUR

flushedToday

events

4. Controller and Geometry (backend/HappyWaste.py)
4.1 system Class

Represents the overall pumping system.

Tracks:

List of Pump objects.

Level L1.

Flows:

F1 — inflow.

F2 — current outflow.

Target outflow F2_target.

Electricity price costperkw.

Simulation time sim_time, current day, flush state, etc.

Key behaviours:

Inflow smoothing:

Maintains a smoothed inflow signal F1_slow with a time constant.

Setpoint logic:

Base nominal level L1p_base (≈ 2.2 m).

Adjusts target level with price: higher prices → slightly lower target level (cheaper to pump later).

Flush and prep windows:

During flush window, aims near flush_threshold to clear the tunnel.

Before flush, ramps level from nominal down towards ≈ 1.0 m to create buffer volume.

PID control:

PID runs on the error (L1 - L1p) to adjust a base outflow F2_base.

Gains tuned to work with feed-forward terms.

Flush enforcement:

During flush, uses volume calculations to ensure enough volume is removed during the flush window.

Safety:

Guarantees L1 stays above a safe minimum using geometric predictions and volume constraints.

Low-level dispatcher:

Enumerates pump on/off masks and chooses a combination (mask + frequency) that:

Meets the flow target within pump constraints.

Respects minimum switch times.

Minimises runtime imbalance (load sharing) and penalties on excessive switching.

Safe-min override:

After initial F2 selection, predicts future volume.

If predicted volume would fall below a safe minimum, scales down flows.

You can treat this class as a black box and only tweak it if you want to experiment with control strategies.

4.2 Pump Class

Encapsulates frequency–flow conversion for small and big pumps.

conv_to_flow(frequency):

Uses a pump curve to map frequency in Hz to flow in m³/h.

conv_to_freq(flow):

Approximate inverse mapping from desired flow to frequency.

4.3 Geometry Functions

Geometry is loaded from:

raw_data/Volume of tunnel vs level Blominmäki.xlsx, sheet Taul1.

Two arrays:

_LEVELS (meters).

_VOLUMES (cubic meters).

Functions:

height_to_volume(level):

Interpolates volume given a level.

volume_to_height(volume):

Interpolates level given a volume.

Both use linear interpolation over the tabulated data.

4.4 simulate_controller(...)

High-level behaviour:

Takes a dataframe with historic data.

Steps through time in sub-intervals (default 60 seconds).

For each sub-step:

Gradually adjusts inflow, price, and (optionally) weather from the current row towards the next row.

Calls system.step(dt_sub).

Integrates volume and keeps L1 within safe bounds.

Accumulates controller cost.

Outputs:

results_df:

Time series of:

L1

F1

F2_ctrl

F2_target

price

Pump fields pump{i}_... (on/off, frequency, flow, etc.).

controller_cost:

Aggregated metric for control performance/cost.

sys:

Final system object.

There is also evaluate_results(...) to compare baseline vs controller performance over the dataset.

5. Frontend: React Dashboard (frontend/src/index.jsx)

Single-page dashboard built with React.

5.1 Data Flow

State is pulled from GET /api/state roughly every 100 ms (10 Hz).

Key front-end state variables:

mode — e.g. "auto".

activeTab — one of overview, trends, maintenance, config, logs.

pumps — current pump list.

price — current electricity price.

alarms — list of active alarms.

events — recent events.

now — current browser time.

lastTelemetryTs — timestamp of last successful telemetry update.

level — tunnel level.

volume — tunnel volume.

inflowM3h — inflow.

flushedToday — boolean.

plan — Next 1 h plan data.

Side effects:

now is updated every second for freshness labels.

A polling effect fetches /api/state about every 100 ms.

If backend data is missing or invalid, the UI falls back to mock data (mockPumps, mockEvents).

5.2 Tariff Logic

Convert price to cents: priceCents = price * 100.

Cheap tariff condition: priceCents ≤ 10 (≤ 10 c/kWh).

tariffLabel:

"Low tariff" when cheap.

"High tariff" otherwise.

Displayed as a chip in the top-right bar, styled via global.css.

5.3 Data Freshness

Compute age in seconds:

ageSec = (now - lastTelemetryTs) / 1000.

Freshness classification:

< 10 s → "live".

Between 10 s and 60 s → "lagging".

> 60 s → "stale".

Used to:

Show “Data live/lagging/stale” label.

Change status banner color and text.

Influence “Station state: Normal/Degraded”.

5.4 Main UI Structure

Main layout: three columns in .main-grid.

Left column:

Tunnel level card.

Flow balance card.

Energy card.

Next 1 h plan card.

Middle column:

Pumps grid (2 rows × 4 columns).

Right column:

Status & communications.

Active alarms.

Recent events (scrollable list).

Tabs across the top:

overview

trends

maintenance

config

logs

Currently only overview is fully implemented; others have placeholders.

5.5 Card Details
Tunnel Level

Displays:

Tunnel level (L1) in meters.

Volume (volume) in cubic meters.

Simple predicted levels:

At +15 minutes: approximately level - 0.2 (clamped at 0).

At +60 minutes: approximately level - 0.4 (clamped at 0).

Clicking this card switches to the trends tab.

Flow Balance

Inflow F1: taken directly from backend.

Outflow F2: computed on the frontend as the sum of flowM3h for all pumps where isOn is true.

Displays a basic bar visualising “safe / warn / alarm” segments.

Energy

Shows only instant power:

Sum of powerKw for pumps that are currently on.

Displays a breakdown of power share:

Big pumps vs small pumps.

Card title: Energy.

No “Energy today” or “Cost today” metrics on the UI.

Next 1 h Plan

Uses the plan array from backend.

If plan is present:

Four chips: Now, +15 min, +30 min, +45 min.

Each chip shows:

Flow in m³/h.

Number of big and small pumps on.

Approximate slot cost in EUR.

If plan is absent:

Fallback plan shows only current state with zero cost.

Slots have keys like:

label

offsetMinutes

flowM3h

bigOn

smallOn

slotCostEUR

current (boolean indicating the active slot).

Pumps

Each pump tile shows:

ID (e.g. 1.3).

Size: big or small.

State: ON, OFF, or OOS (out of service).

Frequency (Hz), flow (m³/h), and power (kW).

Runtime and time since last switch (in minutes).

Starts today and total starts.

Hours to next service.

Service badges based on healthStatus.

Interaction:

Clicking a pump tile switches the active tab to maintenance and selects that pump for details.

Status & Comms

Shows:

Overall station state: Normal or Degraded based on freshness and telemetry.

Last update time and age label.

Telemetry status: Live, Lagging, or Stale.

SCADA link status: fixed as OK.

Tariff information:

Price and severity (Low/High).

System status chips, e.g.:

Level range.

Wear balancing.

Switching behaviour.

F2 limit.

Active Alarms

If backend provides alarms, they are listed with severity styling.

If not, a default row indicates No active alarms or warnings.

5.6 Recent Events (Scrollable)

Events:

Pulled from backend as data.events.

If backend has none, fall back to mockEvents.

Display:

.events-list container with:

max-height around 220 px.

overflow-y: auto for vertical scrolling.

Each event row:

Clickable; clicking can open logs for context (type: "event", given id).

Shows:

Event time (e.g. 16:21).

Event text.

Examples of event texts:

Pump 1.3 ON (AI).

Pump 2.3 OFF.

Level reached X.X m (flush start).

Tariff band changed: High → Low or Low → High.

6. Styling (frontend/src/global.css)

Key elements:

Global reset and base font stack.

Sticky top bar with:

Tabs.

Mode toggle (e.g. auto/manual).

Three-column layout via .main-grid.

Cards:

Rounded corners (border-radius around 18 px).

Box-shadow similar to:

0 16px 40px rgba(15, 23, 42, 0.06) for depth.

Pump tiles:

Arranged in .pump-grid.

Fixed height to prevent layout jumps as content changes.

Color-coded chips:

Tariff chips:

Green/yellow for low/high tariff.

Status dots:

Green/orange/red depending on status.

Alarm severity:

Different colors/tags for warning/error/critical.

Events list:

.events-list configured for vertical scrolling.

Scrollable list of compact event rows.

The CSS file can be extended or modified freely without changing backend or React logic.

7. Running the Project
7.1 Backend Setup

Requirements:

Python 3.10+ recommended.

From the backend/ directory:

Create and activate virtual environment:

python -m venv .venv
# On Unix
source .venv/bin/activate
# On Windows
.venv\Scripts\activate


Install dependencies:

pip install fastapi uvicorn pandas numpy openpyxl


Check that required files exist:

output/cleaned_data.csv with necessary columns.

raw_data/Volume of tunnel vs level Blominmäki.xlsx with correct structure.

Run FastAPI server:

uvicorn main:app --reload --host 127.0.0.1 --port 8000


Sanity check:

curl http://127.0.0.1:8000/api/state


Response should be JSON with keys like L1, F1, F2, pumps, plan, events.

7.2 Frontend Setup

Requirements:

Node.js 18+ recommended.

npm or yarn.

From the frontend/ directory:

Install dependencies:

npm install
# or
yarn install


Run dev server (script name may differ by tooling):

npm start
# or
npm run dev
