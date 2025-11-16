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
- The 7-day simulation loops forever.

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
