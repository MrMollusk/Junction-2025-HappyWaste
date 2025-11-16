// index.jsx
import { useState, useEffect } from 'react';

const mockPumps = [
  {
    id: '1.1',
    kind: 'small',
    isOn: true,
    freqHz: 49.0,
    flowM3h: 2800,
    powerKw: 180,
    runtimeHours: 120,
    stepsSinceSwitch: 10,
    startsToday: 18,
    totalStarts: 3540,
    hoursToService: 120,
    healthStatus: 'ok'
  },
  {
    id: '2.1',
    kind: 'small',
    isOn: false,
    freqHz: 0.0,
    flowM3h: 0,
    powerKw: 0,
    runtimeHours: 118,
    stepsSinceSwitch: 20,
    startsToday: 12,
    totalStarts: 3400,
    hoursToService: 40,
    healthStatus: 'due'
  },
  {
    id: '1.2',
    kind: 'big',
    isOn: true,
    freqHz: 49.1,
    flowM3h: 3200,
    powerKw: 360,
    runtimeHours: 230,
    stepsSinceSwitch: 5,
    startsToday: 10,
    totalStarts: 4100,
    hoursToService: 200,
    healthStatus: 'ok'
  },
  {
    id: '2.2',
    kind: 'big',
    isOn: true,
    freqHz: 49.1,
    flowM3h: 3200,
    powerKw: 360,
    runtimeHours: 220,
    stepsSinceSwitch: 6,
    startsToday: 9,
    totalStarts: 4050,
    hoursToService: 80,
    healthStatus: 'ok'
  },
  {
    id: '1.3',
    kind: 'big',
    isOn: true,
    freqHz: 49.1,
    flowM3h: 3200,
    powerKw: 360,
    runtimeHours: 215,
    stepsSinceSwitch: 6,
    startsToday: 15,
    totalStarts: 3980,
    hoursToService: 24,
    healthStatus: 'due'
  },
  {
    id: '2.3',
    kind: 'big',
    isOn: false,
    freqHz: 0.0,
    flowM3h: 0,
    powerKw: 0,
    runtimeHours: 210,
    stepsSinceSwitch: 18,
    startsToday: 0,
    totalStarts: 3800,
    hoursToService: 0,
    healthStatus: 'out'
  },
  {
    id: '1.5',
    kind: 'big',
    isOn: false,
    freqHz: 0.0,
    flowM3h: 0,
    powerKw: 0,
    runtimeHours: 205,
    stepsSinceSwitch: 18,
    startsToday: 3,
    totalStarts: 3650,
    hoursToService: 60,
    healthStatus: 'ok'
  },
  {
    id: '2.4',
    kind: 'big',
    isOn: false,
    freqHz: 0.0,
    flowM3h: 0,
    powerKw: 0,
    runtimeHours: 205,
    stepsSinceSwitch: 18,
    startsToday: 1,
    totalStarts: 3600,
    hoursToService: 10,
    healthStatus: 'due'
  }
];

// No real alarms, we keep card space with an "empty" row
const mockAlarms = [];

// Fallback events in case backend is not available yet
const mockEvents = [
  { id: 'e1', time: '16:21', text: 'Pump 1.3 ON (AI)' },
  { id: 'e2', time: '16:18', text: 'Level reached 3.0 m (flush start)' },
  { id: 'e3', time: '16:10', text: 'Pump 2.3 TRIP cleared' },
  { id: 'e4', time: '15:55', text: 'Tariff band changed: High → Normal' }
];

function formatAge(seconds) {
  if (seconds < 1) return '<1 s ago';
  if (seconds < 60) return `${seconds.toFixed(1)} s ago`;
  const minutes = Math.floor(seconds / 60);
  const rem = Math.floor(seconds % 60);
  if (minutes < 60) return `${minutes} min${rem ? ` ${rem} s` : ''} ago`;
  const hours = Math.floor(minutes / 60);
  return `${hours} h ${minutes % 60} min ago`;
}

function formatSince(nowMs, iso) {
  const diffSec = (nowMs - new Date(iso).getTime()) / 1000;
  if (diffSec < 60) return `${Math.floor(diffSec)} s`;
  const m = Math.floor(diffSec / 60);
  if (m < 60) return `${m} min`;
  const h = Math.floor(m / 60);
  return `${h} h`;
}

export default function Dashboard() {
  const [mode, setMode] = useState('auto');
  const [activeTab, setActiveTab] = useState('overview');
  const [pumps, setPumps] = useState(mockPumps);
  const [price, setPrice] = useState(0.291);
  const [alarms, setAlarms] = useState(mockAlarms);
  const [now, setNow] = useState(Date.now());
  const [lastTelemetryTs, setLastTelemetryTs] = useState(Date.now());
  const [level, setLevel] = useState(3.2);
  const [volume, setVolume] = useState(10955);
  const [inflowM3h, setInflowM3h] = useState(0);
  const [flushedToday, setFlushedToday] = useState(true);
  const [highlightedAlarmId, setHighlightedAlarmId] = useState(
    mockAlarms[0]?.id ?? null
  );
  const [showDecisionDetails, setShowDecisionDetails] = useState(false);
  const [selectedTrend, setSelectedTrend] = useState(null);
  const [selectedLogFilter, setSelectedLogFilter] = useState(null);
  const [selectedPumpId, setSelectedPumpId] = useState(null);
  const [plan, setPlan] = useState([]);
  const [events, setEvents] = useState(mockEvents);

  // price in cents and tariff severity
  const priceCents = price * 100;
  const isCheapTariff = priceCents <= 10; // <= 10 c/kWh = "good"
  const tariffLabel = isCheapTariff ? 'Low tariff' : 'High tariff';
  const tariffSeverityLabel = isCheapTariff ? 'Low' : 'High';

  // Wall clock for "last update" labels
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);

  // High-frequency state polling: 10x per second
  useEffect(() => {
    let cancelled = false;

    async function fetchState() {
      try {
        const res = await fetch('http://127.0.0.1:8000/api/state');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (cancelled) return;

        setLevel(typeof data.L1 === 'number' ? data.L1 : 0);
        setVolume(typeof data.volume === 'number' ? data.volume : 0);
        setInflowM3h(typeof data.F1 === 'number' ? data.F1 : 0);
        setPrice(typeof data.price === 'number' ? data.price : 0.291);

        if (Array.isArray(data.pumps) && data.pumps.length > 0) {
          setPumps(data.pumps);
        } else {
          setPumps(mockPumps);
        }

        if (typeof data.flushedToday === 'boolean') {
          setFlushedToday(data.flushedToday);
        }

        if (Array.isArray(data.plan)) {
          setPlan(data.plan);
        } else {
          setPlan([]);
        }

        // Dynamic recent events from backend
        if (Array.isArray(data.events) && data.events.length > 0) {
          setEvents(data.events);
        } else {
          setEvents(mockEvents);
        }

        setLastTelemetryTs(Date.now());

        // If backend ever sends alarms later, wire them here:
        if (Array.isArray(data.alarms)) {
          setAlarms(data.alarms);
        }
      } catch (err) {
        console.error('Failed to fetch state', err);
      }
    }

    fetchState();
    const intervalId = setInterval(fetchState, 100); // 10 updates / second
    return () => {
      cancelled = true;
      clearInterval(intervalId);
    };
  }, []);

  useEffect(() => {
    if (!highlightedAlarmId) return;
    const id = setTimeout(() => setHighlightedAlarmId(null), 4000);
    return () => clearTimeout(id);
  }, [highlightedAlarmId]);

  const totalFlow = pumps.reduce((s, p) => s + (p.isOn ? p.flowM3h : 0), 0);
  const totalPower = pumps.reduce((s, p) => s + (p.isOn ? p.powerKw : 0), 0);

  const predictedLevel15 = Math.max(0, level - 0.2);
  const predictedLevel60 = Math.max(0, level - 0.4);

  const bigOn = pumps.filter(p => p.kind === 'big' && p.isOn).length;
  const smallOn = pumps.filter(p => p.kind === 'small' && p.isOn).length;

  const fallbackPlan = [
    {
      label: 'Now',
      offsetMinutes: 0,
      flowM3h: totalFlow,
      bigOn,
      smallOn,
      slotCostEUR: 0,
      current: true
    }
  ];

  const bigPower = pumps
    .filter(p => p.kind === 'big' && p.isOn)
    .reduce((s, p) => s + p.powerKw, 0);
  const smallPower = pumps
    .filter(p => p.kind === 'small' && p.isOn)
    .reduce((s, p) => s + p.powerKw, 0);
  const powerTotal = bigPower + smallPower || 1;
  const bigShare = (bigPower / powerTotal) * 100;
  const smallShare = 100 - bigShare;

  const nowStr = new Date().toLocaleString('fi-FI', {
    weekday: 'short',
    hour: '2-digit',
    minute: '2-digit'
  });

  // Data freshness
  const ageSeconds = (now - lastTelemetryTs) / 1000;
  let freshnessStatus = 'live'; // 'live' | 'lagging' | 'stale'
  if (ageSeconds >= 10 && ageSeconds <= 60) freshnessStatus = 'lagging';
  if (ageSeconds > 60) freshnessStatus = 'stale';

  const lastUpdate = new Date(lastTelemetryTs);
  const lastUpdateTimeStr = lastUpdate.toLocaleTimeString('fi-FI', {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit'
  });
  const ageLabel = formatAge(ageSeconds);
  const ageShort = ageLabel.replace(' ago', '');

  // Alarms
  const activeAlarms = alarms.filter(a => !a.clearedAt);
  const sortedActiveAlarms = [...activeAlarms].sort(
    (a, b) => new Date(b.raisedAt) - new Date(a.raisedAt)
  );

  const criticalUnacked = activeAlarms.filter(
    a => !a.acknowledgedBy && a.severity === 'Critical'
  );
  const majorUnacked = activeAlarms.filter(
    a => !a.acknowledgedBy && a.severity === 'Major'
  );

  const criticalCount = criticalUnacked.length;
  const majorCount = majorUnacked.length;

  const hasStale = freshnessStatus === 'stale';
  const hasLagging = freshnessStatus === 'lagging';

  const showStatusBanner =
    criticalCount > 0 || majorCount > 0 || hasStale || hasLagging;

  let bannerVariant = '';
  if (criticalCount > 0 || hasStale) bannerVariant = 'critical';
  else if (majorCount > 0 || hasLagging) bannerVariant = 'warning';

  const severityParts = [];
  if (criticalCount > 0) severityParts.push(`${criticalCount} critical`);
  if (majorCount > 0) severityParts.push(`${majorCount} major`);

  const stateParts = [];
  if (hasStale) stateParts.push(`DATA STALE (${ageShort})`);
  else if (hasLagging) stateParts.push(`DATA LAGGING (${ageShort})`);

  const bannerText = [severityParts.join(', '), stateParts.join(' · ')]
    .filter(Boolean)
    .join(' · ');

  const handleAcknowledge = id => {
    setAlarms(prev =>
      prev.map(a =>
        a.id === id && !a.acknowledgedBy
          ? {
              ...a,
              acknowledgedBy: 'J. Smith',
              acknowledgedAt: new Date().toISOString()
            }
          : a
      )
    );
  };

  const stationStateLabel =
    freshnessStatus === 'live' ? 'Normal' : 'Degraded';
  const stationStateDetail =
    freshnessStatus === 'live'
      ? 'data live'
      : freshnessStatus === 'lagging'
      ? `data lagging ${ageShort}`
      : `data stale ${ageShort}`;

  const telemetryLabel =
    freshnessStatus === 'live'
      ? 'Live'
      : freshnessStatus === 'lagging'
      ? 'Lagging'
      : 'Stale';

  const scadaLabel = 'OK';

  const openTrends = metric => {
    setSelectedTrend(metric);
    setActiveTab('trends');
  };

  const openLogsForContext = ctx => {
    setSelectedLogFilter(ctx);
    setActiveTab('logs');
  };

  const openPumpDetails = pumpId => {
    setSelectedPumpId(pumpId);
    setActiveTab('maintenance');
  };

  const severityCodeMap = {
    Critical: 'C',
    Major: 'M',
    Minor: 'm',
    Info: 'i'
  };

  // Non-overview tabs
  if (activeTab !== 'overview') {
    return (
      <div className="app-root">
        <header className="topbar">
          <div>
            <div className="topbar-title">Blominmäki · Pumping Station</div>
            <div className="topbar-sub-row">
              <span className="topbar-sub">{nowStr}</span>
              <span className="last-update">
                Last update: {lastUpdateTimeStr} ({ageLabel})
              </span>
            </div>
            <div className="station-tabs">
              {['overview', 'trends', 'maintenance', 'config', 'logs'].map(
                tab => (
                  <button
                    key={tab}
                    className={`station-tab ${
                      activeTab === tab ? 'active' : ''
                    }`}
                    onClick={() => setActiveTab(tab)}
                  >
                    {tab.charAt(0).toUpperCase() + tab.slice(1)}
                  </button>
                )
              )}
            </div>
          </div>

          <div className="topbar-right">
            <div className="operator-info">
              Operator: J. Smith · Role: Operator
            </div>
            <div>
              <div className="mode-toggle">
                <button
                  className={`mode-pill ${mode === 'auto' ? 'active' : ''}`}
                  onClick={() => setMode('auto')}
                >
                  Auto
                </button>
                <button
                  className={`mode-pill ${mode === 'manual' ? 'active' : ''}`}
                  onClick={() => setMode('manual')}
                >
                  Manual
                </button>
              </div>
              <div className="mode-note">
                Last mode change: Auto → Manual by J. Smith at 15:40
              </div>
            </div>
            <div className={`price-chip ${isCheapTariff ? 'cheap' : 'expensive'}`}>
              <span className="value">{priceCents.toFixed(1)} c/kWh</span>
              <span>{tariffLabel}</span>
            </div>
          </div>
        </header>

        {showStatusBanner && (
          <div className={`alarm-banner ${bannerVariant || 'critical'}`}>
            <span>⚠ {bannerText || 'DATA STATE'}</span>
            <a href="#alarms-panel">View alarms</a>
          </div>
        )}

        <div className="secondary-tab">
          <div className="card">
            <div className="card-header">
              <div className="card-title">
                {activeTab === 'trends' && 'Trends'}
                {activeTab === 'maintenance' && 'Maintenance'}
                {activeTab === 'config' && 'Config'}
                {activeTab === 'logs' && 'Logs'}
              </div>
              <div className="card-subtitle">
                Blominmäki ·{' '}
                {activeTab.charAt(0).toUpperCase() + activeTab.slice(1)}
              </div>
            </div>
            <div style={{ fontSize: 13, color: '#374151' }}>
              {activeTab === 'trends' && (
                <>
                  Trends view (placeholder).{' '}
                  {selectedTrend &&
                    `Focused metric: ${selectedTrend.toUpperCase()}.`}
                </>
              )}
              {activeTab === 'maintenance' && (
                <>
                  Maintenance view (placeholder).{' '}
                  {selectedPumpId && `Focused pump: ${selectedPumpId}.`}
                </>
              )}
              {activeTab === 'config' && <>Configuration view (placeholder).</>}
              {activeTab === 'logs' && (
                <>
                  Logs view (placeholder).{' '}
                  {selectedLogFilter &&
                    `Context: ${selectedLogFilter.type} ${selectedLogFilter.id}.`}
                </>
              )}
              {!['trends', 'maintenance', 'config', 'logs'].includes(
                activeTab
              ) && <>Non-overview tab.</>}
            </div>
          </div>
        </div>
      </div>
    );
  }

  // Overview
  return (
    <div className="app-root">
      {/* Top bar */}
      <header className="topbar">
        <div>
          <div className="topbar-title">Blominmäki · Pumping Station</div>
          <div className="topbar-sub-row">
            <span className="topbar-sub">{nowStr}</span>
            <span className="last-update">
              Last update: {lastUpdateTimeStr} ({ageLabel})
            </span>
          </div>
          <div className="station-tabs">
            {['overview', 'trends', 'maintenance', 'config', 'logs'].map(
              tab => (
                <button
                  key={tab}
                  className={`station-tab ${
                    activeTab === tab ? 'active' : ''
                  }`}
                  onClick={() => setActiveTab(tab)}
                >
                  {tab.charAt(0).toUpperCase() + tab.slice(1)}
                </button>
              )
            )}
          </div>
        </div>

        <div className="topbar-right">
          <div className="operator-info">
            Operator: J. Smith · Role: Operator
          </div>
          <div>
            <div className="mode-toggle">
              <button
                className={`mode-pill ${mode === 'auto' ? 'active' : ''}`}
                onClick={() => setMode('auto')}
              >
                Auto
              </button>
              <button
                className={`mode-pill ${mode === 'manual' ? 'active' : ''}`}
                onClick={() => setMode('manual')}
              >
                Manual
              </button>
            </div>
            <div className="mode-note">
              Last mode change: Auto → Manual by J. Smith at 15:40
            </div>
          </div>
          <div className={`price-chip ${isCheapTariff ? 'cheap' : 'expensive'}`}>
            <span className="value">{priceCents.toFixed(1)} c/kWh</span>
            <span>{tariffLabel}</span>
          </div>
        </div>
      </header>

      {showStatusBanner && (
        <div className={`alarm-banner ${bannerVariant || 'critical'}`}>
          <span>⚠ {bannerText || 'DATA STATE'}</span>
          <a href="#alarms-panel">View alarms</a>
        </div>
      )}

      {/* Main grid */}
      <main className="main-grid">
        {/* Left column */}
        <section>
          {/* Tunnel card */}
          <div
            className={`card card-clickable data-card ${
              freshnessStatus !== 'live' ? 'data-card-stale' : ''
            }`}
            style={{ marginBottom: 14 }}
            onClick={() => openTrends('level')}
          >
            <div className="card-header">
              <div>
                <div className="card-title">Tunnel level</div>
                <div className="card-subtitle">
                  Blominmäki balancing tunnel
                </div>
              </div>
              {freshnessStatus !== 'live' && (
                <span
                  className={`data-freshness-chip ${freshnessStatus}`}
                >
                  Data {freshnessStatus}
                </span>
              )}
            </div>
            <div className="tunnel-gauge">
              <div className="tunnel-value">
                <span className="level">{level.toFixed(1)} m</span>
                <span className="label">L1 (0–8 m)</span>
              </div>
              <div className="tunnel-meta">
                <div>
                  Volume: <strong>{volume.toLocaleString()} m³</strong>
                </div>
                <div>
                  Predicted in 15 min:{' '}
                  <strong>{predictedLevel15.toFixed(1)} m</strong>
                </div>
                <div>
                  Predicted in 1 h:{' '}
                  <strong>{predictedLevel60.toFixed(1)} m</strong>
                </div>
              </div>
            </div>
            <div className="limit-bar">
              <div className="limit-segment safe" />
              <div className="limit-segment warn" />
              <div className="limit-segment alarm" />
              <div
                className="limit-marker"
                style={{ left: `${Math.min(level / 10, 1) * 100}%` }}
              />
            </div>
            <div className="limit-caption">
              Safe: 0–8 m · Warning: 8–9 m · Alarm: &gt;9 m
            </div>
          </div>

          {/* Flow & balance */}
          <div
            className={`card card-clickable data-card ${
              freshnessStatus !== 'live' ? 'data-card-stale' : ''
            }`}
            style={{ marginBottom: 14 }}
            onClick={() => openTrends('flow')}
          >
            <div className="card-header">
              <div className="card-title">Flow balance</div>
              {freshnessStatus !== 'live' && (
                <span
                  className={`data-freshness-chip ${freshnessStatus}`}
                >
                  Data {freshnessStatus}
                </span>
              )}
            </div>
            <div className="flow-grid">
              <div className="flow-block">
                <label>Inflow F1</label>
                <div className="flow-primary">
                  {inflowM3h.toLocaleString(undefined, {
                    maximumFractionDigits: 0
                  })}{' '}
                  m³ / h
                </div>
                <div
                  style={{
                    fontSize: 12,
                    color: '#4b5563',
                    marginTop: 2
                  }}
                >
                  Forecast stable next 2 h
                </div>
                <div className="limit-bar">
                  <div className="limit-segment safe" />
                  <div className="limit-segment warn" />
                  <div className="limit-segment alarm" />
                  <div
                    className="limit-marker"
                    style={{
                      left: `${Math.min(inflowM3h / 20000, 1) * 100}%`
                    }}
                  />
                </div>
                <div className="limit-caption">
                  Safe: 0–15 000 m³/h · Warning: 15–16 000 · Alarm: &gt;16 000
                </div>
              </div>
              <div className="flow-block">
                <label>Outflow F2</label>
                <div className="flow-primary">
                  {totalFlow.toLocaleString(undefined, {
                    maximumFractionDigits: 0
                  })}{' '}
                  m³ / h
                </div>
                <div
                  style={{
                    fontSize: 12,
                    color: '#4b5563',
                    marginTop: 2
                  }}
                >
                  Commanded: 12 000 m³ / h
                </div>
                <div className="limit-bar">
                  <div className="limit-segment safe" />
                  <div className="limit-segment warn" />
                  <div className="limit-segment alarm" />
                  <div
                    className="limit-marker"
                    style={{
                      left: `${Math.min(totalFlow / 20000, 1) * 100}%`
                    }}
                  />
                </div>
                <div className="limit-caption">
                  Safe: 0–14 000 m³/h · Warning: 14–16 000 · Alarm: &gt;16 000
                </div>
              </div>
            </div>
          </div>

          {/* Energy card (no Energy today / Cost today) */}
          <div
            className={`card card-clickable data-card ${
              freshnessStatus !== 'live' ? 'data-card-stale' : ''
            }`}
            style={{ marginBottom: 14 }}
            onClick={() => openTrends('energy')}
          >
            <div className="card-header">
              <div className="card-title">Energy</div>
              {freshnessStatus !== 'live' && (
                <span
                  className={`data-freshness-chip ${freshnessStatus}`}
                >
                  Data {freshnessStatus}
                </span>
              )}
            </div>
            <div className="cost-metrics">
              <div>
                <span className="label">Instant power</span>
                <br />
                <span className="value">{totalPower.toFixed(0)} kW</span>
              </div>
            </div>
            <div className="cost-bar">
              <div
                className="cost-bar-big"
                style={{ width: `${bigShare}%` }}
              />
              <div
                className="cost-bar-small"
                style={{ width: `${smallShare}%` }}
              />
            </div>
            <div
              style={{
                display: 'flex',
                justifyContent: 'space-between',
                marginTop: 4,
                fontSize: 11,
                color: '#4b5563'
              }}
            >
              <span>Big pumps {bigShare.toFixed(0)}%</span>
              <span>Small pumps {smallShare.toFixed(0)}%</span>
            </div>
          </div>

          {/* Plan + rationale */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Next 1 h plan</div>
            </div>
            <div className="plan-strip">
              {(plan.length ? plan : fallbackPlan).map(slot => (
                <div
                  key={slot.offsetMinutes ?? slot.offset ?? slot.label}
                  className={`plan-chip ${slot.current ? 'current' : ''}`}
                >
                  <label>{slot.label}</label>
                  <div className="plan-main">
                    {slot.flowM3h.toLocaleString(undefined, {
                      maximumFractionDigits: 0
                    })}{' '}
                    m³/h
                  </div>
                  <div>
                    {slot.bigOn} big · {slot.smallOn} small
                  </div>
                  {'slotCostEUR' in slot && slot.slotCostEUR != null ? (
                    <div>~ {slot.slotCostEUR.toFixed(0)} €</div>
                  ) : (
                    <div>~ – €</div>
                  )}
                </div>
              ))}
            </div>
            <div className="why-compact">
              <div>Commanded flow: 12 000 m³/h</div>
              <div>
                Selected pumps: {bigOn} big, {smallOn} small (AI)
              </div>
              <div>Objective: minimise kWh/m³ within wear limits</div>
              <button
                type="button"
                className="link-button"
                onClick={() =>
                  setShowDecisionDetails(prev => !prev)
                }
              >
                {showDecisionDetails ? 'Hide details' : 'Show details'}
              </button>
            </div>
            {showDecisionDetails && (
              <ul className="why-list">
                <li>Commanded flow from high-level model: 12 000 m³/h.</li>
                <li>
                  Selected {bigOn} big pumps, {smallOn} small pumps at ~49 Hz.
                </li>
                <li>Level at {level.toFixed(1)} m, well inside 0–8 m band.</li>
                <li>
                  Electricity price {priceCents.toFixed(1)} c/kWh, minimizing kWh/m³.
                </li>
                <li>
                  Pumps chosen are those with lowest runtime to balance wear.
                </li>
                <li>
                  No pump switched earlier than 2 h since last toggle.
                </li>
              </ul>
            )}
          </div>
        </section>

        {/* Middle column: pumps */}
        <section>
          <div className="card">
            <div className="card-header">
              <div className="card-title">Pumps</div>
              <div className="card-subtitle">
                Big: 6 installed ({bigOn} on) · Small: 2 installed ({smallOn} on)
              </div>
            </div>
            <div className="pump-grid">
              {pumps.map(p => (
                <PumpTile
                  key={p.id}
                  pump={p}
                  mode={mode}
                  onClick={() => openPumpDetails(p.id)}
                />
              ))}
            </div>
          </div>
        </section>

        {/* Right column: status, alarms, events */}
        <section>
          {/* Status card */}
          <div className="card" style={{ marginBottom: 12 }}>
            <div className="card-header">
              <div className="card-title">Status & comms</div>
              <div className="card-subtitle">Data health & links</div>
            </div>
            <div className="status-block">
              <div className="status-line">
                <span className={`status-dot ${freshnessStatus}`} />
                <span>
                  Station state: {stationStateLabel}
                  {stationStateDetail && ` (${stationStateDetail})`}
                </span>
              </div>
              <div className="status-line" style={{ fontSize: 11 }}>
                Last update: {lastUpdateTimeStr} ({ageLabel})
              </div>
              <div className="status-connections">
                <div>Telemetry: {telemetryLabel}</div>
                <div>SCADA link: {scadaLabel}</div>
                <div>
                  Tariff: {priceCents.toFixed(1)} c/kWh ({tariffSeverityLabel})
                </div>
                <div style={{ marginTop: 6 }}>System status:</div>
                <div className="status-chips">
                  <span className="constraint-chip ok">
                    Level safe · 0–8 m
                  </span>
                  <span className="constraint-chip ok">
                    Wear balanced
                  </span>
                  <span className="constraint-chip ok">
                    Switching within limits
                  </span>
                  <span className="constraint-chip ok">
                    F2{' '}
                    {totalFlow.toLocaleString(undefined, {
                      maximumFractionDigits: 0
                    })}{' '}
                    m³/h · limit 16 000
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Active alarms */}
          <div className="card" id="alarms-panel" style={{ marginBottom: 12 }}>
            <div className="card-header">
              <div className="card-title">Active alarms</div>
              <div className="card-subtitle">
                {activeAlarms.length} active ·{' '}
                {activeAlarms.filter(a => !a.acknowledgedBy).length} unack
              </div>
            </div>
            <div className="alarms-table">
              <div className="alarms-header-row">
                <span>Sev</span>
                <span>Asset</span>
                <span>Message</span>
                <span>Since</span>
                <span>Acked by</span>
                <span />
              </div>
              <div className="alarms-body">
                {sortedActiveAlarms.length === 0 ? (
                  <div className="alarm-row empty">
                    <span>
                      <span className="alarm-severity-tag alarm-severity-info">
                        i
                      </span>
                    </span>
                    <span>–</span>
                    <span>No active alarms or warnings</span>
                    <span>–</span>
                    <span>–</span>
                    <span />
                  </div>
                ) : (
                  sortedActiveAlarms.map(alarm => {
                    const severityCode =
                      severityCodeMap[alarm.severity] || '?';
                    return (
                      <div
                        key={alarm.id}
                        className={`alarm-row ${
                          alarm.acknowledgedBy ? 'acked' : 'unacked'
                        } ${highlightedAlarmId === alarm.id ? 'new' : ''}`}
                        onClick={() =>
                          openLogsForContext({
                            type: 'alarm',
                            id: alarm.id
                          })
                        }
                      >
                        <span>
                          <span
                            className={`alarm-severity-tag alarm-severity-${alarm.severity.toLowerCase()}`}
                          >
                            {severityCode}
                          </span>
                        </span>
                        <span>{alarm.assetId}</span>
                        <span>{alarm.message}</span>
                        <span>{formatSince(now, alarm.raisedAt)}</span>
                        <span>{alarm.acknowledgedBy || '-'}</span>
                        <span className="alarm-actions">
                          <button
                            className="link-button"
                            disabled={!!alarm.acknowledgedBy}
                            onClick={e => {
                              e.stopPropagation();
                              handleAcknowledge(alarm.id);
                            }}
                          >
                            Ack
                          </button>
                          <button
                            className="link-button"
                            onClick={e => {
                              e.stopPropagation();
                              openLogsForContext({
                                type: 'alarm',
                                id: alarm.id
                              });
                            }}
                          >
                            Details
                          </button>
                        </span>
                      </div>
                    );
                  })
                )}
              </div>
            </div>
          </div>

          {/* Recent events */}
          <div className="card">
            <div className="card-header">
              <div className="card-title">Recent events</div>
            </div>
            <div className="events-list">
              {events.map(ev => (
                <div
                  key={ev.id}
                  className="event-row"
                  onClick={() =>
                    openLogsForContext({ type: 'event', id: ev.id })
                  }
                >
                  <span className="event-time">{ev.time}</span>
                  <span className="event-text">{ev.text}</span>
                </div>
              ))}
            </div>
          </div>
        </section>
      </main>

      {mode === 'manual' && (
        <div
          style={{
            position: 'fixed',
            bottom: 0,
            left: 0,
            right: 0,
            padding: '8px 20px',
            background: '#111827',
            color: '#e5e7eb',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontSize: 12
          }}
        >
          <span>Manual mode · constraints may not be enforced.</span>
          <span>Return to Auto to hand control back to optimizer.</span>
        </div>
      )}
    </div>
  );
}

function PumpTile({ pump, mode, onClick }) {
  let state = 'off';
  if (pump.healthStatus === 'out') state = 'oos';
  else if (pump.isOn) state = 'on';

  let stateLabel = 'OFF';
  let stateClass = 'status-off';

  if (state === 'on') {
    stateLabel = 'ON';
    stateClass = 'status-on';
  } else if (state === 'oos') {
    stateLabel = 'OOS';
    stateClass = 'status-oos';
  }

  const freqText = pump.isOn ? pump.freqHz.toFixed(1) : '0.0';
  const flowText = pump.isOn ? pump.flowM3h.toFixed(0) : '0';
  const powerText = pump.powerKw.toFixed(0);

  const isAIControlled = mode === 'auto' && pump.healthStatus !== 'out';

  return (
    <div className="pump-tile" onClick={onClick}>
      <div className="pump-left">
        <div className={`pump-avatar ${pump.kind === 'big' ? 'big' : 'small'}`}>
          {pump.id}
        </div>
        <div className="pump-labels">
          <div className="pump-header-row">
            <span className="pump-main">
              {pump.kind === 'big' ? 'Big pump' : 'Small pump'}
            </span>
            {isAIControlled && (
              <span
                className="pump-ai-chip"
                title="Controlled by optimizer"
              >
                AI
              </span>
            )}
          </div>
          <div className="pump-sub">
            runtime {pump.runtimeHours.toFixed(0)} h · last switch{' '}
            {pump.stepsSinceSwitch * 15} min ago
          </div>
          {pump.healthStatus === 'due' && (
            <span className="pump-service-badge health-pill due">
              ⚠ Service due soon
            </span>
          )}
          {pump.healthStatus === 'out' && (
            <span className="pump-service-badge health-pill out">
              ⛔ Out of service
            </span>
          )}
        </div>
      </div>

      <div className="pump-right">
        <div className={`status-chip ${stateClass}`}>{stateLabel}</div>
        <div className="pump-metrics-row">
          {freqText} Hz · {flowText} m³/h · {powerText} kW
        </div>
        <div className="pump-health-row">
          <span>
            Starts today:{' '}
            {pump.startsToday != null ? pump.startsToday : 0}
          </span>
          <span>·</span>
          <span>
            Total starts:{' '}
            {pump.totalStarts != null
              ? pump.totalStarts.toLocaleString()
              : '–'}
          </span>
          <span>·</span>
          <span>
            Next service:{' '}
            {pump.hoursToService != null
              ? `${pump.hoursToService} h`
              : '–'}
          </span>
        </div>
      </div>
    </div>
  );
}
