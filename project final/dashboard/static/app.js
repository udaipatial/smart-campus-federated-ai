// ==========================================================================
// FEDERATED EDGE-AI SMART CAMPUS ENERGY MONITOR — FRONTEND ENGINE v2
// ==========================================================================

'use strict';

// ── State ──────────────────────────────────────────────────────────────────
let telemetryChart    = null;
let metricsChart      = null;
let latencyChart      = null;
let convergenceChart  = null;
let commChart         = null;
const sparklineCharts = {};     // building_id -> Chart

let isStreamingActive   = true;
let streamInterval      = null;
let selectedBuilding    = 'building_A';
let lastAlertSignature  = '';
let alertSignatureExpiry= 0;

// Chart.js defaults
Chart.defaults.font.family      = "'Inter', 'JetBrains Mono', sans-serif";
Chart.defaults.color            = '#64748B';
Chart.defaults.borderColor      = 'rgba(255,255,255,0.05)';
Chart.defaults.plugins.tooltip.enabled = true;

// ── Bootstrap ──────────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    initTabs();
    initModal();
    initStreamControls();
    loadDashboardData();

    // Building selector in chart header
    const selector = document.getElementById('building-selector');
    if (selector) {
        selector.addEventListener('change', e => selectBuildingNode(e.target.value));
    }

    // Target node in injector console
    const injectTarget = document.getElementById('inject-building-target');
    if (injectTarget) {
        injectTarget.addEventListener('change', e => {
            selectedBuilding = e.target.value;
            if (selector) selector.value = selectedBuilding;
            selectBuildingNode(selectedBuilding);
        });
    }
});

// ==========================================================================
// BUILDING NODE SELECTION
// ==========================================================================
function selectBuildingNode(buildingId) {
    selectedBuilding = buildingId;

    // Sync both dropdowns
    const selector     = document.getElementById('building-selector');
    const injectTarget = document.getElementById('inject-building-target');
    if (selector)     selector.value     = buildingId;
    if (injectTarget) injectTarget.value = buildingId;

    // Highlight the active card
    document.querySelectorAll('.building-card').forEach(c => {
        c.classList.toggle('active-selected', c.getAttribute('data-id') === buildingId);
    });

    fetchTelemetryData(buildingId);
}

// ==========================================================================
// TABS
// ==========================================================================
function initTabs() {
    const tabBtns     = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            tabBtns.forEach(b => { b.classList.remove('active'); b.setAttribute('aria-selected', 'false'); });
            tabContents.forEach(c => c.classList.remove('active'));

            btn.classList.add('active');
            btn.setAttribute('aria-selected', 'true');
            const target = document.getElementById(btn.getAttribute('data-tab'));
            if (target) { target.classList.add('active'); target.focus({ preventScroll: true }); }
        });

        btn.addEventListener('keydown', e => {
            if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
            e.preventDefault();
            const btns = [...tabBtns];
            const idx  = btns.indexOf(btn);
            const next = (e.key === 'ArrowRight') ? (idx + 1) % btns.length : (idx - 1 + btns.length) % btns.length;
            btns[next].focus();
            btns[next].click();
        });
    });
}

// ==========================================================================
// DATA LOADING
// ==========================================================================
async function loadDashboardData() {
    try {
        await Promise.all([
            fetchSummary(),
            fetchBuildings(),
            fetchTelemetryData(selectedBuilding),
            fetchAnomalies(),
            fetchFederatedRounds(),
            fetchPrivacyAudit(),
            fetchFigures()
        ]);
    } catch (err) {
        console.error('Dashboard initialization error:', err);
    }
}

// ── Summary / KPIs ─────────────────────────────────────────────────────────
async function fetchSummary() {
    try {
        const res = await fetch('/api/summary');
        if (!res.ok) return;
        const data    = await res.json();
        const metrics = data.metrics_comparison;

        // KPI count-up
        const f1  = parseFloat(metrics['F1 Score']['Federated']);
        const acc = parseFloat(metrics['Accuracy']['Federated']);
        const comm= parseFloat(metrics['Communication Overhead']['Federated']);

        countUp('kpi-f1',   f1,   '%', 1);
        countUp('kpi-acc',  acc,  '%', 2);
        countUp('kpi-comm', comm, ' MB', 3);

        // Animate KPI bars
        animateBar('kpi-f1-bar',  f1  / 100);
        animateBar('kpi-acc-bar', acc / 100);

        // Update header latency from inference latency metric
        const latency = parseFloat(metrics['Inference Latency']['Federated']);
        const latEl = document.getElementById('header-latency');
        if (latEl) latEl.textContent = `~${latency.toFixed(3)} ms LATENCY`;

        // Render comparison table
        renderComparisonTable(metrics);

        // Render comparison charts
        renderMetricsComparisonChart(metrics);
        renderLatencyChart(metrics);
    } catch (err) {
        console.error('Failed to load summary metrics:', err);
    }
}

// Animated count-up
function countUp(elId, target, suffix = '', decimals = 0) {
    const el = document.getElementById(elId);
    if (!el) return;
    const duration = 900;
    const start    = performance.now();
    const step = ts => {
        const progress = Math.min((ts - start) / duration, 1);
        const ease     = 1 - Math.pow(1 - progress, 3);   // cubic ease-out
        el.textContent = (target * ease).toFixed(decimals) + suffix;
        if (progress < 1) requestAnimationFrame(step);
        else { el.textContent = target.toFixed(decimals) + suffix; el.classList.add('counter-animated'); }
    };
    requestAnimationFrame(step);
}

// Animate KPI bar fill
function animateBar(id, ratio) {
    const el = document.getElementById(id);
    if (!el) return;
    // Force reflow
    el.style.transform = 'scaleX(0)';
    el.offsetHeight; // trigger reflow
    el.style.transition = 'transform 1.2s cubic-bezier(0.16,1,0.3,1)';
    el.style.transform = `scaleX(${Math.min(1, ratio)})`;
}

// ── Comparison table ────────────────────────────────────────────────────────
function renderComparisonTable(metrics) {
    const tbody = document.getElementById('comparison-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    const highlightRows = new Set(['F1 Score', 'Accuracy', 'Privacy Preservation', 'Recall']);

    for (const [name, row] of Object.entries(metrics)) {
        const unit  = row.Unit === 'Category' ? '' : ` ${row.Unit}`;
        const tr    = document.createElement('tr');
        if (highlightRows.has(name)) tr.classList.add('row-highlight');

        const delta     = row.Delta;
        const isNumeric = typeof delta === 'number';
        let deltaBadge  = '';
        if (isNumeric) {
            const cls  = delta > 0 ? 'delta-positive' : delta < 0 ? 'delta-negative' : 'delta-neutral';
            const sign = delta > 0 ? '▲ +' : delta < 0 ? '▼ ' : '';
            deltaBadge = `<span class="delta-badge ${cls}">${sign}${delta}${unit}</span>`;
        } else {
            deltaBadge = `<span class="delta-badge delta-positive">✓ Better</span>`;
        }

        tr.innerHTML = `
            <td><strong>${name}</strong></td>
            <td>${row.Centralized}${unit}</td>
            <td>${row.Federated}${unit}</td>
            <td>${deltaBadge}</td>
        `;
        tbody.appendChild(tr);
    }
}

// ── Buildings ───────────────────────────────────────────────────────────────
async function fetchBuildings() {
    try {
        const res       = await fetch('/api/buildings');
        if (!res.ok) return;
        const buildings = await res.json();

        const grid = document.getElementById('building-grid');
        grid.innerHTML = '';

        buildings.forEach(b => {
            const isAnomaly = b.status === 'ANOMALY';
            const letter    = b.id.replace('building_', '').toUpperCase();
            const card      = document.createElement('div');
            card.className  = `building-card ${b.id === selectedBuilding ? 'active-selected' : ''}`;
            card.setAttribute('data-id', b.id);
            card.setAttribute('role', 'button');
            card.setAttribute('tabindex', '0');
            card.setAttribute('aria-label', `Select ${b.name} edge node — status: ${b.status}`);

            card.addEventListener('click', () => selectBuildingNode(b.id));
            card.addEventListener('keydown', e => {
                if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); selectBuildingNode(b.id); }
            });

            const anomalyPct = b.anomaly_rate_pct;
            const anomalyColor = anomalyPct > 10 ? '#FF3366' : anomalyPct > 5 ? '#F59E0B' : '#10B981';

            card.innerHTML = `
                <div class="building-header">
                    <div>
                        <div class="building-id-badge">${letter}</div>
                        <div class="building-name">${b.name}</div>
                        <div class="building-type">${b.type.toUpperCase()} BLOCK · EDGE NODE</div>
                    </div>
                    <span class="status-badge ${isAnomaly ? 'status-anomaly' : 'status-normal'}">
                        ${b.status}
                    </span>
                </div>
                <div class="building-stats">
                    <div class="stat-item">
                        <div class="stat-label">Active Power</div>
                        <div class="stat-val stat-val-power">${b.current_power_kwh.toFixed(1)} <small style="font-size:0.7rem;color:#64748B">kWh</small></div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Grid Voltage</div>
                        <div class="stat-val stat-val-volt">${b.current_voltage.toFixed(1)} <small style="font-size:0.7rem;color:#64748B">V</small></div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Occupancy</div>
                        <div class="stat-val stat-val-occ">${b.current_occupancy} <small style="font-size:0.7rem;color:#64748B">pax</small></div>
                    </div>
                    <div class="stat-item">
                        <div class="stat-label">Anomaly Rate</div>
                        <div class="stat-val" style="color:${anomalyColor}">${anomalyPct}%</div>
                    </div>
                </div>
                <p class="building-desc">${b.description}</p>
                <div class="sparkline-wrap">
                    <canvas id="sparkline-${b.id}" aria-label="${b.name} energy sparkline"></canvas>
                </div>
            `;
            grid.appendChild(card);

            // Render sparkline mini-chart
            renderSparkline(b.id, b.recent_sparkline, isAnomaly);
        });
    } catch (err) {
        console.error('Failed to load buildings:', err);
    }
}

// Sparkline mini-charts
function renderSparkline(buildingId, data, isAnomaly) {
    const canvas = document.getElementById(`sparkline-${buildingId}`);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    if (sparklineCharts[buildingId]) sparklineCharts[buildingId].destroy();

    const color = isAnomaly ? '#FF3366' : '#00E5FF';
    const grad  = ctx.createLinearGradient(0, 0, 0, 36);
    grad.addColorStop(0, isAnomaly ? 'rgba(255,51,102,0.3)' : 'rgba(0,229,255,0.22)');
    grad.addColorStop(1, 'rgba(0,0,0,0)');

    sparklineCharts[buildingId] = new Chart(ctx, {
        type: 'line',
        data: {
            labels: data.map((_, i) => i),
            datasets: [{
                data,
                borderColor:          color,
                borderWidth:          1.5,
                backgroundColor:      grad,
                fill:                 true,
                tension:              0.4,
                pointRadius:          0,
                pointHoverRadius:     3,
                pointHoverBackgroundColor: color
            }]
        },
        options: {
            responsive:          true,
            maintainAspectRatio: false,
            animation:           { duration: 600, easing: 'easeOutCubic' },
            plugins:             { legend: { display: false }, tooltip: { enabled: false } },
            scales: {
                x: { display: false },
                y: { display: false }
            },
            interaction:         { mode: 'nearest', intersect: false }
        }
    });
}

// ── Telemetry Chart ─────────────────────────────────────────────────────────
async function fetchTelemetryData(buildingId) {
    try {
        const res = await fetch(`/api/building/${buildingId}/telemetry?limit=80`);
        if (!res.ok) return;
        const data = await res.json();

        const titleEl = document.getElementById('telemetry-chart-title');
        if (titleEl) titleEl.textContent = `${buildingId.toUpperCase()}: Live Telemetry & Decision Threshold Bounds`;

        renderTelemetryChart(data);
    } catch (err) {
        console.error('Failed to load telemetry:', err);
    }
}

function renderTelemetryChart(data) {
    const canvas = document.getElementById('energyTelemetryChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    const labels      = data.timestamps.map(t => t.split(' ')[1] || t);
    const power       = data.energy_consumption;
    const pointColors = data.is_anomaly.map(a => a === 1 ? '#FF3366' : 'rgba(0,229,255,0.5)');
    const pointRadii  = data.is_anomaly.map(a => a === 1 ? 6 : 2.5);

    const gradFill = ctx.createLinearGradient(0, 0, 0, 300);
    gradFill.addColorStop(0, 'rgba(0,229,255,0.22)');
    gradFill.addColorStop(0.6, 'rgba(0,229,255,0.06)');
    gradFill.addColorStop(1,   'rgba(0,229,255,0.0)');

    if (telemetryChart) telemetryChart.destroy();

    telemetryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels,
            datasets: [{
                label:                'Load (kWh)',
                data:                 power,
                borderColor:          '#00E5FF',
                borderWidth:          2,
                backgroundColor:      gradFill,
                fill:                 true,
                tension:              0.38,
                pointBackgroundColor: pointColors,
                pointBorderColor:     pointColors,
                pointBorderWidth:     0,
                pointRadius:          pointRadii,
                pointHoverRadius:     8,
                pointHoverBorderWidth: 2,
                pointHoverBorderColor: '#fff'
            }]
        },
        options: {
            responsive:          true,
            maintainAspectRatio: false,
            animation:           { duration: 450, easing: 'easeOutCubic' },
            interaction:         { mode: 'index', intersect: false },
            plugins: {
                legend: {
                    display: true,
                    labels:  { color: '#94A3B8', font: { family: 'Inter', weight: '600', size: 12 }, padding: 16, boxWidth: 12 }
                },
                tooltip: {
                    backgroundColor:  'rgba(9,14,26,0.95)',
                    titleColor:       '#00E5FF',
                    bodyColor:        '#EFF6FF',
                    borderColor:      'rgba(0,229,255,0.25)',
                    borderWidth:      1,
                    padding:          14,
                    cornerRadius:     10,
                    displayColors:    false,
                    titleFont:        { family: 'JetBrains Mono', weight: '600', size: 12 },
                    bodyFont:         { family: 'Inter', size: 12 }
                }
            },
            scales: {
                x: {
                    ticks: { color: '#475569', maxTicksLimit: 10, font: { family: 'JetBrains Mono', size: 10 } },
                    grid:  { color: 'rgba(255,255,255,0.025)' },
                    border:{ color: 'transparent' }
                },
                y: {
                    ticks: { color: '#475569', font: { family: 'JetBrains Mono', size: 10 } },
                    grid:  { color: 'rgba(255,255,255,0.025)' },
                    border:{ color: 'transparent' }
                }
            }
        }
    });

    if (power.length > 0) {
        const lastP = power[power.length - 1];
        const lastV = data.voltage ? data.voltage[data.voltage.length - 1] : 230;
        setFooter(lastP.toFixed(1) + ' kWh', lastV.toFixed(1) + ' V', '0.0142', '2.85 μs');
    }
}

function setFooter(power, voltage, mse, lat) {
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set('footer-power',   power);
    set('footer-voltage', voltage);
    set('footer-mse',     mse);
    set('footer-lat',     lat);
}

// ==========================================================================
// REAL-TIME STREAMING
// ==========================================================================
function initStreamControls() {
    const toggleBtn = document.getElementById('stream-toggle-btn');
    if (!toggleBtn) return;

    toggleBtn.addEventListener('click', () => {
        isStreamingActive = !isStreamingActive;
        const statusText  = document.getElementById('stream-status-text');
        if (isStreamingActive) {
            statusText.textContent  = 'LIVE STREAM: ACTIVE';
            toggleBtn.className     = 'status-pill status-live btn-interactive';
            toggleBtn.setAttribute('aria-pressed', 'true');
            startLiveStreaming();
        } else {
            statusText.textContent  = 'LIVE STREAM: PAUSED';
            toggleBtn.className     = 'status-pill status-privacy btn-interactive';
            toggleBtn.setAttribute('aria-pressed', 'false');
            clearInterval(streamInterval);
        }
    });

    startLiveStreaming();
}

function startLiveStreaming() {
    if (streamInterval) clearInterval(streamInterval);
    streamInterval = setInterval(async () => {
        if (!isStreamingActive) return;
        try {
            const res      = await fetch('/api/stream/tick?building_id=all');
            if (!res.ok)   return;
            const liveData = await res.json();
            handleLiveTick(liveData);
        } catch (err) {
            console.error('Live streaming error:', err);
        }
    }, 2000);
}

function handleLiveTick(allBuildingsData) {
    // 1. Update building cards
    for (const [bId, data] of Object.entries(allBuildingsData)) {
        const card = document.querySelector(`.building-card[data-id="${bId}"]`);
        if (!card) continue;

        const powerEl = card.querySelector('.stat-val-power');
        const voltEl  = card.querySelector('.stat-val-volt');
        const occEl   = card.querySelector('.stat-val-occ');
        const badge   = card.querySelector('.status-badge');

        if (powerEl) powerEl.innerHTML = `${data.energy_consumption.toFixed(1)} <small style="font-size:0.7rem;color:#64748B">kWh</small>`;
        if (voltEl)  voltEl.innerHTML  = `${data.voltage.toFixed(1)} <small style="font-size:0.7rem;color:#64748B">V</small>`;
        if (occEl)   occEl.innerHTML   = `${data.occupancy} <small style="font-size:0.7rem;color:#64748B">pax</small>`;

        if (badge) {
            badge.className = `status-badge ${data.is_detected_anomaly ? 'status-anomaly' : 'status-normal'}`;
            badge.textContent = data.is_detected_anomaly ? 'ANOMALY' : 'NORMAL';
        }

        // Update sparkline live
        const sc = sparklineCharts[bId];
        if (sc) {
            const d = sc.data.datasets[0].data;
            d.push(data.energy_consumption);
            if (d.length > 24) d.shift();
            sc.data.labels = d.map((_, i) => i);
            // update color based on anomaly state
            const col = data.is_detected_anomaly ? '#FF3366' : '#00E5FF';
            sc.data.datasets[0].borderColor = col;
            sc.update('none');
        }
    }

    // 2. Append live point to telemetry chart
    const reading = allBuildingsData[selectedBuilding];
    if (reading && telemetryChart) {
        const labels  = telemetryChart.data.labels;
        const dataset = telemetryChart.data.datasets[0];
        const color   = reading.is_detected_anomaly ? '#FF3366' : 'rgba(0,229,255,0.45)';
        const radius  = reading.is_detected_anomaly ? 6 : 2.5;

        labels.push(reading.timestamp);
        dataset.data.push(reading.energy_consumption);
        dataset.pointBackgroundColor.push(color);
        dataset.pointBorderColor.push(color);
        dataset.pointRadius.push(radius);

        if (labels.length > 50) {
            labels.shift();
            dataset.data.shift();
            dataset.pointBackgroundColor.shift();
            dataset.pointBorderColor.shift();
            dataset.pointRadius.shift();
        }
        telemetryChart.update('none');

        // Footer stats with value-flash animation
        flashValue('footer-power',   `${reading.energy_consumption.toFixed(1)} kWh`);
        flashValue('footer-voltage', `${reading.voltage.toFixed(1)} V`);
        flashValue('footer-mse',     `${reading.reconstruction_error.toFixed(4)}`);
        flashValue('footer-lat',     `${reading.inference_latency_us.toFixed(1)} μs`);

        // Header latency
        const latEl = document.getElementById('header-latency');
        if (latEl) latEl.textContent = `~${reading.inference_latency_us.toFixed(1)} μs LATENCY`;
    }

    // 3. Toast & feed for anomaly events
    for (const [bId, data] of Object.entries(allBuildingsData)) {
        if (data.is_detected_anomaly) {
            const sig = `${bId}:${data.anomaly_type}:${data.timestamp}`;
            const now = Date.now();
            if (sig !== lastAlertSignature || now > alertSignatureExpiry) {
                prependLiveAlert(data);
                showToastAlert(data);
                lastAlertSignature  = sig;
                alertSignatureExpiry= now + 7000;
            }
        }
    }
}

function flashValue(elId, newValue) {
    const el = document.getElementById(elId);
    if (!el) return;
    el.textContent = newValue;
    el.classList.remove('value-flash');
    void el.offsetWidth; // force reflow
    el.classList.add('value-flash');
}

// ── Live alert feed entry ────────────────────────────────────────────────────
function prependLiveAlert(data) {
    const feed = document.getElementById('anomaly-feed');
    if (!feed) return;
    const item      = document.createElement('div');
    item.className  = 'feed-item';
    item.innerHTML  = `
        <div class="feed-info">
            <h4>⚡ LIVE · ${data.building_id.toUpperCase()} — ${data.anomaly_type.replace(/_/g,' ').toUpperCase()}</h4>
            <p>${data.full_timestamp} · Latency: ${data.inference_latency_us} μs · MSE: ${data.reconstruction_error}</p>
        </div>
        <div class="feed-val">${data.energy_consumption.toFixed(1)} kWh</div>
    `;
    feed.prepend(item);
    if (feed.children.length > 30) feed.removeChild(feed.lastChild);
}

// ── Toast notification ────────────────────────────────────────────────────────
function showToastAlert(data) {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast     = document.createElement('div');
    toast.className = 'toast';
    toast.innerHTML = `
        <div class="toast-header">
            <div class="toast-title"><span>⚠️</span><span>EDGE ANOMALY DETECTED</span></div>
            <span style="font-size:0.72rem;color:#64748B;font-family:'JetBrains Mono',monospace">${data.timestamp}</span>
        </div>
        <div class="toast-msg">
            <strong>${data.building_id.toUpperCase()} · ${data.building_name}</strong><br>
            ${data.anomaly_type.replace(/_/g,' ').toUpperCase()}
        </div>
        <div class="toast-meta">
            Power: ${data.energy_consumption.toFixed(1)} kWh &nbsp;|&nbsp; MSE: ${data.reconstruction_error} &nbsp;|&nbsp; Latency: ${data.inference_latency_us} μs
        </div>
        <div class="toast-progress"></div>
    `;
    container.appendChild(toast);
    setTimeout(() => toast?.remove(), 4500);
}

// ── Anomaly injection ─────────────────────────────────────────────────────────
async function triggerLiveInjection(anomalyType) {
    const targetNode = selectedBuilding;
    const btn        = document.getElementById(`btn-inject-${anomalyType === 'night_load' ? 'night' : anomalyType === 'voltage_sag' ? 'volt' : anomalyType}`);
    try {
        if (btn) { btn.classList.add('is-loading'); btn.setAttribute('aria-busy', 'true'); }
        const res = await fetch(`/api/stream/inject?building_id=${targetNode}&anomaly_type=${anomalyType}`, { method: 'POST' });
        if (!res.ok) throw new Error(`Injection failed (${res.status})`);
        showActionFeedback(`${anomalyType.replace(/_/g,' ')} queued for ${targetNode.replace('building_','Building ')}.`, 'success');
        // Immediate tick
        const tickRes  = await fetch('/api/stream/tick?building_id=all');
        const liveData = await tickRes.json();
        handleLiveTick(liveData);
    } catch (err) {
        console.error('Anomaly injection error:', err);
        showActionFeedback('Could not queue the anomaly. Check live API connection.', 'error');
    } finally {
        if (btn) { btn.classList.remove('is-loading'); btn.removeAttribute('aria-busy'); }
    }
}

function showActionFeedback(message, type = 'success') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    const toast     = document.createElement('div');
    toast.className = `toast action-toast ${type === 'error' ? 'error' : ''}`;
    toast.innerHTML = `
        <div class="toast-title">
            <span>${type === 'success' ? '✓' : '!'}</span>
            <span>${message}</span>
        </div>
        <div class="toast-progress"></div>
    `;
    container.appendChild(toast);
    setTimeout(() => toast?.remove(), 2800);
}

function clearLiveFeed() {
    const feed = document.getElementById('anomaly-feed');
    if (feed) feed.innerHTML = '<div style="text-align:center;padding:2.5rem;color:#475569;font-size:0.82rem">Feed cleared — waiting for next events…</div>';
}

// ==========================================================================
// HISTORICAL ANOMALY FEED
// ==========================================================================
async function fetchAnomalies() {
    try {
        const res  = await fetch('/api/anomalies?limit=25');
        if (!res.ok) return;
        const list = await res.json();
        const feed = document.getElementById('anomaly-feed');
        feed.innerHTML = '';

        if (list.length === 0) {
            feed.innerHTML = '<div style="text-align:center;padding:2.5rem;color:#475569;font-size:0.82rem">No anomalies recorded yet</div>';
            return;
        }

        list.forEach(a => {
            const item      = document.createElement('div');
            item.className  = `feed-item ${a.severity === 'WARNING' ? 'warning' : ''}`;
            item.innerHTML  = `
                <div class="feed-info">
                    <h4>${a.building_id.toUpperCase()} — ${a.anomaly_type.replace(/_/g,' ').toUpperCase()}</h4>
                    <p>${a.timestamp} · ${a.severity}</p>
                </div>
                <div class="feed-val">${a.energy_kwh.toFixed(1)} kWh</div>
            `;
            feed.appendChild(item);
        });
    } catch (err) {
        console.error('Failed to load anomalies:', err);
    }
}

// ==========================================================================
// BENCHMARK CHARTS
// ==========================================================================
function renderMetricsComparisonChart(metrics) {
    const canvas = document.getElementById('metricsComparisonChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (metricsChart) metricsChart.destroy();

    const keys        = ['Precision', 'Recall', 'F1 Score', 'Accuracy'];
    const centralized = keys.map(k => metrics[k]['Centralized']);
    const federated   = keys.map(k => metrics[k]['Federated']);

    metricsChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: keys,
            datasets: [
                {
                    label:           'Centralized Baseline',
                    data:            centralized,
                    backgroundColor: 'rgba(56,189,248,0.75)',
                    hoverBackgroundColor: 'rgba(56,189,248,0.95)',
                    borderRadius:    8,
                    borderSkipped:   false
                },
                {
                    label:           'Federated Edge (FedAvg)',
                    data:            federated,
                    backgroundColor: 'rgba(16,185,129,0.78)',
                    hoverBackgroundColor: 'rgba(16,185,129,0.98)',
                    borderRadius:    8,
                    borderSkipped:   false
                }
            ]
        },
        options: {
            responsive:          true,
            maintainAspectRatio: false,
            animation:           { duration: 800, easing: 'easeOutCubic' },
            plugins: {
                legend: { labels: { color: '#94A3B8', font: { family: 'Inter', weight: '600', size: 12 }, padding: 16, boxWidth: 12 } },
                tooltip: {
                    backgroundColor: 'rgba(9,14,26,0.95)',
                    titleColor: '#00E5FF',
                    bodyColor: '#EFF6FF',
                    borderColor: 'rgba(0,229,255,0.2)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 10,
                    callbacks: { label: ctx => ` ${ctx.dataset.label}: ${ctx.parsed.y.toFixed(2)}%` }
                }
            },
            scales: {
                x: { ticks: { color: '#64748B', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.03)' }, border: { color: 'transparent' } },
                y: { ticks: { color: '#64748B', font: { size: 11 } }, min: 0, max: 105, grid: { color: 'rgba(255,255,255,0.04)' }, border: { color: 'transparent' } }
            }
        }
    });
}

function renderLatencyChart(metrics) {
    const canvas = document.getElementById('latencyComparisonChart');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (latencyChart) latencyChart.destroy();

    latencyChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: ['Training Time (s)', 'Inference Latency (ms)'],
            datasets: [
                {
                    label:           'Centralized Cloud',
                    data:            [metrics['Training Time']['Centralized'], metrics['Inference Latency']['Centralized']],
                    backgroundColor: 'rgba(99,102,241,0.78)',
                    hoverBackgroundColor: 'rgba(99,102,241,0.98)',
                    borderRadius:    8,
                    borderSkipped:   false
                },
                {
                    label:           'Federated Edge (FedAvg)',
                    data:            [metrics['Training Time']['Federated'], metrics['Inference Latency']['Federated']],
                    backgroundColor: 'rgba(236,72,153,0.78)',
                    hoverBackgroundColor: 'rgba(236,72,153,0.98)',
                    borderRadius:    8,
                    borderSkipped:   false
                }
            ]
        },
        options: {
            responsive:          true,
            maintainAspectRatio: false,
            animation:           { duration: 800, easing: 'easeOutCubic' },
            plugins: {
                legend: { labels: { color: '#94A3B8', font: { family: 'Inter', weight: '600', size: 12 }, padding: 16, boxWidth: 12 } },
                tooltip: {
                    backgroundColor: 'rgba(9,14,26,0.95)',
                    titleColor: '#A78BFA',
                    bodyColor: '#EFF6FF',
                    borderColor: 'rgba(99,102,241,0.25)',
                    borderWidth: 1,
                    padding: 12,
                    cornerRadius: 10
                }
            },
            scales: {
                x: { ticks: { color: '#64748B', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.03)' }, border: { color: 'transparent' } },
                y: { ticks: { color: '#64748B', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' }, border: { color: 'transparent' } }
            }
        }
    });
}

// ==========================================================================
// FEDERATED CONVERGENCE CHARTS
// ==========================================================================
async function fetchFederatedRounds() {
    try {
        const res = await fetch('/api/federated-rounds');
        if (!res.ok) return;
        const history = await res.json();

        const rounds = history.map(h => `R${h.round}`);
        const losses = history.map(h => h.mean_local_loss);
        const f1s    = history.map(h => h.f1 * 100);
        const commMb = history.map(h => h.cumulative_comm_mb);

        // Convergence chart (dual-axis)
        const ctx1 = document.getElementById('convergenceChart').getContext('2d');
        if (convergenceChart) convergenceChart.destroy();

        const gradLoss = ctx1.createLinearGradient(0, 0, 0, 280);
        gradLoss.addColorStop(0, 'rgba(255,51,102,0.18)');
        gradLoss.addColorStop(1, 'rgba(255,51,102,0.0)');

        const gradF1 = ctx1.createLinearGradient(0, 0, 0, 280);
        gradF1.addColorStop(0, 'rgba(16,185,129,0.15)');
        gradF1.addColorStop(1, 'rgba(16,185,129,0.0)');

        convergenceChart = new Chart(ctx1, {
            type: 'line',
            data: {
                labels: rounds,
                datasets: [
                    {
                        label:           'Recon. Loss (MSE)',
                        data:            losses,
                        borderColor:     '#FF3366',
                        backgroundColor: gradLoss,
                        fill:            true,
                        yAxisID:         'y',
                        tension:         0.4,
                        pointRadius:     4,
                        pointBackgroundColor: '#FF3366',
                        pointHoverRadius: 7
                    },
                    {
                        label:           'Global F1 Score (%)',
                        data:            f1s,
                        borderColor:     '#10B981',
                        backgroundColor: gradF1,
                        fill:            true,
                        yAxisID:         'y1',
                        tension:         0.4,
                        pointRadius:     4,
                        pointBackgroundColor: '#10B981',
                        pointHoverRadius: 7
                    }
                ]
            },
            options: {
                responsive:          true,
                maintainAspectRatio: false,
                animation:           { duration: 900, easing: 'easeOutCubic' },
                interaction:         { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: '#94A3B8', font: { size: 12 }, padding: 16, boxWidth: 12 } },
                    tooltip: { backgroundColor: 'rgba(9,14,26,0.95)', titleColor: '#00E5FF', bodyColor: '#EFF6FF', borderColor: 'rgba(0,229,255,0.2)', borderWidth: 1, padding: 12, cornerRadius: 10 }
                },
                scales: {
                    x:  { ticks: { color: '#64748B' }, grid: { color: 'rgba(255,255,255,0.03)' }, border: { color: 'transparent' } },
                    y:  { type: 'linear', position: 'left',  ticks: { color: '#FF3366', font: { size: 11 } }, grid: { color: 'rgba(255,255,255,0.04)' }, border: { color: 'transparent' } },
                    y1: { type: 'linear', position: 'right', ticks: { color: '#10B981', font: { size: 11 } }, min: 0, max: 100, grid: { drawOnChartArea: false }, border: { color: 'transparent' } }
                }
            }
        });

        // Comm accumulation chart
        const ctx2 = document.getElementById('commProgressionChart').getContext('2d');
        if (commChart) commChart.destroy();

        const gradComm = ctx2.createLinearGradient(0, 0, 0, 280);
        gradComm.addColorStop(0, 'rgba(168,85,247,0.22)');
        gradComm.addColorStop(1, 'rgba(168,85,247,0.0)');

        commChart = new Chart(ctx2, {
            type: 'line',
            data: {
                labels: rounds,
                datasets: [{
                    label:           'Cumulative Weight Exchange (MB)',
                    data:            commMb,
                    borderColor:     '#A855F7',
                    backgroundColor: gradComm,
                    fill:            true,
                    tension:         0.4,
                    pointRadius:     4,
                    pointBackgroundColor: '#A855F7',
                    pointHoverRadius: 7
                }]
            },
            options: {
                responsive:          true,
                maintainAspectRatio: false,
                animation:           { duration: 900, easing: 'easeOutCubic' },
                interaction:         { mode: 'index', intersect: false },
                plugins: {
                    legend: { labels: { color: '#94A3B8', font: { size: 12 }, padding: 16, boxWidth: 12 } },
                    tooltip: { backgroundColor: 'rgba(9,14,26,0.95)', titleColor: '#A855F7', bodyColor: '#EFF6FF', borderColor: 'rgba(168,85,247,0.25)', borderWidth: 1, padding: 12, cornerRadius: 10 }
                },
                scales: {
                    x: { ticks: { color: '#64748B' }, grid: { color: 'rgba(255,255,255,0.03)' }, border: { color: 'transparent' } },
                    y: { ticks: { color: '#64748B' }, grid: { color: 'rgba(255,255,255,0.04)' }, border: { color: 'transparent' } }
                }
            }
        });
    } catch (err) {
        console.error('Failed to load federated rounds:', err);
    }
}

// ==========================================================================
// PRIVACY AUDIT
// ==========================================================================
async function fetchPrivacyAudit() {
    try {
        const res = await fetch('/api/privacy-audit');
        if (!res.ok) return;
        const audit = await res.json();

        const container = document.getElementById('privacy-details');
        container.innerHTML = '';

        for (const [k, v] of Object.entries(audit.privacy_guarantees)) {
            const item      = document.createElement('div');
            item.className  = 'privacy-item';
            item.innerHTML  = `<h4>${k.replace(/_/g,' ').toUpperCase()}</h4><p>${v}</p>`;
            container.appendChild(item);
        }

        const matrix = document.getElementById('privacy-matrix');
        matrix.innerHTML = '';
        for (const [mode, details] of Object.entries(audit.comparison_matrix)) {
            const row       = document.createElement('div');
            row.className   = `matrix-row ${mode === 'Centralized' ? 'danger' : 'success'}`;
            row.innerHTML   = `
                <h4>${mode.toUpperCase()} ARCHITECTURE</h4>
                <p><strong>Raw Data Leaves Edge:</strong> ${details.raw_data_leaves_edge ? '⚠ YES — leaves edge perimeter' : '✓ NO — stays strictly on-device'}</p>
                <p><strong>Data Exposure Risk:</strong> ${details.data_exposure_risk}</p>
                <p><strong>Residency:</strong> ${details.data_residency}</p>
            `;
            matrix.appendChild(row);
        }
    } catch (err) {
        console.error('Failed to load privacy audit:', err);
    }
}

// ==========================================================================
// FIGURES GALLERY & LIGHTBOX
// ==========================================================================
async function fetchFigures() {
    try {
        const res     = await fetch('/api/figures-list');
        if (!res.ok)  return;
        const figures = await res.json();
        const gallery = document.getElementById('figures-gallery');
        gallery.innerHTML = '';

        figures.forEach((fig, idx) => {
            const card      = document.createElement('div');
            card.className  = 'figure-card';
            card.setAttribute('tabindex', '0');
            card.setAttribute('role', 'button');
            card.setAttribute('aria-label', `View research figure: ${fig.name}`);
            card.innerHTML  = `
                <div class="figure-img-wrap">
                    <img src="${fig.url}" alt="${fig.name}" loading="lazy">
                    <div class="figure-overlay">
                        <span class="figure-overlay-label">🔍 Expand HD</span>
                    </div>
                </div>
                <div class="figure-caption">
                    <span style="color:var(--text-dim);font-family:'JetBrains Mono',monospace;font-size:0.65rem;font-weight:700">Fig. ${idx + 1}</span><br>
                    ${fig.name}
                </div>
            `;
            card.addEventListener('click',   () => openModal(fig.url, fig.name));
            card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); openModal(fig.url, fig.name); } });
            gallery.appendChild(card);
        });
    } catch (err) {
        console.error('Failed to load figures:', err);
    }
}

function initModal() {
    const modal    = document.getElementById('image-modal');
    const closeBtn = document.querySelector('.modal-close');
    if (closeBtn) closeBtn.addEventListener('click', () => closeModal(modal));
    window.addEventListener('click', e => { if (e.target === modal) closeModal(modal); });
    window.addEventListener('keydown', e => { if (e.key === 'Escape') closeModal(modal); });
}

function openModal(src, caption) {
    const modal       = document.getElementById('image-modal');
    const img         = document.getElementById('modal-img');
    const cap         = document.getElementById('modal-caption');
    const dlBtn       = document.getElementById('modal-download-btn');
    modal.classList.add('open');
    document.body.classList.add('modal-open');
    img.src           = src;
    cap.textContent   = caption;
    if (dlBtn) { dlBtn.href = src; dlBtn.download = caption.replace(/\s+/g,'_') + '.png'; }
}

function closeModal(modal) {
    modal.classList.remove('open');
    document.body.classList.remove('modal-open');
}
