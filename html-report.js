// File: html-report.js
// ==============================================================================
// S&P 500 AI QUANT PREDICTIONS DASHBOARD (HTML REPORT GENERATOR - V5.2)
// Institutional Telemetry & Visualization Interface:
//   - Delegated Event Listeners (Zero Inline Quote-Escaping Bugs)
//   - Model-Aligned Group Badges (Top Long, Top Short, Neutral via p.group)
//   - Complete Directional Telemetry (Direction, Confidence %, SNR, Uncertainty)
//   - Interactive Multi-Column Table Sorting (Rank, Ticker, SNR, Ret, Uncertainty)
//   - Persistent Search Filtering across Live Socket Ingestion
//   - Dual-Horizon Uncertainty Projections with Chart.js
// ==============================================================================

function buildPredictionHtmlReport(dataOrPredictions, topPctArg = 0.10, marketSpreadArg = 0) {
    // Normalization to support either full result object or legacy parameter list
    let predictions = [];
    let topPct = 0.10;
    let marketSpread = 0;
    let macroState = { dispersion: 0, meanReturn: 0, meanHLSpread: 0, meanNormVol: 0 };
    let signalDate = new Date().toISOString().split('T')[0];

    if (dataOrPredictions && Array.isArray(dataOrPredictions.predictions)) {
        predictions = dataOrPredictions.predictions;
        topPct = dataOrPredictions.topK ? (dataOrPredictions.topK / (dataOrPredictions.universeSize || predictions.length)) : 0.10;
        marketSpread = Number(dataOrPredictions.marketSpread) || 0;
        if (dataOrPredictions.macroState) macroState = dataOrPredictions.macroState;
        if (dataOrPredictions.signalDate) signalDate = dataOrPredictions.signalDate;
    } else if (Array.isArray(dataOrPredictions)) {
        predictions = dataOrPredictions;
        topPct = Number(topPctArg) || 0.10;
        marketSpread = Number(marketSpreadArg) || 0;
    }

    const topK = Math.max(1, Math.ceil(predictions.length * topPct));

    const topTails = [...predictions.slice(0, 15), ...predictions.slice(-15)];
    const tailLabels = JSON.stringify(topTails.map(p => p.ticker));
    const tailSnr = JSON.stringify(topTails.map(p => Number((p.snr || 0).toFixed(4))));
    const tailColors = JSON.stringify(topTails.map((p, i) => i < 15 ? 'rgba(34, 197, 94, 0.85)' : 'rgba(239, 68, 68, 0.85)'));

    const scatterData = JSON.stringify(predictions.map(p => ({
        x: Number((p.uncertainty5d || 0).toFixed(2)),
        y: Number((p.expectedReturn5d || 0).toFixed(2)),
        ticker: p.ticker,
        snr: Number((p.snr || 0).toFixed(4))
    })));

    const initialStock = predictions.length > 0 ? predictions[0] : {
        ticker: 'SPY',
        snr: 0,
        expectedReturn5d: 0,
        uncertainty5d: 1.5,
        direction: 'Neutral',
        directionConfidence: 50.0,
        group: 'Neutral',
        rank: 1
    };

    const tableRowsHtml = predictions.map((p, idx) => {
        const rank = p.rank || (idx + 1);
        const snr = Number(p.snr || 0);
        const expRet = Number(p.expectedReturn5d || 0);
        const unc = Number(p.uncertainty5d || 0);
        const dir = p.direction || (snr >= 0 ? 'Bullish' : 'Bearish');
        const dirConf = Number(p.directionConfidence || 50.0);
        const group = p.group || (idx < topK && expRet > 0 ? 'Top Long' : (idx >= predictions.length - topK && expRet < 0 ? 'Top Short' : 'Neutral'));

        let badgeClass = 'badge-neutral';
        if (group === 'Top Long') badgeClass = 'badge-green';
        else if (group === 'Top Short') badgeClass = 'badge-red';

        let dirIcon = '●';
        let dirColor = '#94a3b8';
        if (dir === 'Bullish') { dirIcon = '▲'; dirColor = '#4ade80'; }
        else if (dir === 'Bearish') { dirIcon = '▼'; dirColor = '#f87171'; }

        const portfolioWeight = Number(p.portfolioWeight) || 0;
        const price = Number(p.price) || 0;
        const suggestedShares = portfolioWeight > 0 && price > 0
            ? Math.floor((10000 * portfolioWeight) / price)
            : null;
        const shareLabel = suggestedShares === null
            ? 'No buy'
            : (suggestedShares > 0 ? `Buy ${suggestedShares} shares` : 'Buy <1 share');

        return `<tr data-ticker="${p.ticker}">
            <td><strong>#${rank}</strong></td>
            <td><strong style="color: #38bdf8;">${p.ticker}</strong><span class="ticker-recommendation suggested-shares" data-weight="${portfolioWeight}" data-price="${price}">${shareLabel}</span></td>
            <td style="color:${dirColor}; font-weight: 600;">${dirIcon} ${dir} <span style="font-size: 11px; color: #94a3b8;">(${dirConf.toFixed(1)}%)</span></td>
            <td style="color:${snr >= 0 ? '#4ade80' : '#f87171'}; font-weight: bold;">${snr >= 0 ? '+' : ''}${snr.toFixed(4)}</td>
            <td style="color:${expRet >= 0 ? '#4ade80' : '#f87171'}; font-weight: bold;">${expRet >= 0 ? '+' : ''}${expRet.toFixed(2)}%</td>
            <td style="color: #cbd5e1;">&plusmn;${unc.toFixed(2)}%</td>
            <td><span class="badge ${badgeClass}">${group}</span></td>
            <td><button class="btn-micro btn-forecast-trigger" data-ticker="${p.ticker}">Forecast</button></td>
        </tr>`;
    }).join('\n');

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S&P 500 AI Quant Factor Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="/socket.io/socket.io.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #0b0f19; color: #f1f5f9; padding: 25px; margin: 0; }
        .container { max-width: 1440px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 15px; }
        .header-left { display: flex; align-items: center; gap: 15px; }
        .back-link { color: #94a3b8; text-decoration: none; font-size: 14px; padding: 6px 12px; background: #1e293b; border-radius: 8px; border: 1px solid #334155; transition: all 0.2s; }
        .back-link:hover { color: #fff; background: #334155; }
        h1 { font-size: 24px; color: #38bdf8; margin: 0; }
        .header-actions { display: flex; align-items: center; gap: 12px; }
        .btn-action { background: #6366f1; color: white; border: none; padding: 8px 16px; border-radius: 8px; font-weight: 600; cursor: pointer; display: inline-flex; align-items: center; gap: 8px; transition: all 0.2s; font-size: 14px; }
        .btn-action:hover { background: #4f46e5; transform: translateY(-1px); }
        .btn-action:disabled { opacity: 0.6; cursor: not-allowed; transform: none; }
        .stats-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin-bottom: 25px; }
        .stat-card { background: #1e293b; border-radius: 10px; padding: 18px; border: 1px solid #334155; }
        .stat-label { color: #94a3b8; font-size: 13px; text-transform: uppercase; letter-spacing: 0.5px; }
        .stat-val { font-size: 24px; font-weight: bold; margin-top: 5px; color: #f8fafc; }
        .grid-2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 25px; }
        .card { background: #1e293b; border-radius: 12px; padding: 20px; border: 1px solid #334155; }
        .card-title { font-size: 16px; font-weight: 600; margin-bottom: 12px; color: #e2e8f0; display: flex; justify-content: space-between; align-items: center; }
        .badge { padding: 4px 9px; border-radius: 4px; font-size: 11px; font-weight: bold; text-transform: uppercase; letter-spacing: 0.4px; }
        .badge-green { background: rgba(34, 197, 94, 0.2); color: #4ade80; border: 1px solid rgba(34, 197, 94, 0.4); }
        .badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }
        .badge-neutral { background: rgba(148, 163, 184, 0.2); color: #94a3b8; border: 1px solid rgba(148, 163, 184, 0.4); }
        .search-box { width: 100%; padding: 10px 15px; background: #0f172a; border: 1px solid #475569; border-radius: 8px; color: #fff; box-sizing: border-box; margin-bottom: 15px; font-size: 13px; }
        .search-box:focus { outline: none; border-color: #38bdf8; }
        .table-container { max-height: 480px; overflow-y: auto; border: 1px solid #334155; border-radius: 8px; }
        .ticker-recommendation { display: block; color: #4ade80; font-size: 11px; font-weight: 600; margin-top: 3px; white-space: nowrap; }
        .portfolio-budget-control { display: flex; align-items: center; justify-content: flex-end; gap: 8px; margin: 2px 0 12px; color: #cbd5e1; font-size: 13px; font-weight: 600; }
        .portfolio-budget-control input { width: 130px; padding: 8px 10px; background: #0f172a; border: 1px solid #64748b; border-radius: 4px; color: #fff; font-size: 14px; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
        th { background: #0f172a; padding: 11px 12px; color: #94a3b8; position: sticky; top: 0; z-index: 2; user-select: none; cursor: pointer; }
        th:hover { color: #f8fafc; }
        td { padding: 10px 12px; border-bottom: 1px solid #1e293b; background: #131c2e; }
        tr:hover td { background: #1e293b; cursor: pointer; }
        .btn-micro { background: #38bdf8; color: #0b0f19; border: none; border-radius: 4px; padding: 4px 10px; font-size: 11px; font-weight: bold; cursor: pointer; transition: all 0.15s; }
        .btn-micro:hover { background: #7dd3fc; transform: scale(1.04); }
        .stock-selector-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 15px; }
        .stock-chips { display: flex; gap: 6px; flex-wrap: wrap; }
        .chip { background: #0f172a; border: 1px solid #475569; color: #94a3b8; padding: 5px 12px; border-radius: 6px; font-size: 12px; cursor: pointer; font-weight: bold; transition: all 0.15s; }
        .chip:hover, .chip.active { background: #6366f1; color: white; border-color: #6366f1; }
        .stock-stat-badge { background: #0f172a; padding: 8px 14px; border-radius: 8px; border: 1px solid #334155; display: inline-flex; flex-direction: column; min-width: 105px; }
        .stock-stat-badge .sub { font-size: 10px; color: #94a3b8; text-transform: uppercase; font-weight: 600; margin-bottom: 2px; }
        .stock-stat-badge .val { font-size: 14px; font-weight: bold; color: #fff; }
        .stock-info-strip { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; margin-bottom: 15px; padding-bottom: 12px; border-bottom: 1px solid #334155; }
        .toast-notify { position: fixed; bottom: 20px; right: 20px; background: #1e293b; border: 1px solid #6366f1; color: #fff; padding: 12px 20px; border-radius: 8px; box-shadow: 0 10px 25px rgba(0,0,0,0.5); z-index: 1000; opacity: 0; transform: translateY(10px); transition: all 0.3s; pointer-events: none; }
        .toast-notify.visible { opacity: 1; transform: translateY(0); }
        @media (max-width: 900px) {
            .stats-grid { grid-template-columns: 1fr 1fr; }
            .grid-2 { grid-template-columns: 1fr; }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <div class="header-left">
                <a href="/" class="back-link">&larr; Main Menu</a>
                <div>
                    <h1>📈 S&P 500 Spatio-Temporal Factor Alpha Engine (V5.2)</h1>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 4px;">
                        ResNet-TCN + 6-Layer SDPA Transformer | As of Session: <strong>${signalDate}</strong> | Universe: <span id="headerUniverseCount">${predictions.length}</span> Assets
                    </div>
                </div>
            </div>
            <div class="header-actions">
                <button class="btn-action" id="btnUpdateModel" onclick="updateModel()">
                    <span id="updateIcon">⚡</span>
                    <span id="updateLabel">Re-Run Model Inference</span>
                </button>
                <span class="badge badge-green" id="serverBadge" style="font-size: 12px; padding: 7px 12px;">LIVE SERVER</span>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Active Universe</div>
                <div class="stat-val" id="statUniverse">${predictions.length} Stocks</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Market Conviction Spread</div>
                <div class="stat-val" id="statSpread" style="color: #38bdf8;">${marketSpread.toFixed(4)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Target Long Basket</div>
                <div class="stat-val" id="statTargetLong" style="color: #4ade80;">${topK} Stocks</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Target Short Basket</div>
                <div class="stat-val" id="statTargetShort" style="color: #f87171;">${topK} Stocks</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 25px;">
            <div class="card-title">
                <span>🎯 Asset Direction &amp; Multi-Horizon Forward Trajectory</span>
                <span id="selectedStockTag" class="badge badge-green" style="font-size: 12px;">${initialStock.group.toUpperCase()}</span>
            </div>

            <div class="stock-selector-row">
                <div style="font-size: 13px; color: #94a3b8; font-weight: bold;">Quick Select:</div>
                <div class="stock-chips" id="quickSelectChips">
                    <span class="chip active" data-ticker="${initialStock.ticker}">${initialStock.ticker}</span>
                    <span class="chip" data-ticker="NVDA">NVDA</span>
                    <span class="chip" data-ticker="AAPL">AAPL</span>
                    <span class="chip" data-ticker="TSLA">TSLA</span>
                    <span class="chip" data-ticker="MSFT">MSFT</span>
                    <span class="chip" data-ticker="AMZN">AMZN</span>
                    <span class="chip" data-ticker="GOOGL">GOOGL</span>
                    <span class="chip" data-ticker="META">META</span>
                    <span class="chip" data-ticker="JPM">JPM</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px; margin-left: auto;">
                    <input type="text" id="customTickerInput" placeholder="Enter Ticker (e.g. AMD)..." style="background:#0f172a; border:1px solid #475569; color:#fff; padding:6px 12px; border-radius:6px; font-size:12px; width:170px; text-transform:uppercase;">
                    <button class="btn-micro" style="padding: 6px 14px; font-size: 12px; height: 31px;" onclick="searchCustomStock()">Forecast &rarr;</button>
                </div>
            </div>

            <div class="stock-info-strip">
                <div class="stock-stat-badge">
                    <span class="sub">Instrument</span>
                    <span class="val" id="detailTicker" style="color: #38bdf8;">${initialStock.ticker}</span>
                </div>
                <div class="stock-stat-badge">
                    <span class="sub">5D Direction</span>
                    <span class="val" id="detailDirection" style="color: ${initialStock.direction === 'Bearish' ? '#f87171' : '#4ade80'};">
                        ${initialStock.direction === 'Bearish' ? '▼' : '▲'} ${initialStock.direction} (${Number(initialStock.directionConfidence || 50).toFixed(1)}%)
                    </span>
                </div>
                <div class="stock-stat-badge">
                    <span class="sub">Composite SNR</span>
                    <span class="val" id="detailSnr" style="color: ${initialStock.snr >= 0 ? '#4ade80' : '#f87171'};">
                        ${initialStock.snr >= 0 ? '+' : ''}${Number(initialStock.snr || 0).toFixed(4)}
                    </span>
                </div>
                <div class="stock-stat-badge">
                    <span class="sub">Expected 5D Return</span>
                    <span class="val" id="detailReturn" style="color: ${initialStock.expectedReturn5d >= 0 ? '#4ade80' : '#f87171'};">
                        ${initialStock.expectedReturn5d >= 0 ? '+' : ''}${Number(initialStock.expectedReturn5d || 0).toFixed(2)}%
                    </span>
                </div>
                <div class="stock-stat-badge">
                    <span class="sub">Uncertainty (σ)</span>
                    <span class="val" id="detailUncertainty">&plusmn;${Number(initialStock.uncertainty5d || 0).toFixed(2)}%</span>
                </div>
                <div class="stock-stat-badge">
                    <span class="sub">Factor Rank</span>
                    <span class="val" id="detailRank">#${initialStock.rank || 1}</span>
                </div>
            </div>

            <canvas id="individualStockChart" height="100"></canvas>
        </div>

        <div class="grid-2">
            <div class="card">
                <div class="card-title">1. S&P 500 Factor Risk vs. Return Cross-Section (${predictions.length} Stocks)</div>
                <canvas id="scatterChart" height="230"></canvas>
            </div>
            <div class="card">
                <div class="card-title">2. Top 15 Longs vs. Bottom 15 Shorts (Extreme Tail Quantiles)</div>
                <canvas id="tailBarChart" height="230"></canvas>
            </div>
        </div>

        <div class="card">
            <div class="card-title">
                <span>3. Cross-Sectional Alpha Factor Ranking Table</span>
                <span style="font-size: 12px; color: #94a3b8; font-weight: normal;">Click any row to stream asset trajectory</span>
            </div>
            <div class="portfolio-budget-control">
                <label for="portfolioBudget">Portfolio value</label>
                <span>$</span>
                <input type="number" id="portfolioBudget" min="0" step="100" value="10000" aria-label="Portfolio value in dollars">
            </div>
            <input type="text" id="searchInput" class="search-box" placeholder="🔍 Search Ticker (e.g. NVDA, AAPL, MSFT, TSLA)...">
            <div class="table-container">
                <table id="stocksTable">
                    <thead>
                        <tr>
                            <th onclick="sortTable(0)">Rank &#x25B4;&#x25BE;</th>
                            <th onclick="sortTable(1)">Ticker &#x25B4;&#x25BE;</th>
                            <th onclick="sortTable(2)">Direction (Conf) &#x25B4;&#x25BE;</th>
                            <th onclick="sortTable(3)">Composite SNR &#x25B4;&#x25BE;</th>
                            <th onclick="sortTable(4)">Expected 5D Ret &#x25B4;&#x25BE;</th>
                            <th onclick="sortTable(5)">Uncertainty (σ) &#x25B4;&#x25BE;</th>
                            <th onclick="sortTable(6)">Target Group &#x25B4;&#x25BE;</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody id="stocksTableBody">
                        ${tableRowsHtml}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <div class="toast-notify" id="toastNotify"></div>

    <script>
        let scatterChartInstance = null;
        let tailBarChartInstance = null;
        let individualStockChartInstance = null;
        let currentSelectedTicker = "${initialStock.ticker}";
        let sortDirection = {};
        let socket = null;

        function showToast(msg) {
            const el = document.getElementById('toastNotify');
            el.textContent = msg;
            el.classList.add('visible');
            setTimeout(() => el.classList.remove('visible'), 3200);
        }

        function updateShareRecommendations() {
            const budget = Math.max(0, Number(document.getElementById('portfolioBudget').value) || 0);
            document.querySelectorAll('.suggested-shares').forEach((label) => {
                const weight = Number(label.dataset.weight) || 0;
                const price = Number(label.dataset.price) || 0;
                if (weight <= 0 || price <= 0) {
                    label.textContent = 'No buy';
                    return;
                }
                const shares = Math.floor((budget * weight) / price);
                label.textContent = shares > 0 ? 'Buy ' + shares + ' shares' : 'Buy <1 share';
            });
        }

        document.getElementById('portfolioBudget').addEventListener('input', updateShareRecommendations);

        try {
            socket = io();
            socket.on('connect', () => {
                const b = document.getElementById('serverBadge');
                b.textContent = 'LIVE SERVER';
                b.className = 'badge badge-green';
            });
            socket.on('disconnect', () => {
                const b = document.getElementById('serverBadge');
                b.textContent = 'OFFLINE';
                b.className = 'badge badge-red';
            });
            socket.on('predictionsUpdated', (data) => {
                showToast('Predictions updated in real-time!');
                refreshDashboardFromData(data);
            });
            socket.on('predictionsError', (err) => {
                showToast(err?.error || 'Prediction engine unavailable');
            });
        } catch (e) {}

        const initialScatter = ${scatterData};
        scatterChartInstance = new Chart(document.getElementById('scatterChart'), {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'S&P 500 Constituent',
                    data: initialScatter,
                    backgroundColor: initialScatter.map(d => d.snr >= 0 ? 'rgba(56, 189, 248, 0.75)' : 'rgba(244, 63, 94, 0.75)'),
                    pointRadius: 4.5,
                    pointHoverRadius: 7
                }]
            },
            options: {
                responsive: true,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (ctx) => {
                                const p = ctx.raw;
                                return p.ticker + ': ExpRet=' + p.y + '%, σ=' + p.x + '%, SNR=' + p.snr;
                            }
                        }
                    }
                },
                scales: {
                    x: { title: { display: true, text: 'Uncertainty / Volatility Band (σ %)', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                    y: { title: { display: true, text: 'Expected 5-Day Open-to-Open Return (%)', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
                }
            }
        });

        tailBarChartInstance = new Chart(document.getElementById('tailBarChart'), {
            type: 'bar',
            data: {
                labels: ${tailLabels},
                datasets: [{
                    data: ${tailSnr},
                    backgroundColor: ${tailColors},
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true,
                plugins: { legend: { display: false } },
                scales: {
                    y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' }, title: { display: true, text: 'Composite Alpha Conviction', color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#f8fafc', font: { size: 10 } } }
                }
            }
        });

        // Delegated Click Handling: Completely immune to single/double quote escaping bugs
        document.getElementById('stocksTableBody').addEventListener('click', (e) => {
            const tr = e.target.closest('tr');
            if (!tr) return;
            const ticker = tr.getAttribute('data-ticker');
            if (ticker) selectStock(ticker);
        });

        document.getElementById('quickSelectChips').addEventListener('click', (e) => {
            const chip = e.target.closest('.chip');
            if (!chip) return;
            const ticker = chip.getAttribute('data-ticker');
            if (ticker) selectStock(ticker);
        });

        async function selectStock(ticker) {
            currentSelectedTicker = ticker;
            document.querySelectorAll('.chip').forEach(c => {
                c.classList.toggle('active', c.getAttribute('data-ticker') === ticker.toUpperCase());
            });

            try {
                const res = await fetch('/api/stock-forecast/' + encodeURIComponent(ticker));
                if (res.status === 202) {
                    showToast('Data cache warming up — retrying...');
                    setTimeout(() => selectStock(ticker), 2500);
                    return;
                }
                if (!res.ok) return;
                const data = await res.json();
                renderIndividualChart(data);
            } catch (err) {
                console.error(err);
            }
        }

        function renderIndividualChart(data) {
            document.getElementById('detailTicker').textContent = data.ticker;
            document.getElementById('detailRank').textContent = '#' + data.rank;
            document.getElementById('detailSnr').textContent = (data.snr >= 0 ? '+' : '') + Number(data.snr).toFixed(4);
            document.getElementById('detailReturn').textContent = (data.expectedReturn5d >= 0 ? '+' : '') + Number(data.expectedReturn5d).toFixed(2) + '%';
            document.getElementById('detailReturn').style.color = data.expectedReturn5d >= 0 ? '#4ade80' : '#f87171';
            document.getElementById('detailUncertainty').textContent = '±' + Number(data.uncertainty5d).toFixed(2) + '%';

            const dirEl = document.getElementById('detailDirection');
            const isBullish = data.direction === 'Bullish' || data.snr >= 0;
            dirEl.textContent = (isBullish ? '▲ ' : '▼ ') + data.direction + ' (' + Number(data.directionConfidence || 50).toFixed(1) + '%)';
            dirEl.style.color = isBullish ? '#4ade80' : '#f87171';

            const tagEl = document.getElementById('selectedStockTag');
            tagEl.textContent = (data.group || 'Neutral').toUpperCase();
            if (data.group === 'Top Long') tagEl.className = 'badge badge-green';
            else if (data.group === 'Top Short') tagEl.className = 'badge badge-red';
            else tagEl.className = 'badge badge-neutral';

            const primaryColor = isBullish ? '#38bdf8' : '#f43f5e';
            const projColor = isBullish ? '#4ade80' : '#f87171';
            const bandFill = isBullish ? 'rgba(56, 189, 248, 0.15)' : 'rgba(244, 63, 94, 0.15)';

            if (individualStockChartInstance) {
                individualStockChartInstance.destroy();
            }

            individualStockChartInstance = new Chart(document.getElementById('individualStockChart'), {
                type: 'line',
                data: {
                    labels: data.labels,
                    datasets: [
                        { label: 'Upper Band (+σ)', data: data.upperBand, borderColor: 'transparent', backgroundColor: bandFill, fill: '+1', pointRadius: 0 },
                        { label: 'Lower Band (-σ)', data: data.lowerBand, borderColor: 'transparent', backgroundColor: 'transparent', fill: false, pointRadius: 0 },
                        { label: 'Historical Price', data: data.historical, borderColor: primaryColor, backgroundColor: primaryColor, borderWidth: 2, tension: 0.15, pointRadius: 0, pointHoverRadius: 4 },
                        { label: 'Predicted 5D Trajectory', data: data.projection, borderColor: projColor, backgroundColor: projColor, borderWidth: 3, borderDash: [6, 4], tension: 0.2, pointRadius: 4 }
                    ]
                },
                options: {
                    responsive: true,
                    interaction: { mode: 'index', intersect: false },
                    plugins: { legend: { labels: { filter: (item) => !item.text.includes('Band'), color: '#e2e8f0' } } },
                    scales: {
                        y: { title: { display: true, text: 'Price (USD)', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                        x: { grid: { color: '#1e293b' }, ticks: { color: '#f8fafc', font: { size: 11 } } }
                    }
                }
            });
        }

        async function updateModel() {
            const btn = document.getElementById('btnUpdateModel');
            const icon = document.getElementById('updateIcon');
            const label = document.getElementById('updateLabel');

            btn.disabled = true;
            icon.textContent = '⏳';
            label.textContent = 'Running Inference...';

            try {
                const res = await fetch('/api/run-inference', { method: 'POST' });
                const data = await res.json();
                if (data && data.predictions) {
                    refreshDashboardFromData(data);
                    showToast('Inference complete (' + data.latency + 'ms)!');
                } else {
                    showToast(data.message || 'Inference started in background');
                }
            } catch (err) {
                console.error(err);
                showToast('Inference request failed');
            } finally {
                btn.disabled = false;
                icon.textContent = '⚡';
                label.textContent = 'Re-Run Model Inference';
            }
        }

        function refreshDashboardFromData(data) {
            if (!data || !data.predictions) return;
            const predictions = data.predictions;

            document.getElementById('statSpread').textContent = Number(data.marketSpread || 0).toFixed(4);
            document.getElementById('statUniverse').textContent = predictions.length + ' Stocks';
            document.getElementById('headerUniverseCount').textContent = predictions.length;

            const topK = data.topK || Math.max(1, Math.ceil(predictions.length * 0.10));
            document.getElementById('statTargetLong').textContent = topK + ' Stocks';
            document.getElementById('statTargetShort').textContent = topK + ' Stocks';

            // Update Tail Bar Chart
            const topTails = [...predictions.slice(0, 25), ...predictions.slice(-25)];
            tailBarChartInstance.data.labels = topTails.map(p => p.ticker);
            tailBarChartInstance.data.datasets[0].data = topTails.map(p => Number((p.snr || 0).toFixed(4)));
            tailBarChartInstance.data.datasets[0].backgroundColor = topTails.map((p, i) => i < 25 ? 'rgba(34, 197, 94, 0.85)' : 'rgba(239, 68, 68, 0.85)');
            tailBarChartInstance.update();

            // Update Scatter Chart
            const newScatter = predictions.map(p => ({
                x: Number((p.uncertainty5d || 0).toFixed(2)),
                y: Number((p.expectedReturn5d || 0).toFixed(2)),
                ticker: p.ticker,
                snr: Number((p.snr || 0).toFixed(4))
            }));
            scatterChartInstance.data.datasets[0].data = newScatter;
            scatterChartInstance.data.datasets[0].backgroundColor = newScatter.map(d => d.snr >= 0 ? 'rgba(56, 189, 248, 0.75)' : 'rgba(244, 63, 94, 0.75)');
            scatterChartInstance.update();

            // Render Table Rows with Delegated Dataset Targets
            const tableBody = document.getElementById('stocksTableBody');
            tableBody.innerHTML = predictions.map((p, idx) => {
                const rank = p.rank || (idx + 1);
                const snr = Number(p.snr || 0);
                const expRet = Number(p.expectedReturn5d || 0);
                const unc = Number(p.uncertainty5d || 0);
                const dir = p.direction || (snr >= 0 ? 'Bullish' : 'Bearish');
                const dirConf = Number(p.directionConfidence || 50.0);
                const group = p.group || (idx < topK && expRet > 0 ? 'Top Long' : (idx >= predictions.length - topK && expRet < 0 ? 'Top Short' : 'Neutral'));

                let badgeClass = 'badge-neutral';
                if (group === 'Top Long') badgeClass = 'badge-green';
                else if (group === 'Top Short') badgeClass = 'badge-red';

                let dirIcon = '●';
                let dirColor = '#94a3b8';
                if (dir === 'Bullish') { dirIcon = '▲'; dirColor = '#4ade80'; }
                else if (dir === 'Bearish') { dirIcon = '▼'; dirColor = '#f87171'; }

                const portfolioWeight = Number(p.portfolioWeight) || 0;
                const price = Number(p.price) || 0;
                const budget = Math.max(0, Number(document.getElementById('portfolioBudget').value) || 0);
                const suggestedShares = portfolioWeight > 0 && price > 0
                    ? Math.floor((budget * portfolioWeight) / price)
                    : null;
                const shareLabel = suggestedShares === null
                    ? 'No buy'
                    : (suggestedShares > 0 ? 'Buy ' + suggestedShares + ' shares' : 'Buy <1 share');

                return '<tr data-ticker="' + p.ticker + '">' +
                    '<td><strong>#' + rank + '</strong></td>' +
                    '<td><strong style="color: #38bdf8;">' + p.ticker + '</strong><span class="ticker-recommendation suggested-shares" data-weight="' + portfolioWeight + '" data-price="' + price + '">' + shareLabel + '</span></td>' +
                    '<td style="color:' + dirColor + '; font-weight: 600;">' + dirIcon + ' ' + dir + ' <span style="font-size: 11px; color: #94a3b8;">(' + dirConf.toFixed(1) + '%)</span></td>' +
                    '<td style="color:' + (snr >= 0 ? '#4ade80' : '#f87171') + '; font-weight: bold;">' + (snr >= 0 ? '+' : '') + snr.toFixed(4) + '</td>' +
                    '<td style="color:' + (expRet >= 0 ? '#4ade80' : '#f87171') + '; font-weight: bold;">' + (expRet >= 0 ? '+' : '') + expRet.toFixed(2) + '%</td>' +
                    '<td style="color: #cbd5e1;">&plusmn;' + unc.toFixed(2) + '%</td>' +
                    '<td><span class="badge ' + badgeClass + '">' + group + '</span></td>' +
                    '<td><button class="btn-micro btn-forecast-trigger" data-ticker="' + p.ticker + '">Forecast</button></td>' +
                '</tr>';
            }).join('');

            applySearchFilter();
            selectStock(currentSelectedTicker);
        }

        function applySearchFilter() {
            const filter = document.getElementById('searchInput').value.trim().toUpperCase();
            const rows = document.querySelectorAll('#stocksTable tbody tr');
            rows.forEach(row => {
                const ticker = row.getAttribute('data-ticker') || '';
                row.style.display = ticker.toUpperCase().includes(filter) ? '' : 'none';
            });
        }

        document.getElementById('searchInput').addEventListener('input', applySearchFilter);

        function searchCustomStock() {
            const input = document.getElementById('customTickerInput');
            const val = input.value.trim().toUpperCase();
            if (val) {
                selectStock(val);
                input.value = '';
            }
        }

        document.getElementById('customTickerInput').addEventListener('keydown', function(e) {
            if (e.key === 'Enter') searchCustomStock();
        });

        function sortTable(colIndex) {
            const table = document.getElementById('stocksTable');
            const tbody = table.querySelector('tbody');
            const rows = Array.from(tbody.querySelectorAll('tr'));
            
            sortDirection[colIndex] = !sortDirection[colIndex];
            const asc = sortDirection[colIndex];

            rows.sort((a, b) => {
                let cellA = a.cells[colIndex].textContent.trim().replace(/[#+%\&plusmn;]/g, '');
                let cellB = b.cells[colIndex].textContent.trim().replace(/[#+%\&plusmn;]/g, '');

                const numA = parseFloat(cellA);
                const numB = parseFloat(cellB);

                if (!isNaN(numA) && !isNaN(numB)) {
                    return asc ? numA - numB : numB - numA;
                }
                return asc ? cellA.localeCompare(cellB) : cellB.localeCompare(cellA);
            });

            rows.forEach(r => tbody.appendChild(r));
        }

        // Initialize display with active stock
        selectStock(currentSelectedTicker);
    </script>
</body>
</html>`;
}

module.exports = { buildPredictionHtmlReport };
