function buildPredictionHtmlReport(predictions, topPct, marketSpread) {
    const topK = Math.max(1, Math.ceil(predictions.length * topPct));

    const topTails = [...predictions.slice(0, 25), ...predictions.slice(-25)];
    const tailLabels = JSON.stringify(topTails.map(p => p.ticker));
    const tailSnr = JSON.stringify(topTails.map(p => Number(p.snr.toFixed(4))));
    const tailColors = JSON.stringify(topTails.map((p, i) => i < 25 ? 'rgba(34, 197, 94, 0.85)' : 'rgba(239, 68, 68, 0.85)'));

    const scatterData = JSON.stringify(predictions.map(p => ({
        x: Number(p.uncertainty5d.toFixed(3)),
        y: Number(p.expectedReturn5d.toFixed(2)),
        ticker: p.ticker,
        snr: Number(p.snr.toFixed(4))
    })));

    const initialStock = predictions[0] ? predictions[0].ticker : "AAPL";

    const tableRowsHtml = predictions.map((p, idx) => {
        const safeTicker = String(p.ticker).replace(/\\/g, '\\\\').replace(/'/g, "\\'");
        let tag = '<span class="badge badge-neutral">Neutral</span>';
        if (idx < topK) tag = '<span class="badge badge-green">Top Long</span>';
        else if (idx >= predictions.length - topK) tag = '<span class="badge badge-red">Short</span>';

        return `<tr data-ticker="${p.ticker}" style="cursor: pointer;" onclick="selectStock('${safeTicker}')">
            <td><strong>${idx + 1}</strong></td>
            <td><strong>${p.ticker}</strong></td>
            <td style="color:${p.snr >= 0 ? '#4ade80' : '#f87171'}">${p.snr.toFixed(4)}</td>
            <td>${p.expectedReturn5d >= 0 ? '+' : ''}${p.expectedReturn5d.toFixed(2)}%</td>
            <td>${p.uncertainty5d.toFixed(4)}</td>
            <td>${tag}</td>
            <td><button class="btn-micro" onclick="event.stopPropagation(); selectStock('${safeTicker}')">Forecast</button></td>
        </tr>`;
    }).join('\n');

    return `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>S&P 500 AI Quant Dashboard</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="/socket.io/socket.io.js"></script>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background-color: #0b0f19; color: #f1f5f9; padding: 25px; margin: 0; }
        .container { max-width: 1400px; margin: 0 auto; }
        .header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px; flex-wrap: wrap; gap: 15px; }
        .header-left { display: flex; align-items: center; gap: 15px; }
        .back-link { color: #94a3b8; text-decoration: none; font-size: 14px; padding: 6px 12px; background: #1e293b; border-radius: 8px; border: 1px solid #334155; transition: all 0.2s; }
        .back-link:hover { color: #fff; background: #334155; }
        h1 { font-size: 26px; color: #38bdf8; margin: 0; }
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
        .badge { padding: 3px 8px; border-radius: 4px; font-size: 11px; font-weight: bold; }
        .badge-green { background: rgba(34, 197, 94, 0.2); color: #4ade80; }
        .badge-red { background: rgba(239, 68, 68, 0.2); color: #f87171; }
        .badge-neutral { background: rgba(148, 163, 184, 0.2); color: #94a3b8; }
        .search-box { width: 100%; padding: 10px 15px; background: #0f172a; border: 1px solid #475569; border-radius: 8px; color: #fff; box-sizing: border-box; margin-bottom: 15px; }
        .table-container { max-height: 440px; overflow-y: auto; }
        table { width: 100%; border-collapse: collapse; font-size: 13px; text-align: left; }
        th { background: #0f172a; padding: 10px; color: #94a3b8; position: sticky; top: 0; z-index: 1; }
        td { padding: 10px; border-bottom: 1px solid #334155; }
        tr:hover { background: #334155; }
        .btn-micro { background: #38bdf8; color: #0b0f19; border: none; border-radius: 4px; padding: 2px 8px; font-size: 11px; font-weight: bold; cursor: pointer; }
        .btn-micro:hover { background: #7dd3fc; }
        .stock-selector-row { display: flex; align-items: center; gap: 10px; flex-wrap: wrap; margin-bottom: 15px; }
        .stock-chips { display: flex; gap: 6px; flex-wrap: wrap; }
        .chip { background: #0f172a; border: 1px solid #475569; color: #94a3b8; padding: 4px 10px; border-radius: 6px; font-size: 12px; cursor: pointer; font-weight: bold; }
        .chip:hover, .chip.active { background: #6366f1; color: white; border-color: #6366f1; }
        .stock-stat-badge { background: #0f172a; padding: 6px 12px; border-radius: 8px; border: 1px solid #334155; display: inline-flex; flex-direction: column; min-width: 100px; }
        .stock-stat-badge .sub { font-size: 10px; color: #94a3b8; text-transform: uppercase; }
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
                <a href="/" class="back-link">&larr; Dashboard</a>
                <div>
                    <h1>📊 S&P 500 AI Quant Predictions</h1>
                    <div style="color: #94a3b8; font-size: 13px; margin-top: 4px;">Dilated Archetype-Transformer | Universe: ${predictions.length} Stocks</div>
                </div>
            </div>
            <div class="header-actions">
                <button class="btn-action" id="btnUpdateModel" onclick="updateModel()">
                    <span id="updateIcon">⚡</span>
                    <span id="updateLabel">Re-Run Model Inference</span>
                </button>
                <span class="badge badge-green" id="serverBadge" style="font-size: 13px; padding: 6px 12px;">LIVE SERVER</span>
            </div>
        </div>

        <div class="stats-grid">
            <div class="stat-card">
                <div class="stat-label">Universe Size</div>
                <div class="stat-val" id="statUniverse">${predictions.length} Stocks</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Market Conviction Spread</div>
                <div class="stat-val" id="statSpread" style="color: #38bdf8;">${marketSpread.toFixed(4)}</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Target Long Holdings</div>
                <div class="stat-val" style="color: #4ade80;">${topK} Stocks</div>
            </div>
            <div class="stat-card">
                <div class="stat-label">Target Short Holdings</div>
                <div class="stat-val" style="color: #f87171;">${topK} Stocks</div>
            </div>
        </div>

        <div class="card" style="margin-bottom: 25px;">
            <div class="card-title">
                <span>🎯 Individual Stock Direction Predictor &amp; Trajectory Forecast</span>
                <span id="selectedStockTag" class="badge badge-green" style="font-size: 13px;">TOP LONG</span>
            </div>

            <div class="stock-selector-row">
                <div style="font-size: 13px; color: #94a3b8; font-weight: bold;">Quick Select:</div>
                <div class="stock-chips">
                    <span class="chip active" onclick="selectStock('AAPL')">AAPL</span>
                    <span class="chip" onclick="selectStock('NVDA')">NVDA</span>
                    <span class="chip" onclick="selectStock('TSLA')">TSLA</span>
                    <span class="chip" onclick="selectStock('MSFT')">MSFT</span>
                    <span class="chip" onclick="selectStock('AMZN')">AMZN</span>
                    <span class="chip" onclick="selectStock('GOOGL')">GOOGL</span>
                    <span class="chip" onclick="selectStock('META')">META</span>
                    <span class="chip" onclick="selectStock('JPM')">JPM</span>
                    <span class="chip" onclick="selectStock('AMD')">AMD</span>
                </div>
                <div style="display: flex; align-items: center; gap: 8px; margin-left: auto;">
                    <input type="text" id="customTickerInput" placeholder="Enter Ticker (e.g. NVDA)..." style="background:#0f172a; border:1px solid #475569; color:#fff; padding:6px 12px; border-radius:6px; font-size:12px; width:170px; text-transform:uppercase;">
                    <button class="btn-micro" style="padding: 6px 14px; font-size: 12px; height: 30px;" onclick="searchCustomStock()">Forecast &rarr;</button>
                </div>
            </div>

            <div class="stock-info-strip">
                <div class="stock-stat-badge">
                    <span class="sub">Ticker</span>
                    <span class="val" id="detailTicker" style="color: #38bdf8;">AAPL</span>
                </div>
                <div class="stock-stat-badge">
                    <span class="sub">5D Direction</span>
                    <span class="val" id="detailDirection" style="color: #4ade80;">▲ Bullish</span>
                </div>
                <div class="stock-stat-badge">
                    <span class="sub">Conviction (SNR)</span>
                    <span class="val" id="detailSnr">0.0000</span>
                </div>
                <div class="stock-stat-badge">
                    <span class="sub">Expected 5D Return</span>
                    <span class="val" id="detailReturn">+0.00%</span>
                </div>
                <div class="stock-stat-badge">
                    <span class="sub">Uncertainty (σ)</span>
                    <span class="val" id="detailUncertainty">0.00%</span>
                </div>
                <div class="stock-stat-badge">
                    <span class="sub">Universe Rank</span>
                    <span class="val" id="detailRank">#1</span>
                </div>
            </div>

            <canvas id="individualStockChart" height="110"></canvas>
        </div>

        <div class="grid-2">
            <div class="card">
                <div class="card-title">1. S&P 500 Cross-Section Risk vs. Return (All ${predictions.length} Stocks)</div>
                <canvas id="scatterChart" height="230"></canvas>
            </div>
            <div class="card">
                <div class="card-title">2. Top 25 Longs vs. Bottom 25 Shorts (Tail Conviction)</div>
                <canvas id="tailBarChart" height="230"></canvas>
            </div>
        </div>

        <div class="card">
            <div class="card-title">
                <span>3. Full S&P 500 Quant Predictions Table</span>
                <span style="font-size: 12px; color: #94a3b8; font-weight: normal;">Click any row to forecast individual stock</span>
            </div>
            <input type="text" id="searchInput" class="search-box" placeholder="🔍 Search by Ticker (e.g. AAPL, NVDA, TSLA)...">
            <div class="table-container">
                <table id="stocksTable">
                    <thead>
                        <tr>
                            <th>Rank</th>
                            <th>Ticker</th>
                            <th>Conviction (SNR)</th>
                            <th>Expected 5D Ret</th>
                            <th>Uncertainty (σ)</th>
                            <th>Target Group</th>
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
        let currentSelectedTicker = "${initialStock}";
        let socket = null;

        function showToast(msg) {
            const el = document.getElementById('toastNotify');
            el.textContent = msg;
            el.classList.add('visible');
            setTimeout(() => el.classList.remove('visible'), 3200);
        }

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
                showToast('Model predictions updated live!');
                refreshDashboardFromData(data);
            });
            socket.on('predictionsError', (err) => {
                showToast(err?.error || 'Prediction data unavailable');
            });
        } catch (e) {}

        const scatterData = ${scatterData};
        scatterChartInstance = new Chart(document.getElementById('scatterChart'), {
            type: 'scatter',
            data: {
                datasets: [{
                    label: 'S&P 500 Stock',
                    data: scatterData,
                    backgroundColor: scatterData.map(d => d.snr >= 0 ? 'rgba(56, 189, 248, 0.7)' : 'rgba(244, 63, 94, 0.7)'),
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
                                return p.ticker + ': Ret=' + p.y + '%, σ=' + p.x + ', SNR=' + p.snr;
                            }
                        }
                    }
                },
                scales: {
                    x: { title: { display: true, text: 'Uncertainty / Volatility (σ)', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                    y: { title: { display: true, text: 'Expected 5-Day Return (%)', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } }
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
                    y: { grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
                    x: { grid: { display: false }, ticks: { color: '#f8fafc', font: { size: 10 } } }
                }
            }
        });

        async function selectStock(ticker) {
            currentSelectedTicker = ticker;
            document.querySelectorAll('.chip').forEach(c => {
                c.classList.toggle('active', c.textContent.trim().toUpperCase() === ticker.toUpperCase());
            });

            try {
                const res = await fetch('/api/stock-forecast/' + encodeURIComponent(ticker));
                if (res.status === 202) {
                    showToast('Predictions still initializing — retrying...');
                    setTimeout(() => selectStock(ticker), 3000);
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
            document.getElementById('detailSnr').textContent = (data.snr >= 0 ? '+' : '') + data.snr.toFixed(4);
            document.getElementById('detailReturn').textContent = (data.expectedReturn5d >= 0 ? '+' : '') + data.expectedReturn5d.toFixed(2) + '%';
            document.getElementById('detailReturn').style.color = data.expectedReturn5d >= 0 ? '#4ade80' : '#f87171';
            document.getElementById('detailUncertainty').textContent = data.uncertainty5d.toFixed(4);

            const dirEl = document.getElementById('detailDirection');
            dirEl.textContent = (data.snr >= 0 ? '▲ ' : '▼ ') + data.direction;
            dirEl.style.color = data.snr >= 0 ? '#4ade80' : '#f87171';

            const tagEl = document.getElementById('selectedStockTag');
            tagEl.textContent = data.group.toUpperCase();
            if (data.group === 'Top Long') tagEl.className = 'badge badge-green';
            else if (data.group === 'Top Short') tagEl.className = 'badge badge-red';
            else tagEl.className = 'badge badge-neutral';

            const isBullish = data.snr >= 0;
            const primaryColor = isBullish ? '#38bdf8' : '#f43f5e';
            const projColor = isBullish ? '#4ade80' : '#f87171';
            const bandFill = isBullish ? 'rgba(56, 189, 248, 0.12)' : 'rgba(244, 63, 94, 0.12)';

            if (individualStockChartInstance) {
                individualStockChartInstance.destroy();
            }

            individualStockChartInstance = new Chart(document.getElementById('individualStockChart'), {
                type: 'line',
                data: {
                    labels: data.labels,
                    datasets: [
                        { label: 'Upper Uncertainty Band (+σ)', data: data.upperBand, borderColor: 'transparent', backgroundColor: bandFill, fill: '+1', pointRadius: 0 },
                        { label: 'Lower Uncertainty Band (-σ)', data: data.lowerBand, borderColor: 'transparent', backgroundColor: 'transparent', fill: false, pointRadius: 0 },
                        { label: 'Historical Price', data: data.historical, borderColor: primaryColor, backgroundColor: primaryColor, borderWidth: 2.5, tension: 0.25, pointRadius: 2.5 },
                        { label: 'Predicted Trajectory (Horizon μ)', data: data.projection, borderColor: projColor, backgroundColor: projColor, borderWidth: 3, borderDash: [5, 5], tension: 0.25, pointRadius: 4 }
                    ]
                },
                options: {
                    responsive: true,
                    interaction: { mode: 'index', intersect: false },
                    plugins: { legend: { labels: { filter: (item) => !item.text.includes('Band'), color: '#e2e8f0' } } },
                    scales: {
                        y: { title: { display: true, text: 'Price Level', color: '#94a3b8' }, grid: { color: '#334155' }, ticks: { color: '#94a3b8' } },
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
                    showToast('Inference re-run complete (' + data.latency + 'ms)!');
                } else {
                    showToast(data.message || 'Inference started — updates will arrive shortly');
                }
            } catch (err) {
                console.error(err);
                showToast('Error re-running inference');
            } finally {
                btn.disabled = false;
                icon.textContent = '⚡';
                label.textContent = 'Re-Run Model Inference';
            }
        }

        function refreshDashboardFromData(data) {
            if (!data || !data.predictions) return;
            document.getElementById('statSpread').textContent = data.marketSpread.toFixed(4);

            const predictions = data.predictions;
            const topTails = [...predictions.slice(0, 25), ...predictions.slice(-25)];
            const tailLabels = topTails.map(p => p.ticker);
            const tailSnr = topTails.map(p => Number(p.snr.toFixed(4)));
            const tailColors = topTails.map((p, i) => i < 25 ? 'rgba(34, 197, 94, 0.85)' : 'rgba(239, 68, 68, 0.85)');

            tailBarChartInstance.data.labels = tailLabels;
            tailBarChartInstance.data.datasets[0].data = tailSnr;
            tailBarChartInstance.data.datasets[0].backgroundColor = tailColors;
            tailBarChartInstance.update();

            const newScatter = predictions.map(p => ({
                x: Number(p.uncertainty5d.toFixed(3)),
                y: Number(p.expectedReturn5d.toFixed(2)),
                ticker: p.ticker,
                snr: Number(p.snr.toFixed(4))
            }));
            scatterChartInstance.data.datasets[0].data = newScatter;
            scatterChartInstance.data.datasets[0].backgroundColor = newScatter.map(d => d.snr >= 0 ? 'rgba(56, 189, 248, 0.7)' : 'rgba(244, 63, 94, 0.7)');
            scatterChartInstance.update();

            const topK = data.topK;
            const tableBody = document.getElementById('stocksTableBody');
            tableBody.innerHTML = predictions.map((p, idx) => {
                let tag = '<span class="badge badge-neutral">Neutral</span>';
                if (idx < topK) tag = '<span class="badge badge-green">Top Long</span>';
                else if (idx >= predictions.length - topK) tag = '<span class="badge badge-red">Short</span>';

                const rowTicker = String(p.ticker).replace(/\\\\/g, '\\\\\\\\').replace(/'/g, "\\\\'");
                return '<tr data-ticker="' + rowTicker + '" style="cursor: pointer;" onclick="selectStock(\\'' + rowTicker + '\\')">' +
                    '<td><strong>' + (idx + 1) + '</strong></td>' +
                    '<td><strong>' + rowTicker + '</strong></td>' +
                    '<td style="color:' + (p.snr >= 0 ? '#4ade80' : '#f87171') + '">' + p.snr.toFixed(4) + '</td>' +
                    '<td>' + (p.expectedReturn5d >= 0 ? '+' : '') + p.expectedReturn5d.toFixed(2) + '%</td>' +
                    '<td>' + p.uncertainty5d.toFixed(4) + '</td>' +
                    '<td>' + tag + '</td>' +
                    '<td><button class="btn-micro" onclick="event.stopPropagation(); selectStock(\\'' + rowTicker + '\\')">Forecast</button></td>' +
                '</tr>';
            }).join('');

            selectStock(currentSelectedTicker);
        }

        document.getElementById('searchInput').addEventListener('keyup', function() {
            const filter = this.value.toUpperCase();
            const rows = document.querySelectorAll('#stocksTable tbody tr');
            rows.forEach(row => {
                const ticker = row.cells[1].textContent.trim();
                row.style.display = ticker.toUpperCase().includes(filter) ? '' : 'none';
            });
        });

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

        selectStock(currentSelectedTicker);
    </script>
</body>
</html>`;
}

module.exports = { buildPredictionHtmlReport };