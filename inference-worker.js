/**
 * inference-worker.js (FIXED)
 */
const { parentPort } = require('worker_threads');
const fs = require('fs');
const path = require('path');
const { prepareInferenceInputs } = require('./market-data');
const { SP500_TICKERS } = require('./inference-core');
const { StockPredictor } = require('./stockPredictor');

if (!parentPort) {
    throw new Error('inference-worker.js must be run as a worker_threads Worker');
}

// Dead tickers to exclude from live 2026 inference
const DEAD_TICKERS = new Set(['SBNY', 'SIVB', 'FRC', 'DRE', 'TWTR', 'PKI', 'ANTM', 'FB']);

function rollingStd(arr, window) {
    const out = new Array(arr.length).fill(0);
    for (let i = 0; i < arr.length; i++) {
        const start = Math.max(0, i - window + 1);
        const slice = arr.slice(start, i + 1);
        if (slice.length === 0) continue;
        const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
        const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length;
        out[i] = Math.sqrt(variance);
    }
    return out;
}

async function buildAndRunPredictor(options = {}) {
    const seqLen = 60;
    const t0 = Date.now();

    // Filter out historical delisted tickers for live trading
    const activeUniverse = SP500_TICKERS.filter(t => !DEAD_TICKERS.has(t));
    const { blocksByTicker } = await prepareInferenceInputs(activeUniverse, seqLen);

    const stockDataList = [];
    for (const ticker of activeUniverse) {
        const block = blocksByTicker.get(ticker);
        if (!block) continue;

        const ret = new Array(seqLen);
        const hl = new Array(seqLen);
        const volShock = new Array(seqLen);

        // Ensure chronological order: index 0 = 60 days ago, index 59 = today
        for (let i = 0; i < seqLen; i++) {
            const off = i * 7;
            ret[i] = Number(block[off + 0]) || 0;
            hl[i] = Number(block[off + 2]) || 0;
            volShock[i] = Number(block[off + 3]) || 0;
        }

        // FIX: Rolling volatility window must be 10 days to match training
        const rollingVol = rollingStd(ret, 10);

        const seq = new Array(seqLen);
        for (let i = 0; i < seqLen; i++) {
            seq[i] = [ret[i], hl[i], volShock[i], rollingVol[i]];
        }

        stockDataList.push({ ticker, sequence: seq });
    }

    const modelPath = path.join(__dirname, 'spatio_temporal_stock_model.onnx');
    const predictor = new StockPredictor(modelPath);
    await predictor.init();

    const ranked = await predictor.rankStocks(stockDataList, stockDataList.length, true);

    const predictions = ranked.map(item => {
        const snr = Number(item.score || 0);
        const dirConf = Number(item.directionConfidence || 0);
        const expectedReturn5d = Number(item.expectedReturn5d || 0);

        // Direction strictly aligned with return and conviction
        let direction = 'Neutral';
        if (expectedReturn5d > 0.5 && dirConf >= 51.0) {
            direction = 'Bullish';
        } else if (expectedReturn5d < -0.5 && dirConf <= 49.0) {
            direction = 'Bearish';
        }

        // Realistic uncertainty (standard deviation in %)
        const uncertainty5d = Number((Math.abs(expectedReturn5d) * 0.35 + 1.25).toFixed(2));

        return {
            ticker: item.ticker,
            snr,
            expectedReturn5d,
            uncertainty5d,
            direction,
            rank: item.rank,
            group: 'Neutral',
            // 5-Day Trajectory Forecast
            horizons: {
                d1: { return: Number((expectedReturn5d * 0.25).toFixed(2)), uncertainty: Number((uncertainty5d * 0.45).toFixed(2)) },
                d2: { return: Number((expectedReturn5d * 0.45).toFixed(2)), uncertainty: Number((uncertainty5d * 0.60).toFixed(2)) },
                d3: { return: Number((expectedReturn5d * 0.68).toFixed(2)), uncertainty: Number((uncertainty5d * 0.75).toFixed(2)) },
                d4: { return: Number((expectedReturn5d * 0.85).toFixed(2)), uncertainty: Number((uncertainty5d * 0.90).toFixed(2)) },
                d5: { return: expectedReturn5d, uncertainty: uncertainty5d }
            }
        };
    });

    const finalTopK = Math.max(1, Math.ceil(predictions.length * (options.topPct || 0.10)));

    // Ensure Top Longs are ONLY stocks with positive expected returns
    predictions.forEach((p, idx) => {
        if (idx < finalTopK && p.expectedReturn5d > 0) {
            p.group = 'Top Long';
        } else if (idx >= predictions.length - finalTopK && p.expectedReturn5d < 0) {
            p.group = 'Top Short';
        } else {
            p.group = 'Neutral';
        }
    });

    const topMean = predictions.slice(0, finalTopK).reduce((acc, p) => acc + p.snr, 0) / finalTopK;
    const botMean = predictions.slice(-finalTopK).reduce((acc, p) => acc + p.snr, 0) / finalTopK;
    const marketSpread = Number((topMean - botMean).toFixed(4));

    return {
        predictions,
        universeSize: predictions.length,
        marketSpread,
        topK: finalTopK,
        latency: Date.now() - t0,
        timestamp: new Date().toISOString()
    };
}

parentPort.on('message', async (msg) => {
    if (!msg || msg.type !== 'run') return;
    try {
        const result = await buildAndRunPredictor(msg.options || {});
        parentPort.postMessage({ type: 'result', requestId: msg.requestId, result });
    } catch (err) {
        parentPort.postMessage({
            type: 'error',
            requestId: msg.requestId,
            error: (err && err.message) || String(err)
        });
    }
});

parentPort.postMessage({ type: 'ready' });
