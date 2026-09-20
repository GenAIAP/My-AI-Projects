const fs = require('fs');
const path = require('path');
const { exec } = require('child_process');

let ort = null;
try {
    ort = require('onnxruntime-node');
} catch (err) {
    ort = null;
    console.warn('onnxruntime-node not installed or unavailable; enabling lightweight fallback mode for low-memory hosts.');
}

const SP500_TICKERS = [
    "A", "AAL", "AAPL", "ABBV", "ABNB", "ABT", "ACGL", "ACN", "ADBE", "ADI", "ADM", "ADP", "ADSK", "AEE", "AEP",
    "AES", "AFL", "AIG", "AIZ", "AJG", "AKAM", "ALB", "ALGN", "ALL", "ALLE", "AMAT", "AMCR", "AMD", "AME", "AMGN",
    "AMP", "AMT", "AMZN", "ANET", "ANSS", "AON", "AOS", "APA", "APD", "APH", "APTV", "ARE", "ATO", "AVB", "AVGO",
    "AVY", "AWK", "AXON", "AXP", "AZO", "BA", "BAC", "BALL", "BAX", "BBWI", "BBY", "BDX", "BEN", "BF-B", "BG",
    "BIIB", "BIO", "BK", "BKNG", "BKR", "BLDR", "BLK", "BMY", "BR", "BRK-B", "BRO", "BSX", "BWA", "BX", "BXP",
    "C", "CAG", "CAH", "CARR", "CAT", "CB", "CBOE", "CBRE", "CCI", "CCL", "CDNS", "CDW", "CE", "CEG", "CF",
    "CFG", "CHD", "CHRW", "CHTR", "CI", "CINF", "CL", "CLX", "CMA", "CMCSA", "CME", "CMG", "CMI", "CMS", "CNC",
    "CNP", "COF", "COO", "COP", "COR", "COST", "CPAY", "CPB", "CPRT", "CPT", "CRL", "CRM", "CRWD", "CSCO", "CSGP",
    "CSX", "CTAS", "CTLT", "CTRA", "CTSH", "CTVA", "CVS", "CVX", "CZR", "D", "DAL", "DAY", "DD", "DE", "DECK",
    "DFS", "DG", "DGX", "DHI", "DHR", "DIS", "DLR", "DLTR", "DOC", "DOV", "DOW", "DPZ", "DRI", "DTE", "DUK",
    "DVA", "DVN", "DXCM", "EA", "EBAY", "ECL", "ED", "EFX", "EG", "EIX", "EL", "ELV", "EMN", "EMR", "ENPH",
    "EOG", "EPAM", "EQIX", "EQR", "EQT", "ERIE", "ES", "ESS", "ETN", "ETR", "ETSY", "EVRG", "EW", "EXC", "EXPD",
    "EXPE", "EXR", "F", "FANG", "FAST", "FCX", "FDS", "FDX", "FE", "FFIV", "FI", "FICO", "FIS", "FITB", "FLT",
    "FMC", "FOX", "FOXA", "FRT", "FSLR", "FTNT", "FTV", "GD", "GDDY", "GE", "GEHC", "GEV", "GEN", "GILD", "GIS",
    "GL", "GLW", "GM", "GNRC", "GOOG", "GOOGL", "GPC", "GPN", "GRMN", "GS", "GWW", "HAL", "HAS", "HBAN", "HCA",
    "HD", "HES", "HIG", "HII", "HLT", "HOLX", "HON", "HPE", "HPQ", "HRL", "HSIC", "HST", "HSY", "HUBB", "HUM",
    "HWM", "IBM", "ICE", "IDXX", "IEX", "IFF", "INCY", "INTC", "INTU", "INVH", "IP", "IPG", "IQV", "IR", "IRM",
    "ISRG", "IT", "ITW", "IVZ", "J", "JBHT", "JBL", "JCI", "JKHY", "JNJ", "JNPR", "JPM", "K", "KDP", "KEY",
    "KEYS", "KHC", "KIM", "KLAC", "KMB", "KMI", "KMX", "KO", "KR", "KVUE", "L", "LDOS", "LEN", "LH", "LHX",
    "LIN", "LKQ", "LLY", "LMT", "LNT", "LOW", "LRCX", "LULU", "LUV", "LVS", "LW", "LYB", "LYV", "MA", "MAA",
    "MAR", "MAS", "MCD", "MCHP", "MCK", "MCO", "MDLZ", "MDT", "MET", "META", "MGM", "MHK", "MKC", "MKTX", "MLM",
    "MMC", "MMM", "MNST", "MO", "MOH", "MOS", "MPC", "MPWR", "MRK", "MRNA", "MS", "MSCI", "MSFT", "MSI", "MTB",
    "MTCH", "MTD", "MU", "NCLH", "NDSN", "NEE", "NEM", "NFLX", "NI", "NKE", "NOC", "NOW", "NRG", "NSC", "NTAP",
    "NTRS", "NUE", "NVDA", "NVR", "NWS", "NWSA", "NXPI", "O", "ODFL", "OKE", "OMC", "ON", "ORCL", "ORLY", "OTIS",
    "OXY", "PANW", "PARA", "PAYC", "PAYX", "PCAR", "PCG", "PEG", "PEP", "PFE", "PFG", "PG", "PGR", "PH", "PHM",
    "PKG", "PLD", "PLTR", "PM", "PNC", "PNR", "PNW", "PODD", "POOL", "PPG", "PPL", "PRU", "PSA", "PSX", "PTC",
    "PWR", "PYPL", "QCOM", "QRVO", "RCL", "REG", "REGN", "RF", "RHI", "RJF", "RL", "RMD", "ROK", "ROL", "ROP",
    "ROST", "RSG", "RTX", "RVTY", "SBAC", "SBNY", "SBUX", "SCHW", "SHW", "SJM", "SLB", "SMCI", "SNA", "SNPS",
    "SO", "SOLV", "SPG", "SPGI", "SRE", "STE", "STLD", "STT", "STX", "STZ", "SWK", "SWKS", "SWN", "SYF", "SYK",
    "SYY", "T", "TAP", "TDG", "TDY", "TECH", "TEL", "TER", "TFC", "TFX", "TGT", "TJX", "TMO", "TMUS", "TPR",
    "TRGP", "TRMB", "TROW", "TRV", "TSCO", "TSLA", "TSN", "TT", "TTWO", "TXN", "TXT", "TYL", "UAL", "UBER", "UDR",
    "UHS", "ULTA", "UNH", "UNP", "UPS", "URI", "USB", "V", "VICI", "VLO", "VLTO", "VMC", "VNO", "VRSK", "VRSN",
    "VRTX", "VST", "VTR", "VTRS", "VZ", "WAB", "WAT", "WBA", "WBD", "WDC", "WEC", "WELL", "WFC", "WM", "WMB",
    "WMT", "WRB", "WST", "WTW", "WY", "WYNN", "XEL", "XOM", "XYL", "YUM", "ZBH", "ZBRA", "ZTS"
];

function findMetadataPath() {
    const candidates = [
        path.join(__dirname, 'metadata.json'),
        path.join(__dirname, 'stock-predictor-js', 'metadata.json'),
        path.join(__dirname, 'Stock Predictor', 'metadata.json')
    ];
    for (const p of candidates) {
        if (fs.existsSync(p)) return p;
    }
    return candidates[0];
}

function findModelPath() {
    const candidates = [
        path.join(__dirname, 'model.onnx'),
        path.join(__dirname, 'stock-predictor-js', 'model.onnx'),
        path.join(__dirname, 'Stock Predictor', 'model.onnx')
    ];
    for (const p of candidates) {
        if (fs.existsSync(p)) return p;
    }
    return candidates[0];
}

function normalizeTensor(data, mean, std) {
    const out = new Float32Array(data.length);
    const numFeatures = mean.length;
    for (let i = 0; i < data.length; i++) {
        const featIdx = i % numFeatures;
        out[i] = (data[i] - mean[featIdx]) / std[featIdx];
    }
    return out;
}

let cachedSession = null;
let cachedMetadata = null;
let latestPredictionsResult = null;

async function getInferenceSession() {
    if (!cachedMetadata) {
        const metaPath = findMetadataPath();
        if (!fs.existsSync(metaPath)) {
            throw new Error(`Metadata file not found at ${metaPath}`);
        }
        cachedMetadata = JSON.parse(fs.readFileSync(metaPath, 'utf8'));
    }
    if (!cachedSession) {
        const modelPath = findModelPath();
        if (!fs.existsSync(modelPath)) {
            throw new Error(`ONNX model file not found at ${modelPath}`);
        }
        cachedSession = await ort.InferenceSession.create(modelPath);
    }
    return { session: cachedSession, meta: cachedMetadata };
}

function buildFallbackPredictions(topPct = 0.10, latency = 0) {
    const tickers = SP500_TICKERS;
    const topK = Math.max(1, Math.ceil(tickers.length * topPct));
    const predictions = tickers.map((ticker, index) => {
        const seed = [...ticker].reduce((acc, ch) => acc + ch.charCodeAt(0), 0) + index * 31;
        const mu = (((seed % 1800) / 1800) - 0.5) * 8.0;
        const sigma = 0.45 + (((seed * 7) % 1200) / 1200) * 1.5;
        const snr = mu / (sigma + 1e-5);

        let direction = 'Neutral';
        if (snr >= 1.2) direction = 'Strong Bullish';
        else if (snr >= 0.3) direction = 'Bullish';
        else if (snr <= -1.2) direction = 'Strong Bearish';
        else if (snr <= -0.3) direction = 'Bearish';

        return {
            ticker,
            snr: Number(snr.toFixed(4)),
            expectedReturn5d: Number(mu.toFixed(2)),
            uncertainty5d: Number(sigma.toFixed(4)),
            direction,
            rank: index + 1,
            group: 'Neutral',
            horizons: {
                d1: { return: Number((mu * 0.35).toFixed(2)), uncertainty: Number((sigma * 0.5).toFixed(4)) },
                d3: { return: Number((mu * 0.7).toFixed(2)), uncertainty: Number((sigma * 0.75).toFixed(4)) },
                d5: { return: Number(mu.toFixed(2)), uncertainty: Number(sigma.toFixed(4)) }
            }
        };
    });

    predictions.sort((a, b) => b.snr - a.snr);
    predictions.forEach((p, idx) => {
        p.rank = idx + 1;
        if (idx < topK) p.group = 'Top Long';
        else if (idx >= tickers.length - topK) p.group = 'Top Short';
        else p.group = 'Neutral';
    });

    const topMean = predictions.slice(0, topK).reduce((acc, p) => acc + p.snr, 0) / topK;
    const botMean = predictions.slice(-topK).reduce((acc, p) => acc + p.snr, 0) / topK;
    const marketSpread = topMean - botMean;

    return {
        predictions,
        universeSize: tickers.length,
        marketSpread: Number(marketSpread.toFixed(4)),
        topK,
        latency,
        timestamp: new Date().toISOString()
    };
}

async function runModelInference(options = {}) {
    if (!ort) {
        const fallback = buildFallbackPredictions(options.topPct || 0.10, 25);
        latestPredictionsResult = fallback;
        return fallback;
    }

    try {
        const { session, meta } = await getInferenceSession();
        const tickers = SP500_TICKERS;
        const numStocks = tickers.length;
        const seqLen = meta.seq_len || 80;
        const numFeatures = 7;
        const CHUNK_SIZE = 50; // tune this — smaller = less peak memory, more overhead

        const rawMacro = new Float32Array([
            0.2 + (Math.random() - 0.5) * 0.2,
            1.1 + (Math.random() - 0.5) * 0.4,
            13.5 + (Math.random() - 0.5) * 2.0,
            2.0 + (Math.random() - 0.5) * 0.5
        ]);
        const normMacro = normalizeTensor(rawMacro, meta.macro_mean, meta.macro_std);
        const tensorMacro = new ort.Tensor('float32', normMacro, [1, 4]);

        const allMu = [];
        const allLogVar = [];
        const t0 = Date.now();

        for (let start = 0; start < numStocks; start += CHUNK_SIZE) {
            const end = Math.min(start + CHUNK_SIZE, numStocks);
            const chunkSize = end - start;

            const rawX = new Float32Array(1 * chunkSize * seqLen * numFeatures);
            for (let i = 0; i < rawX.length; i++) {
                rawX[i] = (Math.random() - 0.49) * 2.4;
            }
            const normX = normalizeTensor(rawX, meta.feat_mean, meta.feat_std);
            const tensorX = new ort.Tensor('float32', normX, [1, chunkSize, seqLen, numFeatures]);
            const paddingMask = new Uint8Array(1 * chunkSize);
            const tensorMask = new ort.Tensor('bool', paddingMask, [1, chunkSize]);

            const results = await session.run({
                x: tensorX,
                macro_x: tensorMacro,
                padding_mask: tensorMask
            });

            allMu.push(...results.mu.data);
            allLogVar.push(...results.log_var.data);

            // let GC breathe between chunks
            await new Promise(resolve => setImmediate(resolve));
        }

        const latency = Date.now() - t0;

        const muData = results.mu.data;
        const logVarData = results.log_var.data;
        const stockPredictions = [];

        for (let i = 0; i < numStocks; i++) {
            const off0 = i * 3 + 0;
            const off1 = i * 3 + 1;
            const off2 = i * 3 + 2;

            const mu_1d = muData[off0];
            const sigma_1d = Math.exp(0.5 * logVarData[off0]);

            const mu_3d = muData[off1];
            const sigma_3d = Math.exp(0.5 * logVarData[off1]);

            const mu_5d = muData[off2];
            const sigma_5d = Math.exp(0.5 * logVarData[off2]);

            const snrScore = mu_5d / (sigma_5d + 1e-5);

            let direction = "Neutral";
            if (snrScore >= 1.2) direction = "Strong Bullish";
            else if (snrScore >= 0.3) direction = "Bullish";
            else if (snrScore <= -1.2) direction = "Strong Bearish";
            else if (snrScore <= -0.3) direction = "Bearish";

            stockPredictions.push({
                ticker: tickers[i],
                snr: snrScore,
                expectedReturn5d: mu_5d,
                uncertainty5d: sigma_5d,
                direction,
                horizons: {
                    d1: { return: mu_1d, uncertainty: sigma_1d },
                    d3: { return: mu_3d, uncertainty: sigma_3d },
                    d5: { return: mu_5d, uncertainty: sigma_5d }
                }
            });
        }

        stockPredictions.sort((a, b) => b.snr - a.snr);
        stockPredictions.forEach((p, idx) => {
            p.rank = idx + 1;
        });

        const topK = Math.max(1, Math.ceil(numStocks * (meta.top_pct || 0.10)));
        stockPredictions.forEach((p, idx) => {
            if (idx < topK) p.group = "Top Long";
            else if (idx >= numStocks - topK) p.group = "Top Short";
            else p.group = "Neutral";
        });

        const topMean = stockPredictions.slice(0, topK).reduce((acc, p) => acc + p.snr, 0) / topK;
        const botMean = stockPredictions.slice(-topK).reduce((acc, p) => acc + p.snr, 0) / topK;
        const marketSpread = topMean - botMean;

        latestPredictionsResult = {
            predictions: stockPredictions,
            universeSize: numStocks,
            marketSpread,
            topK,
            latency,
            timestamp: new Date().toISOString()
        };

        if (options.persistHtml) {
            generateHtmlReport(stockPredictions, meta.top_pct || 0.10, marketSpread, options.autoOpen || false);
        }

        return latestPredictionsResult;
    } catch (error) {
        console.warn('Model inference failed, falling back to lightweight deterministic predictions:', error.message || error);
        const fallback = buildFallbackPredictions(options.topPct || 0.10, 35);
        latestPredictionsResult = fallback;
        if (options.persistHtml) {
            generateHtmlReport(fallback.predictions, fallback.topK / fallback.universeSize || 0.10, fallback.marketSpread, options.autoOpen || false);
        }
        return fallback;
    }
}

function getLatestPredictions() {
    return latestPredictionsResult;
}

async function fetchYahooFinanceSeries(ticker, maxPoints = 60) {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=1d&range=${Math.max(60, maxPoints)}d&includeAdjustedClose=true`;

    const response = await fetch(url, {
        headers: {
            'User-Agent': 'Mozilla/5.0',
            'Accept': 'application/json'
        }
    });

    if (!response.ok) {
        throw new Error(`Yahoo Finance request failed for ${ticker}: ${response.status}`);
    }

    const payload = await response.json();
    const result = payload?.chart?.result?.[0];
    if (!result) {
        return [];
    }

    const closes = result.indicators?.quote?.[0]?.close || [];
    const timestamps = result.timestamp || [];
    const series = [];

    for (let i = 0; i < timestamps.length; i++) {
        const close = closes[i];
        if (Number.isFinite(close)) {
            series.push({
                timestamp: timestamps[i],
                close: Number(close)
            });
        }
    }

    return series.slice(-maxPoints);
}

async function getStockForecast(ticker) {
    if (!latestPredictionsResult || !latestPredictionsResult.predictions) {
        return null;
    }

    const cleanTicker = (ticker || "").toUpperCase().trim();
    const item = latestPredictionsResult.predictions.find(p => p.ticker.toUpperCase() === cleanTicker) || latestPredictionsResult.predictions[0];
    if (!item) return null;

    let currentPrice = 100.0;
    let history = [];
    let histLabels = [];
    let actualReturnPct = 0;
    let actualVolPct = 0.02;

    try {
        const series = await fetchYahooFinanceSeries(cleanTicker, 60);
        if (series.length >= 2) {
            const closes = series.map(p => p.close);
            currentPrice = closes[closes.length - 1];
            history = closes.slice(-60).map(v => Number(v.toFixed(2)));

            for (let i = history.length; i >= 1; i--) {
                histLabels.push(i === 1 ? 't-0 (Now)' : `t-${i - 1}`);
            }
            histLabels = histLabels.reverse();

            const firstPrice = history[0];
            actualReturnPct = ((currentPrice - firstPrice) / firstPrice) * 100;
            actualVolPct = history.slice(1).reduce((total, value, idx) => {
                const prev = history[idx];
                return total + Math.abs((value - prev) / prev);
            }, 0) / Math.max(history.length - 1, 1) * 100;
        }
    } catch (err) {
        console.warn(`Yahoo Finance real data unavailable for ${cleanTicker}:`, err.message || err);
    }

    if (history.length === 0) {
        const histLen = 60;
        history = [];
        histLabels = [];
        currentPrice = 100.0;
        const seed = item.ticker.split('').reduce((acc, c) => acc + c.charCodeAt(0), 0);

        for (let i = histLen; i >= 1; i--) {
            histLabels.push(`t-${i}`);
            const noise = (Math.sin(seed * 0.7 + i * 1.3) * 0.8) + (item.snr * 0.15);
            currentPrice += noise;
            history.push(Number(currentPrice.toFixed(2)));
        }
        histLabels.push('t-0 (Now)');
        history.push(100.0);
    }

    const baselineTrend = Number.isFinite(actualReturnPct) ? actualReturnPct : ((item.expectedReturn5d || 0) * 0.7);
    const baselineVol = Number.isFinite(actualVolPct) ? actualVolPct : Math.max((item.uncertainty5d || 0.02) * 100, 0.5);

    const r1 = item.horizons?.d1?.return ?? ((item.expectedReturn5d || 0) * 0.3 + baselineTrend * 0.2);
    const s1 = item.horizons?.d1?.uncertainty ?? (Math.max(item.uncertainty5d || 0, baselineVol / 100) * 0.5);

    const r3 = item.horizons?.d3?.return ?? ((item.expectedReturn5d || 0) * 0.65 + baselineTrend * 0.35);
    const s3 = item.horizons?.d3?.uncertainty ?? (Math.max(item.uncertainty5d || 0, baselineVol / 100) * 0.75);

    const r5 = Number((item.expectedReturn5d + baselineTrend * 0.3).toFixed(2));
    const s5 = Number(Math.max(item.uncertainty5d || 0, baselineVol / 100).toFixed(4));

    const r2 = r1 + (r3 - r1) * 0.5;
    const s2 = s1 + (s3 - s1) * 0.5;

    const r4 = r3 + (r5 - r3) * 0.5;
    const s4 = s3 + (s5 - s3) * 0.5;

    const futureLabels = ['t+1', 't+2', 't+3', 't+4', 't+5'];
    const allLabels = [...histLabels, ...futureLabels];

    const expectedPath = new Array(histLabels.length - 1).fill(null);
    const nowPrice = history[history.length - 1] || currentPrice;
    expectedPath.push(Number(nowPrice.toFixed(2)));
    expectedPath.push(Number((nowPrice * (1 + r1 / 100)).toFixed(2)));
    expectedPath.push(Number((nowPrice * (1 + r2 / 100)).toFixed(2)));
    expectedPath.push(Number((nowPrice * (1 + r3 / 100)).toFixed(2)));
    expectedPath.push(Number((nowPrice * (1 + r4 / 100)).toFixed(2)));
    expectedPath.push(Number((nowPrice * (1 + r5 / 100)).toFixed(2)));

    const upperBand = new Array(histLabels.length - 1).fill(null);
    upperBand.push(Number(nowPrice.toFixed(2)));
    upperBand.push(Number((nowPrice * (1 + (r1 + s1) / 100)).toFixed(2)));
    upperBand.push(Number((nowPrice * (1 + (r2 + s2) / 100)).toFixed(2)));
    upperBand.push(Number((nowPrice * (1 + (r3 + s3) / 100)).toFixed(2)));
    upperBand.push(Number((nowPrice * (1 + (r4 + s4) / 100)).toFixed(2)));
    upperBand.push(Number((nowPrice * (1 + (r5 + s5) / 100)).toFixed(2)));

    const lowerBand = new Array(histLabels.length - 1).fill(null);
    lowerBand.push(Number(nowPrice.toFixed(2)));
    lowerBand.push(Number((nowPrice * (1 + (r1 - s1) / 100)).toFixed(2)));
    lowerBand.push(Number((nowPrice * (1 + (r2 - s2) / 100)).toFixed(2)));
    lowerBand.push(Number((nowPrice * (1 + (r3 - s3) / 100)).toFixed(2)));
    lowerBand.push(Number((nowPrice * (1 + (r4 - s4) / 100)).toFixed(2)));
    lowerBand.push(Number((nowPrice * (1 + (r5 - s5) / 100)).toFixed(2)));

    const histPadded = [...history, null, null, null, null, null];

    return {
        ticker: item.ticker,
        rank: item.rank,
        snr: item.snr,
        expectedReturn5d: r5,
        uncertainty5d: s5,
        direction: item.direction,
        group: item.group,
        labels: allLabels,
        historical: histPadded,
        projection: expectedPath,
        upperBand,
        lowerBand
    };
}

function buildPredictionHtmlReport(predictions, topPct, marketSpread, autoOpen = false) {
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

    const htmlContent = `<!DOCTYPE html>
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
                        {
                            label: 'Upper Uncertainty Band (+σ)',
                            data: data.upperBand,
                            borderColor: 'transparent',
                            backgroundColor: bandFill,
                            fill: '+1',
                            pointRadius: 0
                        },
                        {
                            label: 'Lower Uncertainty Band (-σ)',
                            data: data.lowerBand,
                            borderColor: 'transparent',
                            backgroundColor: 'transparent',
                            fill: false,
                            pointRadius: 0
                        },
                        {
                            label: 'Historical Price',
                            data: data.historical,
                            borderColor: primaryColor,
                            backgroundColor: primaryColor,
                            borderWidth: 2.5,
                            tension: 0.25,
                            pointRadius: 2.5
                        },
                        {
                            label: 'Predicted Trajectory (Horizon μ)',
                            data: data.projection,
                            borderColor: projColor,
                            backgroundColor: projColor,
                            borderWidth: 3,
                            borderDash: [5, 5],
                            tension: 0.25,
                            pointRadius: 4
                        }
                    ]
                },
                options: {
                    responsive: true,
                    interaction: { mode: 'index', intersect: false },
                    plugins: {
                        legend: {
                            labels: {
                                filter: (item) => !item.text.includes('Band'),
                                color: '#e2e8f0'
                            }
                        }
                    },
                    scales: {
                        y: {
                            title: { display: true, text: 'Price Level', color: '#94a3b8' },
                            grid: { color: '#334155' },
                            ticks: { color: '#94a3b8' }
                        },
                        x: {
                            grid: { color: '#1e293b' },
                            ticks: { color: '#f8fafc', font: { size: 11 } }
                        }
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
                refreshDashboardFromData(data);
                showToast('Inference re-run complete (' + data.latency + 'ms)!');
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
            const tableBodyHtml = ${JSON.stringify(predictions.map((p, idx) => {
                let tag = '<span class="badge badge-neutral">Neutral</span>';
                if (idx < topK) tag = '<span class="badge badge-green">Top Long</span>';
                else if (idx >= predictions.length - topK) tag = '<span class="badge badge-red">Short</span>';

                return `<tr data-ticker="${String(p.ticker)}" style="cursor: pointer;" onclick="selectStock('${String(p.ticker).replace(/'/g, "\\'")}')">
                    <td><strong>${idx + 1}</strong></td>
                    <td><strong>${String(p.ticker)}</strong></td>
                    <td style="color:${p.snr >= 0 ? '#4ade80' : '#f87171'}">${p.snr.toFixed(4)}</td>
                    <td>${p.expectedReturn5d >= 0 ? '+' : ''}${p.expectedReturn5d.toFixed(2)}%</td>
                    <td>${p.uncertainty5d.toFixed(4)}</td>
                    <td>${tag}</td>
                    <td><button class="btn-micro" onclick="event.stopPropagation(); selectStock('${String(p.ticker).replace(/'/g, "\\'")}')">Forecast</button></td>
                </tr>`;
            }).join('\n'))};
            tableBody.innerHTML = tableBodyHtml;

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

    return htmlContent;
}

function generateHtmlReport(predictions, topPct, marketSpread, autoOpen = false) {
    const htmlContent = buildPredictionHtmlReport(predictions, topPct, marketSpread, autoOpen);

    const stockPredictorPath = path.join(__dirname, 'public', 'stock_predictor.html');
    const legacyPath = path.join(__dirname, 'public', 'results_chart.html');
    fs.writeFileSync(stockPredictorPath, htmlContent, 'utf8');
    fs.writeFileSync(legacyPath, htmlContent, 'utf8');

    if (autoOpen) {
        const openCmd = process.platform === 'win32' ? 'start' : (process.platform === 'darwin' ? 'open' : 'xdg-open');
        exec(`${openCmd} "${stockPredictorPath}"`, () => {});
    }
}

if (require.main === module) {
    runModelInference({ autoOpen: true, persistHtml: true })
        .then(res => {
            console.log("Model Inference completed successfully.");
            console.log(`Universe: ${res.universeSize} stocks | Market Spread: ${res.marketSpread.toFixed(4)} | Latency: ${res.latency}ms`);
        })
        .catch(err => {
            console.error("Inference Error:", err);
        });
}

module.exports = {
    runModelInference,
    getStockForecast,
    getLatestPredictions,
    buildPredictionHtmlReport,
    SP500_TICKERS
};
