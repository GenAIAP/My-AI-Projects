// market-data.js
// Fetches real OHLCV history from Yahoo Finance and builds the exact
// feature set the model was trained on (see the Python training script):
//
//   Per-stock features (7):
//     0: ret          - daily % return
//     1: intraday     - (close-open)/open * 100
//     2: hl_vol       - (high-low)/close * 100
//     3: vol_shock    - clip(volume/vol_sma20 - 1, -3, 3)
//     4: mkt_ret      - S&P 500 daily % return (same across all stocks that day)
//     5: excess_ret   - ret - mkt_ret
//     6: dist_sma20   - clip((close-sma20)/sma20 * 100, -25, 25)
//
//   Macro vector (4), shared by every stock for the "current" day:
//     0: mkt_ret            - S&P 500 daily % return
//     1: mkt_5d_ret         - S&P 500 5-day % return
//     2: mkt_vol20          - 20-day rolling std of mkt_ret, annualized (*sqrt(252))
//     3: mkt_sma50_dist     - clip((close-sma50)/sma50 * 100, -25, 25)

const INDEX_TICKER = '^GSPC';
const STOCK_RANGE_DAYS = 150;   // calendar days of history to request per stock (~100+ trading days)
const INDEX_RANGE_DAYS = 250;   // index needs more lookback for the 50-day SMA
const CACHE_TTL_MS = 20 * 60 * 1000; // 20 minutes
const FETCH_BATCH_SIZE = 20;    // concurrent requests per batch
const BATCH_DELAY_MS = 150;     // small delay between batches to be polite to Yahoo

const ohlcvCache = new Map(); // ticker -> { data, fetchedAt }

function clip(value, min, max) {
    return Math.max(min, Math.min(max, value));
}

function rollingMean(arr, window, minPeriods = 5) {
    const out = new Array(arr.length).fill(NaN);
    let sum = 0;
    const q = [];
    for (let i = 0; i < arr.length; i++) {
        q.push(arr[i]);
        sum += arr[i];
        if (q.length > window) sum -= q.shift();
        if (q.length >= Math.min(minPeriods, window)) {
            out[i] = sum / q.length;
        }
    }
    return out;
}

function rollingStd(arr, window, minPeriods = 5) {
    const out = new Array(arr.length).fill(NaN);
    for (let i = 0; i < arr.length; i++) {
        const start = Math.max(0, i - window + 1);
        const slice = arr.slice(start, i + 1);
        if (slice.length >= Math.min(minPeriods, window)) {
            const mean = slice.reduce((a, b) => a + b, 0) / slice.length;
            const variance = slice.reduce((a, b) => a + (b - mean) ** 2, 0) / slice.length;
            out[i] = Math.sqrt(variance);
        }
    }
    return out;
}

async function fetchRawOHLCV(ticker, rangeDays) {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=1d&range=${rangeDays}d&includeAdjustedClose=true`;

    const response = await fetch(url, {
        headers: { 'User-Agent': 'Mozilla/5.0', 'Accept': 'application/json' }
    });

    if (!response.ok) {
        throw new Error(`Yahoo Finance request failed for ${ticker}: ${response.status}`);
    }

    const payload = await response.json();
    const result = payload?.chart?.result?.[0];
    if (!result) return null;

    const timestamps = result.timestamp || [];
    const quote = result.indicators?.quote?.[0] || {};
    const { open = [], high = [], low = [], close = [], volume = [] } = quote;

    const dates = [];
    const opens = [];
    const highs = [];
    const lows = [];
    const closes = [];
    const volumes = [];

    for (let i = 0; i < timestamps.length; i++) {
        const c = close[i];
        const o = open[i];
        const h = high[i];
        const l = low[i];
        const v = volume[i];
        // Skip incomplete/holiday rows
        if ([c, o, h, l].every(Number.isFinite)) {
            dates.push(timestamps[i]);
            opens.push(o);
            highs.push(h);
            lows.push(l);
            closes.push(c);
            volumes.push(Number.isFinite(v) ? v : 0);
        }
    }

    return { dates, opens, highs, lows, closes, volumes };
}

async function getOHLCV(ticker, rangeDays) {
    const cached = ohlcvCache.get(ticker);
    if (cached && (Date.now() - cached.fetchedAt) < CACHE_TTL_MS) {
        return cached.data;
    }
    const data = await fetchRawOHLCV(ticker, rangeDays);
    if (data) {
        ohlcvCache.set(ticker, { data, fetchedAt: Date.now() });
    }
    return data;
}

/** Fetch OHLCV for many tickers with limited concurrency. Returns Map<ticker, series|null>. */
async function fetchManyOHLCV(tickers, rangeDays) {
    const out = new Map();
    for (let i = 0; i < tickers.length; i += FETCH_BATCH_SIZE) {
        const batch = tickers.slice(i, i + FETCH_BATCH_SIZE);
        const results = await Promise.all(batch.map(async (t) => {
            try {
                const data = await getOHLCV(t, rangeDays);
                return [t, data];
            } catch (err) {
                console.warn(`OHLCV fetch failed for ${t}:`, err.message || err);
                return [t, null];
            }
        }));
        for (const [t, data] of results) out.set(t, data);
        if (i + FETCH_BATCH_SIZE < tickers.length) {
            await new Promise(resolve => setTimeout(resolve, BATCH_DELAY_MS));
        }
    }
    return out;
}

/** Compute the per-day market return series (%) from the index OHLCV. */
function computeMarketReturns(indexSeries) {
    const { closes } = indexSeries;
    const ret = new Array(closes.length).fill(0);
    for (let i = 1; i < closes.length; i++) {
        ret[i] = ((closes[i] - closes[i - 1]) / (closes[i - 1] + 1e-8)) * 100;
    }
    return ret;
}

/** Build the 4-value macro vector for the most recent available day. */
function buildMacroVector(indexSeries, mktRet) {
    const { closes } = indexSeries;
    const n = closes.length;
    const last = n - 1;

    const mkt5dRet = n > 5
        ? ((closes[last] - closes[last - 5]) / (closes[last - 5] + 1e-8)) * 100
        : 0;

    const vol20Series = rollingStd(mktRet, 20, 5);
    const mktVol20 = Number.isFinite(vol20Series[last])
        ? vol20Series[last] * Math.sqrt(252)
        : 16.0; // fallback matches the Python default fillna(16.0)

    const sma50Series = rollingMean(closes, 50, 10);
    const sma50 = sma50Series[last];
    const mktSma50Dist = Number.isFinite(sma50)
        ? clip(((closes[last] - sma50) / (sma50 + 1e-8)) * 100, -25, 25)
        : 0;

    return new Float32Array([mktRet[last], mkt5dRet, mktVol20, mktSma50Dist]);
}

/**
 * Build the [seqLen, 7] feature block for one stock, aligned against the
 * index's return series by trailing position (both series are fetched over
 * the same daily interval, so the last N trading days line up in practice).
 * Returns null if there isn't enough history to fill seqLen rows.
 */
function buildTickerFeatureBlock(series, mktRet, seqLen) {
    if (!series) return null;
    const { opens, highs, lows, closes, volumes } = series;
    const n = closes.length;
    if (n < seqLen + 5) return null; // not enough history

    const ret = new Array(n).fill(0);
    for (let i = 1; i < n; i++) {
        ret[i] = ((closes[i] - closes[i - 1]) / (closes[i - 1] + 1e-8)) * 100;
    }

    const intraday = closes.map((c, i) => ((c - opens[i]) / (opens[i] + 1e-8)) * 100);
    const hlVol = closes.map((c, i) => ((highs[i] - lows[i]) / (c + 1e-8)) * 100);

    const volSma20 = rollingMean(volumes, 20, 5);
    const volShock = volumes.map((v, i) => {
        const base = volSma20[i];
        if (!Number.isFinite(base) || base === 0) return 0;
        return clip((v / base) - 1, -3, 3);
    });

    const sma20 = rollingMean(closes, 20, 5);
    const distSma20 = closes.map((c, i) => {
        const s = sma20[i];
        if (!Number.isFinite(s)) return 0;
        return clip(((c - s) / (s + 1e-8)) * 100, -25, 25);
    });

    // Align market return by trailing position: use the last `n` entries of mktRet
    const mktAligned = mktRet.slice(-n);
    // pad the front if the index has fewer points than the stock (shouldn't normally happen)
    while (mktAligned.length < n) mktAligned.unshift(0);

    const excessRet = ret.map((r, i) => r - mktAligned[i]);

    // Take the last `seqLen` rows
    const startIdx = n - seqLen;
    const block = new Float32Array(seqLen * 7);
    for (let i = 0; i < seqLen; i++) {
        const srcIdx = startIdx + i;
        const off = i * 7;
        block[off + 0] = ret[srcIdx];
        block[off + 1] = intraday[srcIdx];
        block[off + 2] = hlVol[srcIdx];
        block[off + 3] = volShock[srcIdx];
        block[off + 4] = mktAligned[srcIdx];
        block[off + 5] = excessRet[srcIdx];
        block[off + 6] = distSma20[srcIdx];
    }
    return block;
}

/**
 * Fetch and prepare everything needed for a full inference run:
 * - index series + market return series + macro vector
 * - per-ticker feature blocks (null for tickers with unusable data)
 */
async function prepareInferenceInputs(tickers, seqLen) {
    const indexSeries = await getOHLCV(INDEX_TICKER, INDEX_RANGE_DAYS);
    if (!indexSeries || indexSeries.closes.length < seqLen + 5) {
        throw new Error('Unable to fetch sufficient S&P 500 index history');
    }

    const mktRet = computeMarketReturns(indexSeries);
    const macroVector = buildMacroVector(indexSeries, mktRet);

    const seriesByTicker = await fetchManyOHLCV(tickers, STOCK_RANGE_DAYS);

    const blocksByTicker = new Map();
    for (const ticker of tickers) {
        const series = seriesByTicker.get(ticker);
        const block = buildTickerFeatureBlock(series, mktRet, seqLen);
        blocksByTicker.set(ticker, block); // may be null
    }

    return { macroVector, blocksByTicker };
}

module.exports = {
    prepareInferenceInputs,
    getOHLCV,
    INDEX_TICKER
};
