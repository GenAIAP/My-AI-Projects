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
    if (!result) return [];

    const closes = result.indicators?.quote?.[0]?.close || [];
    const timestamps = result.timestamp || [];
    const series = [];

    for (let i = 0; i < timestamps.length; i++) {
        const close = closes[i];
        if (Number.isFinite(close)) {
            series.push({ timestamp: timestamps[i], close: Number(close) });
        }
    }

    return series.slice(-maxPoints);
}

/**
 * predictionsResult: the object returned by runModelInference / getLatestPredictions()
 * (i.e. { predictions, universeSize, marketSpread, topK, latency, timestamp })
 */
async function getStockForecast(ticker, predictionsResult) {
    if (!predictionsResult || !predictionsResult.predictions) {
        return null;
    }

    const cleanTicker = (ticker || "").toUpperCase().trim();
    const item = predictionsResult.predictions.find(p => p.ticker.toUpperCase() === cleanTicker)
        || predictionsResult.predictions[0];
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

    const nowPrice = history[history.length - 1] || currentPrice;

    const expectedPath = new Array(histLabels.length - 1).fill(null);
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

module.exports = { getStockForecast, fetchYahooFinanceSeries };