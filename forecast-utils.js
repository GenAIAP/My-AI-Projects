/**
 * forecast-utils.js (FIXED)
 * Pulls real dollar close prices from getOHLCV cache and draws smooth, visible trajectories.
 */

const { getOHLCV } = require('./market-data');

const STOCK_RANGE_DAYS = 150; // Matches market-data.js cache range

async function getStockForecast(ticker, latestResult) {
  if (!latestResult || !latestResult.predictions) return null;

  const cleanTicker = ticker.trim().toUpperCase();
  const pred = latestResult.predictions.find(
    (p) => p.ticker.toUpperCase() === cleanTicker,
  );

  if (!pred) return null;

  // 1. Fetch real dollar OHLCV data from the cache
  const ohlcv = await getOHLCV(cleanTicker, STOCK_RANGE_DAYS);
  if (!ohlcv || !ohlcv.closes || ohlcv.closes.length < 60) {
    return null;
  }

  // Slice the last 60 trading days of real dollar prices (e.g. $540.00 -> $660.25)
  const historicalPrices = ohlcv.closes
    .slice(-60)
    .map((p) => Number(p.toFixed(2)));
  const lastClose = historicalPrices[historicalPrices.length - 1]; // Real dollar price (e.g. $660.25)

  const expRet = Number(pred.expectedReturn5d || 0); // e.g. +2.98% or +2.28%
  const unc = Number(pred.uncertainty5d || 2.0);

  // 2. Build real dollar trajectory:
  // For Meta at $660.25 and +2.28%:
  // Day 1: $664.01
  // Day 2: $667.03
  // Day 3: $670.51
  // Day 4: $673.05
  // Day 5: $675.30  <-- A smooth, visible, upward-sloping curve!
  const p1 = Number((lastClose * (1 + (expRet * 0.25) / 100)).toFixed(2));
  const p2 = Number((lastClose * (1 + (expRet * 0.45) / 100)).toFixed(2));
  const p3 = Number((lastClose * (1 + (expRet * 0.68) / 100)).toFixed(2));
  const p4 = Number((lastClose * (1 + (expRet * 0.85) / 100)).toFixed(2));
  const p5 = Number((lastClose * (1 + expRet / 100)).toFixed(2));

  // Alignment for Chart.js
  const historicalSeries = [
    ...historicalPrices,
    null,
    null,
    null,
    null,
    null,
  ];

  const projection = [
    ...new Array(historicalPrices.length - 1).fill(null),
    lastClose, // Connects the green line to the dotted projection
    p1,
    p2,
    p3,
    p4,
    p5,
  ];

  // Uncertainty bands (+/- sigma in dollars)
  const upperBand = projection.map((val) =>
    val !== null ? Number((val * (1 + unc / 100)).toFixed(2)) : null,
  );
  const lowerBand = projection.map((val) =>
    val !== null ? Number((val * (1 - unc / 100)).toFixed(2)) : null,
  );

  // Labels: Clean chronological order
  const labels = [];
  for (let i = historicalPrices.length - 1; i >= 0; i--) {
    labels.push(i === 0 ? 't-0 (Now)' : `t-${i}`);
  }
  labels.push('t+1', 't+2', 't+3', 't+4', 't+5');

  return {
    ticker: pred.ticker,
    rank: pred.rank,
    group: pred.group,
    snr: pred.snr,
    expectedReturn5d: expRet,
    uncertainty5d: unc,
    direction: pred.direction,
    historical: historicalSeries,
    projection: projection,
    upperBand: upperBand,
    lowerBand: lowerBand,
    labels: labels,
  };
}

module.exports = { getStockForecast };
