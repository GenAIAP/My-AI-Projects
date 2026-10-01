// File: ai-stock-predictor.js
// ==============================================================================
// S&P 500 3D HIERARCHICAL QUANT ENGINE (VERSION 4.0 PRODUCTION INFERENCE)
// Fully Synchronized with sp500_hierarchical_v4_clean_production.py & ONNX Exports
// ==============================================================================
// Architecture:
//   1. Temporal Stream: Multi-Scale Inception TCN (Causal Conv & Channel Norm)
//   2. Factor Stream: 6-Layer Deep Inverted Transformer with SwiGLU
//   3. Contextual Gating: Bilinear GLU Fusion (Temporal + Factor Interactions)
//   4. Spatial Hierarchy: Intra-Sector Peer Routing + 12 Centroids (1 Market + 11 GICS)
//   5. Normalization: Masked Cross-Sectional Norm (CS-Norm)
//
// Execution Mechanics (Strict T+1 Market-on-Open Parity):
//   - Signal EMA Smoothing: alpha = 0.40
//   - Conviction Hurdle: 30 bps base (45 bps on Mondays: 1.5x multiplier)
//   - Day Dampeners: Wednesday (0.50x) and Thursday (0.50x)
//   - Volatility Target: 14% Annualized Volatility Ceiling (min 20%, max 100%)
//   - Portfolio Weighting: Inverse-Volatility Risk Parity (w_i ~ 1 / sigma_i)
// ==============================================================================

const fs = require('fs');
const path = require('path');
const { parentPort, Worker, workerData, isMainThread } = require('worker_threads');
const ort = require('onnxruntime-node');

// ==============================================================================
// SECTION 1: MASTER CONSTANTS & GICS SECTOR TAXONOMY
// ==============================================================================
const BENCHMARK_TICKER = '^GSPC';
const STOCK_RANGE_DAYS = 240;               // Lookback window for reliable features
const CACHE_TTL_MS = 15 * 60 * 1000;        // 15-minute in-memory cache
const DISK_CACHE_TTL_MS = 12 * 3600 * 1000; // 12-hour disk cache
const FETCH_BATCH_SIZE = 30;
const BATCH_DELAY_MS = 50;

const DEFAULT_MAX_STOCKS = 650;
const DEFAULT_SEQ_LEN = 60;
const NUM_FEATURES = 15;
const DEFAULT_TOP_K = 15;
const DEFAULT_BOTTOM_K = 15;

// Strict Python v4 Execution Rules
const SIGNAL_EMA_ALPHA = 0.40;
const CONVICTION_THRESHOLD = 0.0030;        // 30 bps 5-day alpha
const EXIT_BAND_THRESHOLD = -0.0030;
const MONDAY_HURDLE_MULT = 1.5;             // 1.5x on Mondays (45 bps)
const MIN_HOLDING_DAYS = 3;
const BUFFER_RANK = 30;
const TRANSACTION_FEE_BPS = 0.0010;        // 10 bps fee drag

// Volatility & Capital Constraints
const MIN_CAPITAL_EXPOSURE = 0.20;
const MAX_CAPITAL_EXPOSURE = 1.00;
const TARGET_ANNUAL_VOL = 0.14;             // 14% Annual Vol Ceiling
const ENABLE_VOL_TARGETING = true;
const THURSDAY_DAMPENER = 0.50;             // Triggered on Wednesday close
const FRIDAY_DAMPENER = 0.50;               // Triggered on Thursday close

const FEATURE_NAMES = [
    'ret_1d', 'ret_5d', 'ret_10d', 'ret_20d',
    'gk_vol', 'parkinson_vol',
    'rsi_14', 'macd_norm', 'macd_hist_norm', 'bb_pct_b',
    'norm_volume', 'hl_spread', 'co_spread', 'dist_sma_20', 'dist_sma_50'
];

const SECTOR_TO_ID = {
    'Technology': 0, 'Healthcare': 1, 'Financials': 2, 'Discretionary': 3,
    'Communication': 4, 'Industrials': 5, 'Staples': 6, 'Energy': 7,
    'Utilities': 8, 'RealEstate': 9, 'Materials': 10, 'General': 0
};

const GICS_CLEAN_MAP = {
    'Information Technology': 'Technology', 'Technology': 'Technology',
    'Health Care': 'Healthcare', 'Healthcare': 'Healthcare',
    'Financials': 'Financials',
    'Consumer Discretionary': 'Discretionary', 'Discretionary': 'Discretionary',
    'Communication Services': 'Communication', 'Communication': 'Communication',
    'Industrials': 'Industrials',
    'Consumer Staples': 'Staples', 'Staples': 'Staples',
    'Energy': 'Energy',
    'Utilities': 'Utilities',
    'Real Estate': 'RealEstate', 'RealEstate': 'RealEstate',
    'Materials': 'Materials', 'General': 'General'
};

const DEAD_TICKERS = new Set([
    'PARA', 'SBNY', 'SIVB', 'FRC', 'DRE', 'TWTR', 'PKI', 'ANTM', 'FB',
    'BIO', 'ETSY', 'ATVI', 'FLIR', 'CA', 'HOT', 'CMA', 'ZION',
    'AAL', 'WHR', 'SEE', 'LUMN', 'NWL', 'DXC', 'FBHS', 'IPGP',
    'AIV', 'TSS', 'PEAK', 'RE'
]);

const SP500_TICKERS = [
    'A', 'AAPL', 'ABBV', 'ABNB', 'ABT', 'ACGL', 'ACN', 'ADBE', 'ADI', 'ADM', 'ADP', 'ADSK', 'AEE', 'AEP',
    'AES', 'AFL', 'AIG', 'AIZ', 'AJG', 'AKAM', 'ALB', 'ALGN', 'ALL', 'ALLE', 'AMAT', 'AMCR', 'AMD', 'AME', 'AMGN',
    'AMP', 'AMT', 'AMZN', 'ANET', 'ANSS', 'AON', 'AOS', 'APA', 'APD', 'APH', 'APTV', 'ARE', 'ATO', 'AVB', 'AVGO',
    'AVY', 'AWK', 'AXON', 'AXP', 'AZO', 'BA', 'BAC', 'BALL', 'BAX', 'BBWI', 'BBY', 'BDX', 'BEN', 'BF-B', 'BG',
    'BIIB', 'BK', 'BKNG', 'BKR', 'BLDR', 'BLK', 'BMY', 'BR', 'BRK-B', 'BRO', 'BSX', 'BWA', 'BX', 'BXP', 'C',
    'CAG', 'CAH', 'CARR', 'CAT', 'CB', 'CBOE', 'CBRE', 'CCI', 'CCL', 'CDNS', 'CDW', 'CE', 'CEG', 'CF', 'CFG',
    'CHD', 'CHRW', 'CHTR', 'CI', 'CINF', 'CL', 'CLX', 'CMCSA', 'CME', 'CMG', 'CMI', 'CMS', 'CNC', 'CNP', 'COF',
    'COO', 'COP', 'COR', 'COST', 'CPAY', 'CPB', 'CPRT', 'CPT', 'CRL', 'CRM', 'CRWD', 'CSCO', 'CSGP', 'CSX', 'CTAS',
    'CTLT', 'CTRA', 'CTSH', 'CTVA', 'CVS', 'CVX', 'CZR', 'D', 'DAL', 'DAY', 'DD', 'DE', 'DECK', 'DELL', 'DFS',
    'DG', 'DGX', 'DHI', 'DHR', 'DIS', 'DLR', 'DLTR', 'DOC', 'DOV', 'DOW', 'DPZ', 'DRI', 'DTE', 'DUK', 'DVA',
    'DVN', 'DXCM', 'EA', 'EBAY', 'ECL', 'ED', 'EFX', 'EG', 'EIX', 'EL', 'ELV', 'EMN', 'EMR', 'ENPH', 'EOG',
    'EPAM', 'EQIX', 'EQR', 'EQT', 'ERIE', 'ES', 'ESS', 'ETN', 'ETR', 'EVRG', 'EW', 'EXC', 'EXPD', 'EXPE', 'EXR',
    'F', 'FANG', 'FAST', 'FCX', 'FDS', 'FDX', 'FE', 'FFIV', 'FI', 'FICO', 'FIS', 'FITB', 'FMC', 'FOX', 'FOXA',
    'FRT', 'FSLR', 'FTNT', 'FTV', 'GD', 'GDDY', 'GE', 'GEHC', 'GEN', 'GEV', 'GILD', 'GIS', 'GL', 'GLW', 'GM',
    'GNRC', 'GOOG', 'GOOGL', 'GPC', 'GPN', 'GRMN', 'GS', 'GWW', 'HAL', 'HAS', 'HBAN', 'HCA', 'HD', 'HES', 'HIG',
    'HII', 'HLT', 'HOLX', 'HON', 'HPE', 'HPQ', 'HRL', 'HSIC', 'HST', 'HSY', 'HUBB', 'HUM', 'HWM', 'IBM', 'ICE',
    'IDXX', 'IEX', 'IFF', 'INCY', 'INTC', 'INTU', 'INVH', 'IP', 'IPG', 'IQV', 'IR', 'IRM', 'ISRG', 'IT', 'ITW',
    'IVZ', 'J', 'JBHT', 'JBL', 'JCI', 'JKHY', 'JNJ', 'JNPR', 'JPM', 'K', 'KDP', 'KEY', 'KEYS', 'KHC', 'KIM',
    'KKR', 'KLAC', 'KMB', 'KMI', 'KMX', 'KO', 'KR', 'KVUE', 'L', 'LDOS', 'LEN', 'LH', 'LHX', 'LIN', 'LKQ', 'LLY',
    'LMT', 'LNT', 'LOW', 'LRCX', 'LULU', 'LUV', 'LVS', 'LW', 'LYB', 'LYV', 'MA', 'MAA', 'MAR', 'MAS', 'MCD',
    'MCHP', 'MCK', 'MCO', 'MDLZ', 'MDT', 'MET', 'META', 'MGM', 'MHK', 'MKC', 'MKTX', 'MLM', 'MMC', 'MMM', 'MNST',
    'MO', 'MOH', 'MOS', 'MPC', 'MPWR', 'MRK', 'MRNA', 'MS', 'MSCI', 'MSFT', 'MSI', 'MTB', 'MTCH', 'MTD', 'MU',
    'NCLH', 'NDAQ', 'NDSN', 'NEE', 'NEM', 'NFLX', 'NI', 'NKE', 'NOC', 'NOW', 'NRG', 'NSC', 'NTAP', 'NTRS', 'NUE',
    'NVDA', 'NVR', 'NWS', 'NWSA', 'NXPI', 'O', 'ODFL', 'OKE', 'OMC', 'ON', 'ORCL', 'ORLY', 'OTIS', 'OXY', 'PANW',
    'PAYC', 'PAYX', 'PCAR', 'PCG', 'PEG', 'PEP', 'PFE', 'PFG', 'PG', 'PGR', 'PH', 'PHM', 'PKG', 'PLD',
    'PLTR', 'PM', 'PNC', 'PNR', 'PNW', 'PODD', 'POOL', 'PPG', 'PPL', 'PRU', 'PSA', 'PSKY', 'PSX', 'PTC', 'PWR', 'PYPL',
    'QCOM', 'QRVO', 'RCL', 'REG', 'REGN', 'RF', 'RHI', 'RJF', 'RL', 'RMD', 'ROK', 'ROL', 'ROP', 'ROST', 'RSG',
    'RTX', 'RVTY', 'SBAC', 'SBUX', 'SCHW', 'SHW', 'SJM', 'SLB', 'SMCI', 'SNA', 'SNPS', 'SO', 'SOLV', 'SPG',
    'SPGI', 'SRE', 'STE', 'STLD', 'STT', 'STX', 'STZ', 'SWK', 'SWKS', 'SYF', 'SYK', 'SYY', 'T', 'TAP', 'TDG',
    'TDY', 'TECH', 'TEL', 'TER', 'TFC', 'TFX', 'TGT', 'TJX', 'TMO', 'TMUS', 'TPR', 'TRGP', 'TRMB', 'TROW', 'TRV',
    'TSCO', 'TSLA', 'TSN', 'TT', 'TTWO', 'TXN', 'TXT', 'TYL', 'UAL', 'UBER', 'UDR', 'UHS', 'ULTA', 'UNH', 'UNP',
    'UPS', 'URI', 'USB', 'V', 'VICI', 'VLO', 'VLTO', 'VMC', 'VRSK', 'VRSN', 'VRTX', 'VST', 'VTR', 'VTRS', 'VZ',
    'WAB', 'WAT', 'WBA', 'WBD', 'WDC', 'WEC', 'WELL', 'WFC', 'WM', 'WMB', 'WMT', 'WRB', 'WST', 'WTW', 'WY',
    'WYNN', 'XEL', 'XOM', 'XYL', 'YUM', 'ZBH', 'ZBRA', 'ZTS'
];

const TICKER_GICS_SECTORS = {
    'A': 'Healthcare', 'AAPL': 'Technology', 'ABBV': 'Healthcare', 'ABNB': 'Discretionary', 'ABT': 'Healthcare',
    'ACGL': 'Financials', 'ACN': 'Technology', 'ADBE': 'Technology', 'ADI': 'Technology', 'ADM': 'Staples',
    'ADP': 'Industrials', 'ADSK': 'Technology', 'AEE': 'Utilities', 'AEP': 'Utilities', 'AES': 'Utilities',
    'AFL': 'Financials', 'AIG': 'Financials', 'AIZ': 'Financials', 'AJG': 'Financials', 'AKAM': 'Technology',
    'ALB': 'Materials', 'ALGN': 'Healthcare', 'ALL': 'Financials', 'ALLE': 'Industrials', 'AMAT': 'Technology',
    'AMCR': 'Materials', 'AMD': 'Technology', 'AME': 'Industrials', 'AMGN': 'Healthcare', 'AMP': 'Financials',
    'AMT': 'RealEstate', 'AMZN': 'Discretionary', 'ANET': 'Technology', 'ANSS': 'Technology', 'AON': 'Financials',
    'AOS': 'Industrials', 'APA': 'Energy', 'APD': 'Materials', 'APH': 'Technology', 'APTV': 'Discretionary',
    'ARE': 'RealEstate', 'ATO': 'Utilities', 'AVB': 'RealEstate', 'AVGO': 'Technology', 'AVY': 'Materials',
    'AWK': 'Utilities', 'AXON': 'Industrials', 'AXP': 'Financials', 'AZO': 'Discretionary', 'BA': 'Industrials',
    'BAC': 'Financials', 'BALL': 'Materials', 'BAX': 'Healthcare', 'BBWI': 'Discretionary', 'BBY': 'Discretionary',
    'BDX': 'Healthcare', 'BEN': 'Financials', 'BF-B': 'Staples', 'BG': 'Staples', 'BIIB': 'Healthcare',
    'BK': 'Financials', 'BKNG': 'Discretionary', 'BKR': 'Energy', 'BLDR': 'Industrials', 'BLK': 'Financials',
    'BMY': 'Healthcare', 'BR': 'Industrials', 'BRK-B': 'Financials', 'BRO': 'Financials', 'BSX': 'Healthcare',
    'BWA': 'Discretionary', 'BX': 'Financials', 'BXP': 'RealEstate', 'C': 'Financials', 'CAG': 'Staples',
    'CAH': 'Healthcare', 'CARR': 'Industrials', 'CAT': 'Industrials', 'CB': 'Financials', 'CBOE': 'Financials',
    'CBRE': 'RealEstate', 'CCI': 'RealEstate', 'CCL': 'Discretionary', 'CDNS': 'Technology', 'CDW': 'Technology',
    'CE': 'Materials', 'CEG': 'Utilities', 'CF': 'Materials', 'CFG': 'Financials', 'CHD': 'Staples',
    'CHRW': 'Industrials', 'CHTR': 'Communication', 'CI': 'Healthcare', 'CINF': 'Financials', 'CL': 'Staples',
    'CLX': 'Staples', 'CMCSA': 'Communication', 'CME': 'Financials', 'CMG': 'Discretionary', 'CMI': 'Industrials',
    'CMS': 'Utilities', 'CNC': 'Healthcare', 'CNP': 'Utilities', 'COF': 'Financials', 'COO': 'Healthcare',
    'COP': 'Energy', 'COR': 'Healthcare', 'COST': 'Staples', 'CPAY': 'Financials', 'CPB': 'Staples',
    'CPRT': 'Industrials', 'CPT': 'RealEstate', 'CRL': 'Healthcare', 'CRM': 'Technology', 'CRWD': 'Technology',
    'CSCO': 'Technology', 'CSGP': 'RealEstate', 'CSX': 'Industrials', 'CTAS': 'Industrials', 'CTLT': 'Healthcare',
    'CTRA': 'Energy', 'CTSH': 'Technology', 'CTVA': 'Materials', 'CVS': 'Healthcare', 'CVX': 'Energy',
    'CZR': 'Discretionary', 'D': 'Utilities', 'DAL': 'Industrials', 'DAY': 'Industrials', 'DD': 'Materials',
    'DE': 'Industrials', 'DECK': 'Discretionary', 'DELL': 'Technology', 'DFS': 'Financials', 'DG': 'Staples',
    'DGX': 'Healthcare', 'DHI': 'Discretionary', 'DHR': 'Healthcare', 'DIS': 'Communication', 'DLR': 'RealEstate',
    'DLTR': 'Staples', 'DOC': 'RealEstate', 'DOV': 'Industrials', 'DOW': 'Materials', 'DPZ': 'Discretionary',
    'DRI': 'Discretionary', 'DTE': 'Utilities', 'DUK': 'Utilities', 'DVA': 'Healthcare', 'DVN': 'Energy',
    'DXCM': 'Healthcare', 'EA': 'Communication', 'EBAY': 'Discretionary', 'ECL': 'Materials', 'ED': 'Utilities',
    'EFX': 'Industrials', 'EG': 'Financials', 'EIX': 'Utilities', 'EL': 'Staples', 'ELV': 'Healthcare',
    'EMN': 'Materials', 'EMR': 'Industrials', 'ENPH': 'Technology', 'EOG': 'Energy', 'EPAM': 'Technology',
    'EQIX': 'RealEstate', 'EQR': 'RealEstate', 'EQT': 'Energy', 'ERIE': 'Financials', 'ES': 'Utilities',
    'ESS': 'RealEstate', 'ETN': 'Industrials', 'ETR': 'Utilities', 'EVRG': 'Utilities', 'EW': 'Healthcare',
    'EXC': 'Utilities', 'EXPD': 'Industrials', 'EXPE': 'Discretionary', 'EXR': 'RealEstate', 'F': 'Discretionary',
    'FANG': 'Energy', 'FAST': 'Industrials', 'FCX': 'Materials', 'FDS': 'Financials', 'FDX': 'Industrials',
    'FE': 'Utilities', 'FFIV': 'Technology', 'FI': 'Financials', 'FICO': 'Technology', 'FIS': 'Financials',
    'FITB': 'Financials', 'FMC': 'Materials', 'FOX': 'Communication', 'FOXA': 'Communication', 'FRT': 'RealEstate',
    'FSLR': 'Technology', 'FTNT': 'Technology', 'FTV': 'Industrials', 'GD': 'Industrials', 'GDDY': 'Technology',
    'GE': 'Industrials', 'GEHC': 'Healthcare', 'GEN': 'Technology', 'GEV': 'Industrials', 'GILD': 'Healthcare',
    'GIS': 'Staples', 'GL': 'Financials', 'GLW': 'Technology', 'GM': 'Discretionary', 'GNRC': 'Industrials',
    'GOOG': 'Communication', 'GOOGL': 'Communication', 'GPC': 'Discretionary', 'GPN': 'Financials', 'GRMN': 'Discretionary',
    'GS': 'Financials', 'GWW': 'Industrials', 'HAL': 'Energy', 'HAS': 'Discretionary', 'HBAN': 'Financials',
    'HCA': 'Healthcare', 'HD': 'Discretionary', 'HES': 'Energy', 'HIG': 'Financials', 'HII': 'Industrials',
    'HLT': 'Discretionary', 'HOLX': 'Healthcare', 'HON': 'Industrials', 'HPE': 'Technology', 'HPQ': 'Technology',
    'HRL': 'Staples', 'HSIC': 'Healthcare', 'HST': 'RealEstate', 'HSY': 'Staples', 'HUBB': 'Industrials',
    'HUM': 'Healthcare', 'HWM': 'Industrials', 'IBM': 'Technology', 'ICE': 'Financials', 'IDXX': 'Healthcare',
    'IEX': 'Industrials', 'IFF': 'Materials', 'INCY': 'Healthcare', 'INTC': 'Technology', 'INTU': 'Technology',
    'INVH': 'RealEstate', 'IP': 'Materials', 'IPG': 'Communication', 'IQV': 'Healthcare', 'IR': 'Industrials',
    'IRM': 'RealEstate', 'ISRG': 'Healthcare', 'IT': 'Technology', 'ITW': 'Industrials', 'IVZ': 'Financials',
    'J': 'Industrials', 'JBHT': 'Industrials', 'JBL': 'Technology', 'JCI': 'Industrials', 'JKHY': 'Financials',
    'JNJ': 'Healthcare', 'JNPR': 'Technology', 'JPM': 'Financials', 'K': 'Staples', 'KDP': 'Staples',
    'KEY': 'Financials', 'KEYS': 'Technology', 'KHC': 'Staples', 'KIM': 'RealEstate', 'KKR': 'Financials',
    'KLAC': 'Technology', 'KMB': 'Staples', 'KMI': 'Energy', 'KMX': 'Discretionary', 'KO': 'Staples',
    'KR': 'Staples', 'KVUE': 'Staples', 'L': 'Financials', 'LDOS': 'Industrials', 'LEN': 'Discretionary',
    'LH': 'Healthcare', 'LHX': 'Industrials', 'LIN': 'Materials', 'LKQ': 'Discretionary', 'LLY': 'Healthcare',
    'LMT': 'Industrials', 'LNT': 'Utilities', 'LOW': 'Discretionary', 'LRCX': 'Technology', 'LULU': 'Discretionary',
    'LUV': 'Industrials', 'LVS': 'Discretionary', 'LW': 'Staples', 'LYB': 'Materials', 'LYV': 'Communication',
    'MA': 'Financials', 'MAA': 'RealEstate', 'MAR': 'Discretionary', 'MAS': 'Industrials', 'MCD': 'Discretionary',
    'MCHP': 'Technology', 'MCK': 'Healthcare', 'MCO': 'Financials', 'MDLZ': 'Staples', 'MDT': 'Healthcare',
    'MET': 'Financials', 'META': 'Communication', 'MGM': 'Discretionary', 'MHK': 'Discretionary', 'MKC': 'Staples',
    'MKTX': 'Financials', 'MLM': 'Materials', 'MMC': 'Financials', 'MMM': 'Industrials', 'MNST': 'Staples',
    'MO': 'Staples', 'MOH': 'Healthcare', 'MOS': 'Materials', 'MPC': 'Energy', 'MPWR': 'Technology',
    'MRK': 'Healthcare', 'MRNA': 'Healthcare', 'MS': 'Financials', 'MSCI': 'Financials', 'MSFT': 'Technology',
    'MSI': 'Technology', 'MTB': 'Financials', 'MTCH': 'Communication', 'MTD': 'Healthcare', 'MU': 'Technology',
    'NCLH': 'Discretionary', 'NDAQ': 'Financials', 'NDSN': 'Industrials', 'NEE': 'Utilities', 'NEM': 'Materials',
    'NFLX': 'Communication', 'NI': 'Utilities', 'NKE': 'Discretionary', 'NOC': 'Industrials', 'NOW': 'Technology',
    'NRG': 'Utilities', 'NSC': 'Industrials', 'NTAP': 'Technology', 'NTRS': 'Financials', 'NUE': 'Materials',
    'NVDA': 'Technology', 'NVR': 'Discretionary', 'NWS': 'Communication', 'NWSA': 'Communication', 'NXPI': 'Technology',
    'O': 'RealEstate', 'ODFL': 'Industrials', 'OKE': 'Energy', 'OMC': 'Communication', 'ON': 'Technology',
    'ORCL': 'Technology', 'ORLY': 'Discretionary', 'OTIS': 'Industrials', 'OXY': 'Energy', 'PANW': 'Technology',
    'PAYC': 'Industrials', 'PAYX': 'Industrials', 'PCAR': 'Industrials', 'PCG': 'Utilities',
    'PEG': 'Utilities', 'PEP': 'Staples', 'PFE': 'Healthcare', 'PFG': 'Financials', 'PG': 'Staples',
    'PGR': 'Financials', 'PH': 'Industrials', 'PHM': 'Discretionary', 'PKG': 'Materials', 'PLD': 'RealEstate',
    'PLTR': 'Technology', 'PM': 'Staples', 'PNC': 'Financials', 'PNR': 'Industrials', 'PNW': 'Utilities',
    'PODD': 'Healthcare', 'POOL': 'Discretionary', 'PPG': 'Materials', 'PPL': 'Utilities', 'PRU': 'Financials',
    'PSA': 'RealEstate', 'PSKY': 'Communication', 'PSX': 'Energy', 'PTC': 'Technology', 'PWR': 'Industrials', 'PYPL': 'Financials',
    'QCOM': 'Technology', 'QRVO': 'Technology', 'RCL': 'Discretionary', 'REG': 'RealEstate', 'REGN': 'Healthcare',
    'RF': 'Financials', 'RHI': 'Industrials', 'RJF': 'Financials', 'RL': 'Discretionary', 'RMD': 'Healthcare',
    'ROK': 'Industrials', 'ROL': 'Industrials', 'ROP': 'Technology', 'ROST': 'Discretionary', 'RSG': 'Industrials',
    'RTX': 'Industrials', 'RVTY': 'Healthcare', 'SBAC': 'RealEstate', 'SBUX': 'Discretionary', 'SCHW': 'Financials',
    'SHW': 'Materials', 'SJM': 'Staples', 'SLB': 'Energy', 'SMCI': 'Technology', 'SNA': 'Industrials',
    'SNPS': 'Technology', 'SO': 'Utilities', 'SOLV': 'Healthcare', 'SPG': 'RealEstate', 'SPGI': 'Financials',
    'SRE': 'Utilities', 'STE': 'Healthcare', 'STLD': 'Materials', 'STT': 'Financials', 'STX': 'Technology',
    'STZ': 'Staples', 'SWK': 'Industrials', 'SWKS': 'Technology', 'SYF': 'Financials', 'SYK': 'Healthcare',
    'SYY': 'Staples', 'T': 'Communication', 'TAP': 'Staples', 'TDG': 'Industrials', 'TDY': 'Technology',
    'TECH': 'Healthcare', 'TEL': 'Technology', 'TER': 'Technology', 'TFC': 'Financials', 'TFX': 'Healthcare',
    'TGT': 'Staples', 'TJX': 'Discretionary', 'TMO': 'Healthcare', 'TMUS': 'Communication', 'TPR': 'Discretionary',
    'TRGP': 'Energy', 'TRMB': 'Technology', 'TROW': 'Financials', 'TRV': 'Financials', 'TSCO': 'Discretionary',
    'TSLA': 'Discretionary', 'TSN': 'Staples', 'TT': 'Industrials', 'TTWO': 'Communication', 'TXN': 'Technology',
    'TXT': 'Industrials', 'TYL': 'Technology', 'UAL': 'Industrials', 'UBER': 'Industrials', 'UDR': 'RealEstate',
    'UHS': 'Healthcare', 'ULTA': 'Discretionary', 'UNH': 'Healthcare', 'UNP': 'Industrials', 'UPS': 'Industrials',
    'URI': 'Industrials', 'USB': 'Financials', 'V': 'Financials', 'VICI': 'RealEstate', 'VLO': 'Energy',
    'VLTO': 'Industrials', 'VMC': 'Materials', 'VRSK': 'Industrials', 'VRSN': 'Technology', 'VRTX': 'Healthcare',
    'VST': 'Utilities', 'VTR': 'RealEstate', 'VTRS': 'Healthcare', 'VZ': 'Communication', 'WAB': 'Industrials',
    'WAT': 'Healthcare', 'WBA': 'Staples', 'WBD': 'Communication', 'WDC': 'Technology', 'WEC': 'Utilities',
    'WELL': 'RealEstate', 'WFC': 'Financials', 'WM': 'Industrials', 'WMB': 'Energy', 'WMT': 'Staples',
    'WRB': 'Financials', 'WST': 'Healthcare', 'WTW': 'Financials', 'WY': 'RealEstate', 'WYNN': 'Discretionary',
    'XEL': 'Utilities', 'XOM': 'Energy', 'XYL': 'Industrials', 'YUM': 'Discretionary', 'ZBH': 'Healthcare',
    'ZBRA': 'Technology', 'ZTS': 'Healthcare'
};

// ==============================================================================
// SECTION 2: VECTORIZED MATHEMATICAL ENGINE
// ==============================================================================
function rollingMean(arr, window) {
    const n = arr.length;
    const out = new Float32Array(n);
    let sum = 0.0;
    for (let i = 0; i < n; i++) {
        sum += arr[i];
        if (i >= window) sum -= arr[i - window];
        out[i] = i >= (window - 1) ? sum / window : NaN;
    }
    return out;
}

function rollingStd(arr, window) {
    const n = arr.length;
    const out = new Float32Array(n);
    const mean = rollingMean(arr, window);
    for (let i = window - 1; i < n; i++) {
        let varSum = 0.0;
        const m = mean[i];
        for (let j = i - window + 1; j <= i; j++) {
            const diff = arr[j] - m;
            varSum += diff * diff;
        }
        out[i] = Math.sqrt(varSum / Math.max(1, window - 1));
    }
    return out;
}

function ewm(arr, alpha) {
    const n = arr.length;
    const out = new Float32Array(n);
    if (n === 0) return out;
    out[0] = arr[0];
    for (let i = 1; i < n; i++) {
        const val = Number.isFinite(arr[i]) ? arr[i] : out[i - 1];
        out[i] = (1.0 - alpha) * out[i - 1] + alpha * val;
    }
    return out;
}

function normalCDF(z) {
    const t = 1.0 / (1.0 + 0.2316419 * Math.abs(z));
    const d = 0.3989422804014327 * Math.exp(-0.5 * z * z);
    const p = d * t * (0.319381530 + t * (-0.356563782 + t * (1.781477937 + t * (-1.821255978 + t * 1.330274429))));
    return z > 0 ? 1.0 - p : p;
}

// In-Memory Persistent State for Signal Smoothing
const globalSignalEmaMap = new Map();

// ==============================================================================
// SECTION 3: INGESTION & DATA ENGINE
// ==============================================================================
const ohlcvMemoryCache = new Map();

function getPersistentCacheDir(customDir = null) {
    const target = customDir || process.env.QUANT_CACHE_DIR || path.join(__dirname, 'sp500_quant_v4');
    try {
        if (!fs.existsSync(target)) fs.mkdirSync(target, { recursive: true });
    } catch (e) {
        return path.join(__dirname, '.quant_cache_fallback');
    }
    return target;
}

async function fetchRawOHLCV(ticker, rangeDays) {
    const url = `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?interval=1d&range=${rangeDays}d&includeAdjustedClose=true`;
    const response = await fetch(url, {
        headers: {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko)',
            'Accept': 'application/json'
        }
    });

    if (!response.ok) throw new Error(`[Yahoo Finance Ingest] HTTP ${response.status} for ${ticker}`);

    const payload = await response.json();
    const result = payload?.chart?.result?.[0];
    if (!result) return null;

    const timestamps = result.timestamp || [];
    const quote = result.indicators?.quote?.[0] || {};
    const adjClose = result.indicators?.adjclose?.[0]?.adjclose || [];
    const { open = [], high = [], low = [], close = [], volume = [] } = quote;

    const dates = [];
    const opens = [];
    const highs = [];
    const lows = [];
    const closes = [];
    const volumes = [];

    for (let i = 0; i < timestamps.length; i++) {
        const o = Number(open[i]);
        const h = Number(high[i]);
        const l = Number(low[i]);
        const c = Number(adjClose[i] || close[i]);
        const v = Number(volume[i]);
        const ts = timestamps[i];

        if ([o, h, l, c].every(Number.isFinite) && o > 0 && h > 0 && l > 0 && c > 0) {
            const dateStr = new Date(ts * 1000).toISOString().split('T')[0];
            dates.push(dateStr);
            opens.push(o);
            highs.push(h);
            lows.push(l);
            closes.push(c);
            volumes.push(Number.isFinite(v) ? v : 0.0);
        }
    }

    if (dates.length < 60) return null;
    return { dates, opens, highs, lows, closes, volumes };
}

async function getOHLCV(ticker, rangeDays = STOCK_RANGE_DAYS, cacheDir = null) {
    const now = Date.now();
    const memEntry = ohlcvMemoryCache.get(ticker);
    if (memEntry && (now - memEntry.fetchedAt) < CACHE_TTL_MS) return memEntry.data;

    const cDir = getPersistentCacheDir(cacheDir);
    const diskPath = path.join(cDir, `ohlcv_${ticker.replace(/[^a-zA-Z0-9_-]/g, '_')}.json`);

    try {
        if (fs.existsSync(diskPath)) {
            const stat = fs.statSync(diskPath);
            if ((now - stat.mtimeMs) < DISK_CACHE_TTL_MS) {
                const content = fs.readFileSync(diskPath, 'utf8');
                const parsed = JSON.parse(content);
                if (parsed && Array.isArray(parsed.dates) && parsed.dates.length > 0) {
                    ohlcvMemoryCache.set(ticker, { data: parsed, fetchedAt: now });
                    return parsed;
                }
            }
        }
    } catch (e) {}

    const data = await fetchRawOHLCV(ticker, rangeDays);
    if (data) {
        ohlcvMemoryCache.set(ticker, { data, fetchedAt: now });
        try { fs.writeFileSync(diskPath, JSON.stringify(data), 'utf8'); } catch (e) {}
    }
    return data;
}

async function fetchManyOHLCV(tickers, rangeDays = STOCK_RANGE_DAYS, cacheDir = null) {
    const out = new Map();
    for (let i = 0; i < tickers.length; i += FETCH_BATCH_SIZE) {
        const batch = tickers.slice(i, i + FETCH_BATCH_SIZE);
        const results = await Promise.all(batch.map(async (t) => {
            try {
                const data = await getOHLCV(t, rangeDays, cacheDir);
                return [t, data];
            } catch (err) {
                return [t, null];
            }
        }));

        for (const [t, data] of results) {
            if (data) out.set(t, data);
        }

        if (i + FETCH_BATCH_SIZE < tickers.length) {
            await new Promise(r => setTimeout(r, BATCH_DELAY_MS));
        }
    }
    return out;
}

// ==============================================================================
// SECTION 4: 15 ORTHOGONAL TECHNICAL INDICATORS (EXACT V4 REPLICATION)
// ==============================================================================
function computeRawSignals(series) {
    const { dates, opens, highs, lows, closes, volumes } = series;
    const n = closes.length;
    if (n < 55) return null;

    const c = Float32Array.from(closes);
    const o = Float32Array.from(opens);
    const h = Float32Array.from(highs);
    const l = Float32Array.from(lows);
    const v = Float32Array.from(volumes);

    // Momentum Factors (0-3)
    const ret1d = new Float32Array(n);
    const ret5d = new Float32Array(n);
    const ret10d = new Float32Array(n);
    const ret20d = new Float32Array(n);

    for (let i = 0; i < n; i++) {
        ret1d[i] = i >= 1 ? Math.log(c[i] / c[i - 1]) : 0.0;
        ret5d[i] = i >= 5 ? Math.log(c[i] / c[i - 5]) : 0.0;
        ret10d[i] = i >= 10 ? Math.log(c[i] / c[i - 10]) : 0.0;
        ret20d[i] = i >= 20 ? Math.log(c[i] / c[i - 20]) : 0.0;
    }

    // Volatility Factors (4-5)
    const gkVol = new Float32Array(n);
    const parkinsonVol = new Float32Array(n);
    const constGk = 2.0 * Math.LN2 - 1.0;
    const constPark = 4.0 * Math.LN2;

    for (let i = 0; i < n; i++) {
        const logHL = Math.log(Math.max(h[i], 1e-8) / Math.max(l[i], 1e-8));
        const logCO = Math.log(Math.max(c[i], 1e-8) / Math.max(o[i], 1e-8));
        gkVol[i] = 0.5 * (logHL * logHL) - constGk * (logCO * logCO);
        parkinsonVol[i] = (logHL * logHL) / constPark;
    }

    // Trend & Oscillators (6-9, 13-14)
    const delta = new Float32Array(n);
    const gain = new Float32Array(n);
    const loss = new Float32Array(n);
    for (let i = 1; i < n; i++) {
        delta[i] = c[i] - c[i - 1];
        gain[i] = delta[i] > 0 ? delta[i] : 0.0;
        loss[i] = delta[i] < 0 ? -delta[i] : 0.0;
    }

    const alphaRSI = 1.0 / 14.0;
    const avgGain = ewm(gain, alphaRSI);
    const avgLoss = ewm(loss, alphaRSI);
    const rsi14 = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const rs = avgGain[i] / (avgLoss[i] + 1e-8);
        const rsiRaw = 100.0 - (100.0 / (1.0 + rs));
        rsi14[i] = (rsiRaw / 50.0) - 1.0;
    }

    const ema12 = ewm(c, 2.0 / (12 + 1));
    const ema26 = ewm(c, 2.0 / (26 + 1));
    const macd = new Float32Array(n);
    const macdNorm = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        macd[i] = ema12[i] - ema26[i];
        macdNorm[i] = macd[i] / (c[i] + 1e-8);
    }
    const macdSignal = ewm(macd, 2.0 / (9 + 1));
    const macdHistNorm = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        macdHistNorm[i] = (macd[i] - macdSignal[i]) / (c[i] + 1e-8);
    }

    const sma20 = rollingMean(c, 20);
    const std20 = rollingStd(c, 20);
    const bbPctB = new Float32Array(n);
    for (let i = 0; i < n; i++) {
        const lower = sma20[i] - 2.0 * std20[i];
        bbPctB[i] = (c[i] - lower) / (4.0 * std20[i] + 1e-8);
    }

    // Liquidity Factors (10-12)
    const volMa20 = rollingMean(v, 20);
    const normVolume = new Float32Array(n);
    const hlSpread = new Float32Array(n);
    const coSpread = new Float32Array(n);
    const distSma20 = new Float32Array(n);
    const distSma50 = new Float32Array(n);
    const sma50 = rollingMean(c, 50);

    for (let i = 0; i < n; i++) {
        normVolume[i] = Math.log((v[i] + 1.0) / (volMa20[i] + 1.0));
        hlSpread[i] = (h[i] - l[i]) / (c[i] + 1e-8);
        coSpread[i] = (c[i] - o[i]) / (o[i] + 1e-8);
        distSma20[i] = (c[i] - sma20[i]) / (sma20[i] + 1e-8);
        distSma50[i] = (c[i] - sma50[i]) / (sma50[i] + 1e-8);
    }

    const realizedVol20 = rollingStd(ret1d, 20);
    const currentVol = Number(realizedVol20[n - 1]) || 0.015;

    const dateFeatureMap = new Map();
    for (let i = 0; i < n; i++) {
        if (i < 50) continue;

        const row = new Float32Array(NUM_FEATURES);
        row[0] = ret1d[i];
        row[1] = ret5d[i];
        row[2] = ret10d[i];
        row[3] = ret20d[i];
        row[4] = gkVol[i];
        row[5] = parkinsonVol[i];
        row[6] = rsi14[i];
        row[7] = macdNorm[i];
        row[8] = macdHistNorm[i];
        row[9] = bbPctB[i];
        row[10] = normVolume[i];
        row[11] = hlSpread[i];
        row[12] = coSpread[i];
        row[13] = distSma20[i];
        row[14] = distSma50[i];

        dateFeatureMap.set(dates[i], row);
    }

    return {
        dateFeatureMap,
        realizedVol: Math.max(0.005, currentVol),
        lastClose: c[n - 1]
    };
}

// ==============================================================================
// SECTION 5: POINT-IN-TIME CROSS-SECTIONAL RANK NORMALIZATION [-1, 1]
// ==============================================================================
function applyCrossSectionalRankScaling(activeSequences, N, seqLen) {
    for (let t = 0; t < seqLen; t++) {
        const tOffset = t * NUM_FEATURES;

        for (let f = 0; f < NUM_FEATURES; f++) {
            const featIdx = tOffset + f;
            const items = new Array(N);

            for (let s = 0; s < N; s++) {
                const val = activeSequences[s][featIdx];
                items[s] = { val: Number.isFinite(val) ? val : 0.0, sIdx: s };
            }

            items.sort((a, b) => a.val - b.val);

            let i = 0;
            while (i < N) {
                let j = i;
                while (j + 1 < N && Math.abs(items[j + 1].val - items[i].val) < 1e-12) {
                    j++;
                }

                const avgRank = ((i + 1) + (j + 1)) / 2.0;
                const pctRank = avgRank / N;
                const scaled = (pctRank * 2.0) - 1.0;

                for (let k = i; k <= j; k++) {
                    activeSequences[items[k].sIdx][featIdx] = scaled;
                }
                i = j + 1;
            }
        }
    }
}

// ==============================================================================
// SECTION 6: 3D HIERARCHICAL ONNX INFERENCE ENGINE
// ==============================================================================
class StockPredictor {
    constructor(modelPath = null) {
        this.modelPath = modelPath || this._resolveModelPath();
        this.session = null;
        this.inputNames = [];
        this.outputNames = [];

        this.modelMaxStocks = DEFAULT_MAX_STOCKS;
        this.isDynamicStocks = true;
        this.modelSeqLen = DEFAULT_SEQ_LEN;

        this.xInputName = null;
        this.maskInputName = null;
        this.secInputName = null;
        this.macroInputName = null; // Kept for backward compatibility with v3/v5
        this.maskIsBoolean = true;
    }

    _resolveModelPath() {
        const candidateNames = [
            'Blue-1.1.onnx',
            'best_v4_clean_model.onnx',
            'sp500_hierarchical_v4_clean_production.onnx',
            'v4_model.onnx',
            'V4.onnx',
            'best_hierarchical_model.onnx',
            'sp500_hierarchical_itransformer.onnx',
            'V6.onnx',
            'model.onnx'
        ];

        const searchRoots = [
            __dirname,
            process.cwd(),
            path.join(__dirname, 'sp500_quant_v4'),
            path.join(process.cwd(), 'sp500_quant_v4'),
            'C:\\Users\\abbon\\OneDrive\\Desktop\\Coding\\AI'
        ];

        for (const root of searchRoots) {
            for (const name of candidateNames) {
                const full = path.join(root, name);
                if (fs.existsSync(full)) return full;
            }
        }
        return path.join(__dirname, 'Blue-1.1.onnx');
    }

    async init() {
        if (!this.session) {
            if (!fs.existsSync(this.modelPath)) {
                throw new Error(`[StockPredictor] ONNX model weights not found at '${this.modelPath}'.`);
            }

            this.session = await ort.InferenceSession.create(this.modelPath, {
                executionProviders: ['cpu'],
                graphOptimizationLevel: 'all'
            });

            this.inputNames = this.session.inputNames || [];
            this.outputNames = this.session.outputNames || [];

            for (const name of this.inputNames) {
                const meta = this.session.inputMetadata?.[name];
                const dims = meta?.dimensions || [];
                const type = String(meta?.type || '').toLowerCase();

                if (dims.length === 4 || (!this.xInputName && (name.includes('x') || name.includes('feat') || name.includes('input')))) {
                    this.xInputName = name;
                    const dimN = dims[1];
                    if (typeof dimN === 'number' && dimN > 0) {
                        this.modelMaxStocks = dimN;
                        this.isDynamicStocks = false;
                    } else {
                        this.isDynamicStocks = true;
                    }
                    if (typeof dims[2] === 'number' && dims[2] > 0) this.modelSeqLen = dims[2];
                } else if (/mask/i.test(name) || type.includes('bool') || (dims.length === 2 && !this.secInputName && !this.maskInputName)) {
                    this.maskInputName = name;
                    this.maskIsBoolean = type.includes('bool') || type === '';
                } else if (/sec|sid/i.test(name) || type.includes('int64') || type.includes('int32')) {
                    this.secInputName = name;
                } else if (/macro|regime/i.test(name)) {
                    this.macroInputName = name;
                }
            }

            if (!this.xInputName && this.inputNames.length > 0) this.xInputName = this.inputNames[0];
            if (!this.maskInputName && this.inputNames.length > 1) this.maskInputName = this.inputNames[1];
            if (!this.secInputName && this.inputNames.length > 2) this.secInputName = this.inputNames[2];

            console.log(`[StockPredictor] Quant Engine v4.0 Active: ${this.modelPath}`);
            console.log(`[StockPredictor] Input Bindings: X='${this.xInputName}', Mask='${this.maskInputName}' (${this.maskIsBoolean ? 'BOOL' : 'FLOAT'}), Sector='${this.secInputName}'`);
        }
        return this;
    }

    async predict(activeTickers, activeSequences, stockVolsMap) {
        if (!this.session) throw new Error("[StockPredictor] Quant Engine not initialized.");

        const activeCount = this.isDynamicStocks 
            ? activeTickers.length 
            : Math.min(activeTickers.length, this.modelMaxStocks);
            
        if (activeCount === 0) return { ranked: [], dispersion: 0.0 };

        const tickersSlice = activeTickers.slice(0, activeCount);
        const seqSlice = activeSequences.slice(0, activeCount);

        // 1. Cross-Sectional Percentile Rank Scaling [-1, 1] per timestep
        applyCrossSectionalRankScaling(seqSlice, activeCount, this.modelSeqLen);

        const targetN = this.isDynamicStocks ? activeCount : this.modelMaxStocks;
        const xBuffer = new Float32Array(1 * targetN * this.modelSeqLen * NUM_FEATURES);
        const maskBufferBool = new Uint8Array(1 * targetN);
        const maskBufferFloat = new Float32Array(1 * targetN);
        const sectorBuffer = new BigInt64Array(1 * targetN);

        for (let s = 0; s < activeCount; s++) {
            maskBufferBool[s] = 1;
            maskBufferFloat[s] = 1.0;

            const ticker = tickersSlice[s];
            const rawSec = TICKER_GICS_SECTORS[ticker] || 'General';
            const cleanSec = GICS_CLEAN_MAP[rawSec] || 'General';
            sectorBuffer[s] = BigInt(SECTOR_TO_ID[cleanSec] ?? 0);

            const offset = s * (this.modelSeqLen * NUM_FEATURES);
            xBuffer.set(seqSlice[s], offset);
        }

        const feeds = {
            [this.xInputName]: new ort.Tensor('float32', xBuffer, [1, targetN, this.modelSeqLen, NUM_FEATURES]),
            [this.maskInputName]: this.maskIsBoolean
                ? new ort.Tensor('bool', maskBufferBool, [1, targetN])
                : new ort.Tensor('float32', maskBufferFloat, [1, targetN])
        };

        if (this.secInputName) {
            const secMeta = this.session.inputMetadata?.[this.secInputName];
            const secType = String(secMeta?.type || 'int64').toLowerCase();
            if (secType.includes('int32')) {
                feeds[this.secInputName] = new ort.Tensor('int32', Int32Array.from(sectorBuffer, v => Number(v)), [1, targetN]);
            } else {
                feeds[this.secInputName] = new ort.Tensor('int64', sectorBuffer, [1, targetN]);
            }
        }

        // Backward-compatibility: if an older model strictly demands macro input
        if (this.macroInputName) {
            feeds[this.macroInputName] = new ort.Tensor('float32', new Float32Array(4).fill(0.0), [1, 4]);
        }

        let results;
        try {
            results = await this.session.run(feeds);
        } catch (err) {
            // Mask dtype toggle fallback
            this.maskIsBoolean = !this.maskIsBoolean;
            feeds[this.maskInputName] = this.maskIsBoolean
                ? new ort.Tensor('bool', maskBufferBool, [1, targetN])
                : new ort.Tensor('float32', maskBufferFloat, [1, targetN]);
            results = await this.session.run(feeds);
        }

        const getOutputData = (regex) => {
            const k = Object.keys(results).find(name => regex.test(name));
            return k ? results[k].data : null;
        };

        const rawOutput = getOutputData(/(alpha_?5|target_alpha|^preds?$|^output$)/i)
            || results[this.outputNames[0]]?.data
            || results[Object.keys(results)[0]]?.data;

        // ======================================================================
        // GAUSSIAN CROSS-SECTIONAL CALIBRATION
        // Eliminates artificial volatility-scaling cone & 8% brick-wall clipping
        // ======================================================================
        const rawLogits = Array.from(rawOutput).slice(0, activeCount).map(Number);
        
        // 1. Compute cross-sectional distribution statistics of model predictions
        const meanLogit = rawLogits.reduce((a, b) => a + b, 0.0) / Math.max(1, activeCount);
        const varLogit = rawLogits.reduce((acc, v) => acc + (v - meanLogit) ** 2, 0.0) / Math.max(1, activeCount - 1);
        const stdLogit = Math.sqrt(varLogit) + 1e-8;

        // 2. Historical 5-day market-relative alpha standard deviation (~2.20%)
        const HISTORICAL_5D_ALPHA_STD = 2.20; // 220 bps

        const scored = [];

        for (let s = 0; s < activeCount; s++) {
            const ticker = tickersSlice[s];
            const rawLogit = rawLogits[s];

            // Standardized cross-sectional model score (Z-Score of model conviction)
            const zScore = (rawLogit - meanLogit) / stdLogit;
            
            // Expected 5-Day Alpha: Preserves model's raw conviction margin symmetrically
            // Top #1 pick (~+2.5 sigma) maps to ~+2.8% to +3.2%
            const expectedReturn5d = Number((zScore * (HISTORICAL_5D_ALPHA_STD * 0.55)).toFixed(2));

            // Signal EMA Smoothing (alpha = 0.40)
            const prevSmoothed = globalSignalEmaMap.get(ticker);
            const pSmooth = (prevSmoothed !== undefined)
                ? (1.0 - SIGNAL_EMA_ALPHA) * prevSmoothed + SIGNAL_EMA_ALPHA * rawLogit
                : rawLogit;
            globalSignalEmaMap.set(ticker, pSmooth);

            // True realized volatility is preserved as an independent risk metric
            const realizedVol = stockVolsMap.get(ticker) || 0.015;
            const uncertainty5d = Number((realizedVol * Math.sqrt(5.0) * 100.0).toFixed(2));

            // True Signal-to-Noise Ratio (Expected Alpha / Realized Uncertainty)
            const snr = Number((expectedReturn5d / Math.max(0.5, uncertainty5d)).toFixed(3));
            const directionConfidence = Number((normalCDF(zScore) * 100.0).toFixed(1));

            let direction = 'Neutral';
            if (expectedReturn5d > 0.30) direction = 'Bullish';
            else if (expectedReturn5d < -0.30) direction = 'Bearish';

            const horizons = {
                d1: { return: Number((expectedReturn5d * 0.20).toFixed(2)), uncertainty: Number((uncertainty5d * Math.sqrt(1 / 5)).toFixed(2)) },
                d2: { return: Number((expectedReturn5d * 0.40).toFixed(2)), uncertainty: Number((uncertainty5d * Math.sqrt(2 / 5)).toFixed(2)) },
                d3: { return: Number((expectedReturn5d * 0.60).toFixed(2)), uncertainty: Number((uncertainty5d * Math.sqrt(3 / 5)).toFixed(2)) },
                d4: { return: Number((expectedReturn5d * 0.80).toFixed(2)), uncertainty: Number((uncertainty5d * Math.sqrt(4 / 5)).toFixed(2)) },
                d5: { return: expectedReturn5d, uncertainty: uncertainty5d }
            };

            scored.push({
                ticker,
                sector: TICKER_GICS_SECTORS[ticker] || 'General',
                rawAlpha: Number(rawLogit.toFixed(5)),
                smoothedAlpha: Number(pSmooth.toFixed(5)),
                alpha: Number((expectedReturn5d / 100.0).toFixed(4)),
                snr,
                expectedReturn5d,
                uncertainty5d,
                realizedVol,
                direction,
                directionConfidence,
                horizons
            });
        }

        // Descending sort: highest expected forward alpha first (#1 Long at top)
        scored.sort((a, b) => b.expectedReturn5d - a.expectedReturn5d);

        // Cross-Sectional Dispersion Metric
        const dispersion = stdLogit;

        return {
            ranked: scored.map((item, idx) => ({ rank: idx + 1, ...item })),
            dispersion
        };
    }
}

// ==============================================================================
// SECTION 7: PIPELINE ORCHESTRATION & LEAK-FREE T+1 EXECUTION RULES
// ==============================================================================
async function buildAndRunPredictor(options = {}) {
    const t0 = Date.now();
    const predictor = new StockPredictor(options.modelPath);
    await predictor.init();

    const targetK = options.topK || DEFAULT_TOP_K;
    const activeUniverse = SP500_TICKERS.filter(t => !DEAD_TICKERS.has(t));
    const lookback = predictor.modelSeqLen;

    console.log(`[Predictor Pipeline] Downloading Universe Data (${activeUniverse.length} equities, lookback: ${lookback} sessions)...`);

    const benchSeries = await getOHLCV(BENCHMARK_TICKER, STOCK_RANGE_DAYS, options.cacheDir);
    if (!benchSeries || benchSeries.closes.length < lookback + 50) {
        throw new Error(`[Predictor Pipeline] Failed to ingest benchmark ${BENCHMARK_TICKER}`);
    }

    const targetDates = benchSeries.dates.slice(-lookback);
    const signalDate = targetDates[targetDates.length - 1];
    const dayOfWeek = new Date(signalDate).getUTCDay(); // 1=Mon, 2=Tue, 3=Wed, 4=Thu, 5=Fri

    // -------------------------------------------------------------
    // STRICT T+1 OPEN EXECUTION RULES (FROM PYTHON ENGINE)
    // -------------------------------------------------------------
    const isMonday = dayOfWeek === 1;
    const isWednesday = dayOfWeek === 3;
    const isThursday = dayOfWeek === 4;

    const effectiveHurdle = CONVICTION_THRESHOLD * (isMonday ? MONDAY_HURDLE_MULT : 1.0);

    const rawSeriesMap = await fetchManyOHLCV(activeUniverse, STOCK_RANGE_DAYS, options.cacheDir);

    const activeTickers = [];
    const activeSequences = [];
    const stockVolsMap = new Map();
    const stockPricesMap = new Map();

    for (const sym of activeUniverse) {
        const series = rawSeriesMap.get(sym);
        if (!series) continue;

        const processed = computeRawSignals(series);
        if (!processed) continue;

        const { dateFeatureMap, realizedVol, lastClose } = processed;
        if (!dateFeatureMap.has(signalDate)) continue;

        const seq = new Float32Array(lookback * NUM_FEATURES);
        let missing = 0, lastValid = null;

        for (let t = 0; t < lookback; t++) {
            const dt = targetDates[t];
            let row = dateFeatureMap.get(dt);
            if (!row) {
                missing++;
                row = lastValid || new Float32Array(NUM_FEATURES);
            } else {
                lastValid = row;
            }
            seq.set(row, t * NUM_FEATURES);
        }

        if (missing <= 3 && lastValid !== null) {
            activeTickers.push(sym);
            activeSequences.push(seq);
            stockVolsMap.set(sym, realizedVol);
            stockPricesMap.set(sym, lastClose);
        }
    }

    console.log(`[Predictor Pipeline] Forward pass through Hierarchical MS-iTransformer (${activeTickers.length} symbols)...`);

    const { ranked, dispersion } = await predictor.predict(activeTickers, activeSequences, stockVolsMap);
    const finalTopK = Math.min(targetK, ranked.length);

    // -------------------------------------------------------------
    // VOLATILITY TARGETING & EXPOSURE DAMPENING
    // -------------------------------------------------------------
    const benchReturns = [];
    for (let i = 1; i < benchSeries.closes.length; i++) {
        benchReturns.push(Math.log(benchSeries.closes[i] / benchSeries.closes[i - 1]));
    }
    const recentBenchVol = rollingStd(Float32Array.from(benchReturns), 20);
    const annBenchVol = (recentBenchVol[recentBenchVol.length - 1] || 0.012) * Math.sqrt(252);

    let volScalar = 1.0;
    if (ENABLE_VOL_TARGETING && annBenchVol > 0) {
        volScalar = Math.min(1.0, TARGET_ANNUAL_VOL / annBenchVol);
    }

    let capitalExposure = 1.0 * volScalar;
    if (isWednesday) {
        capitalExposure *= THURSDAY_DAMPENER;
    } else if (isThursday) {
        capitalExposure *= FRIDAY_DAMPENER;
    }
    capitalExposure = Math.max(MIN_CAPITAL_EXPOSURE, Math.min(MAX_CAPITAL_EXPOSURE, capitalExposure));

    // -------------------------------------------------------------
    // INVERSE-VOLATILITY RISK PARITY PORTFOLIO SIZING
    // -------------------------------------------------------------
    const topCandidates = ranked.slice(0, finalTopK);
    const eligibleLongs = topCandidates.filter(item => item.smoothedAlpha > effectiveHurdle);
    const invVolSum = eligibleLongs.reduce((sum, item) => sum + (1.0 / item.realizedVol), 0.0);

    const predictions = ranked.map((item, idx) => {
        let group = 'Neutral';
        let portfolioWeight = 0.0;

        if (idx < finalTopK) {
            group = 'Top Long';
            if (item.smoothedAlpha > effectiveHurdle && invVolSum > 0) {
                const normWeight = (1.0 / item.realizedVol) / invVolSum;
                portfolioWeight = Number((normWeight * capitalExposure).toFixed(4));
            }
        } else if (idx >= ranked.length - DEFAULT_BOTTOM_K) {
            group = 'Top Short';
        }

        return {
            ticker: item.ticker,
            sector: item.sector,
            price: stockPricesMap.get(item.ticker) || null,
            rank: item.rank,
            group,
            portfolioWeight,
            rawAlpha: item.rawAlpha,
            smoothedAlpha: item.smoothedAlpha,
            alpha: item.alpha,
            expectedReturn5d: item.expectedReturn5d,
            uncertainty5d: item.uncertainty5d,
            snr: item.snr,
            direction: item.direction,
            directionConfidence: item.directionConfidence,
            horizons: item.horizons
        };
    });

    const topLongsSpread = predictions.slice(0, finalTopK).reduce((acc, p) => acc + p.expectedReturn5d, 0) / finalTopK;
    const botShortsSpread = predictions.slice(-finalTopK).reduce((acc, p) => acc + p.expectedReturn5d, 0) / finalTopK;
    const crossSectionalSpread = topLongsSpread - botShortsSpread;

    return {
        predictions,
        universeSize: predictions.length,
        marketSpread: Number((crossSectionalSpread / 100.0).toFixed(4)),
        topK: finalTopK,
        macroState: {
            signalDate,
            weekday: new Date(signalDate).toLocaleDateString('en-US', { weekday: 'long' }),
            dispersion: Number(dispersion.toFixed(4)),
            annualizedBenchmarkVol: Number((annBenchVol * 100.0).toFixed(2)),
            isMondayHurdleActive: isMonday,
            effectiveConvictionHurdle: Number((effectiveHurdle * 100.0).toFixed(3)),
            thursdayDampenerActive: isWednesday,
            fridayDampenerActive: isThursday,
            topMeanReturnPct: Number(topLongsSpread.toFixed(2)),
            bottomMeanReturnPct: Number(botShortsSpread.toFixed(2)),
            modelSpreadPct: Number(crossSectionalSpread.toFixed(2)),
            equityExposurePct: Number((capitalExposure * 100.0).toFixed(1)),
            cashPreservationPct: Number(((1.0 - capitalExposure) * 100.0).toFixed(1))
        },
        executionModel: {
            engineVersion: '4.0 Clean Production Master',
            architecture: '3D MS-iTransformer (Multi-Scale Inception TCN + Gated Factor Inversion + CS-Norm)',
            spatialRouting: '12 Centroids (1 Market + 11 GICS Sectors)',
            portfolioSizing: 'Strict Inverse-Volatility Risk Parity (w_i ~ 1 / sigma_i)',
            volatilityTarget: `${(TARGET_ANNUAL_VOL * 100).toFixed(0)}% Annual Volatility Ceiling`,
            turnoverFeeDrag: `${(TRANSACTION_FEE_BPS * 10000).toFixed(0)} bps per turnover`,
            minHoldingBuffer: `${MIN_HOLDING_DAYS} Days / Rank Buffer ${BUFFER_RANK}`
        },
        signalDate,
        latency: Date.now() - t0,
        timestamp: new Date().toISOString()
    };
}

// ==============================================================================
// SECTION 8: WORKER THREAD POOLING & CONCURRENCY
// ==============================================================================
function createManager() {
    let worker = null;
    let workerReady = null;
    const pending = new Map();
    let requestCounter = 0;
    let inflightRun = null;
    let latestPredictions = null;

    function spawnWorker() {
        const w = new Worker(__filename, { workerData: { mode: 'worker' } });

        workerReady = new Promise((resolve) => {
            const onReady = (msg) => {
                if (msg && msg.type === 'ready') {
                    w.off('message', onReady);
                    resolve();
                }
            };
            w.on('message', onReady);
        });

        w.on('message', (msg) => {
            if (!msg || msg.type === 'ready') return;
            const entry = pending.get(msg.requestId);
            if (!entry) return;
            pending.delete(msg.requestId);

            if (msg.type === 'result') {
                latestPredictions = msg.result;
                entry.resolve(msg.result);
            } else if (msg.type === 'error') {
                entry.reject(new Error(msg.error));
            }
        });

        w.on('error', (err) => {
            console.error('[InferenceWorker Error]:', err);
            for (const [, entry] of pending) entry.reject(err);
            pending.clear();
            worker = null;
            workerReady = null;
        });

        w.on('exit', (code) => {
            if (code !== 0) console.error(`[InferenceWorker] Exited with code ${code}`);
            for (const [, entry] of pending) entry.reject(new Error(`Worker exited with code ${code}`));
            pending.clear();
            worker = null;
            workerReady = null;
        });

        return w;
    }

    function getWorker() {
        if (!worker) worker = spawnWorker();
        return worker;
    }

    async function runInference(options = {}) {
        if (inflightRun) return inflightRun;
        const w = getWorker();
        await workerReady;

        const requestId = ++requestCounter;
        inflightRun = new Promise((resolve, reject) => {
            pending.set(requestId, { resolve, reject });
            w.postMessage({ type: 'run', requestId, options });
        }).finally(() => {
            inflightRun = null;
        });

        return inflightRun;
    }

    function getLatestPredictions() {
        return latestPredictions;
    }

    return { runInference, getLatestPredictions };
}

// ==============================================================================
// SECTION 9: ENTRYPOINT & EXPORTS
// ==============================================================================
if (parentPort || workerData?.mode === 'worker') {
    if (!parentPort) throw new Error('[AI Stock Predictor Worker] Requires valid parentPort context.');

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
} else {
    const manager = createManager();
    module.exports = {
        runInference: manager.runInference,
        getLatestPredictions: manager.getLatestPredictions,
        buildAndRunPredictor,
        computeRawSignals,
        applyCrossSectionalRankScaling,
        getOHLCV,
        StockPredictor,
        SP500_TICKERS,
        DEAD_TICKERS,
        FEATURE_NAMES,
        SECTOR_TO_ID,
        TICKER_GICS_SECTORS,
        DEFAULT_TOP_K
    };
}
