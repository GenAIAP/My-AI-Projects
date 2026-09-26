require('dotenv').config();

const express = require('express');
const http = require('http');
const crypto = require('crypto');
const fs = require('fs');
const { Server } = require('socket.io');
const cors = require('cors');
const path = require('path');

const { runInference, getLatestPredictions } = require('./ai-stock-predictor');
const { getStockForecast } = require('./forecast-utils');
const { buildPredictionHtmlReport } = require('./html-report');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: false }));

const adminCredentials = JSON.parse(process.env.ADMIN_CREDENTIALS || '{}');
const sessionSecret = process.env.ADMIN_SESSION_SECRET;
const adminCookieName = 'admin_session';
const adminSessionDuration = 8 * 60 * 60 * 1000;
const adminBanStorePath = path.join(__dirname, '.admin-bans.json');
const activeBans = new Map();
const adminSites = [
    { path: '/', label: 'Home' },
    { path: '/chat.html', label: 'Chat' },
    { path: '/stock_predictor.html', label: 'Stock Predictor' },
    { path: '/results_chart.html', label: 'Results Chart' }
];
const adminSitePaths = new Set(adminSites.map(site => site.path));

try {
    const storedBans = JSON.parse(fs.readFileSync(adminBanStorePath, 'utf8'));
    for (const [address, expiresAt] of Object.entries(storedBans)) {
        if (Number.isFinite(expiresAt) && expiresAt > Date.now()) activeBans.set(address, expiresAt);
    }
} catch (error) {
    if (error.code !== 'ENOENT') throw error;
}

if (!sessionSecret || sessionSecret.length < 32 ||
    !adminCredentials || typeof adminCredentials !== 'object' ||
    Array.isArray(adminCredentials) || Object.keys(adminCredentials).length === 0 ||
    Object.values(adminCredentials).some(password => typeof password !== 'string' || !password)) {
    throw new Error('Set ADMIN_CREDENTIALS and a 32+ character ADMIN_SESSION_SECRET in .env');
}

function safeStringEqual(left, right) {
    const leftBuffer = Buffer.from(String(left));
    const rightBuffer = Buffer.from(String(right));
    return leftBuffer.length === rightBuffer.length && crypto.timingSafeEqual(leftBuffer, rightBuffer);
}

function getSocketAddress(socket) {
    const address = socket.handshake.address;
    return normalizeAddress(address);
}

function normalizeAddress(address) {
    return address?.startsWith('::ffff:') ? address.slice(7) : address;
}

function getSocketSite(socket) {
    try {
        return new URL(socket.handshake.headers.referer).pathname || '/';
    } catch {
        return '/';
    }
}

function getSiteBanKey(address, site) {
    return JSON.stringify([address, site]);
}

function getBanExpiry(address) {
    const expiresAt = activeBans.get(address);
    if (!expiresAt) return null;
    if (expiresAt <= Date.now()) {
        activeBans.delete(address);
        return null;
    }
    return expiresAt;
}

function getBannedSites(address) {
    if (getBanExpiry(address)) return adminSites.map(site => site.path);
    return adminSites
        .map(site => site.path)
        .filter(site => getBanExpiry(getSiteBanKey(address, site)));
}

function getSiteBanExpiries(address) {
    const globalExpiry = getBanExpiry(address);
    return Object.fromEntries(adminSites.flatMap(site => {
        const expiresAt = globalExpiry || getBanExpiry(getSiteBanKey(address, site.path));
        return expiresAt ? [[site.path, expiresAt]] : [];
    }));
}

function getRequestedSite(req) {
    if (adminSitePaths.has(req.path)) return req.path;
    try {
        const refererPath = new URL(req.get('referer')).pathname;
        return adminSitePaths.has(refererPath) ? refererPath : null;
    } catch {
        return null;
    }
}

app.use((req, res, next) => {
    if (req.path.startsWith('/admin') || req.path.startsWith('/socket.io')) return next();
    const site = getRequestedSite(req);
    if (!site) return next();

    const address = normalizeAddress(req.socket.remoteAddress);
    const expiresAt = getBanExpiry(address) || getBanExpiry(getSiteBanKey(address, site));
    if (!expiresAt) return next();
    const remainingSeconds = Math.max(0, Math.ceil((expiresAt - Date.now()) / 1000));

    return res.status(403).set('Cache-Control', 'no-store').type('html').send(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Access denied</title>
<style>*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;background:#171311;color:#f5eee9;font:16px/1.5 system-ui,sans-serif}main{width:min(100%,460px);padding:36px;border-left:4px solid #ee765c;background:#241c19}h1{margin:0 0 10px;font-size:30px}p{margin:0;color:#c5b7b0}</style></head>
<body><main><h1>Access denied</h1><p>This page is unavailable until the ban expires.</p><p>Time remaining: <strong id="ban-countdown">${remainingSeconds} seconds</strong></p></main>
<script>
const banExpiresAt = ${expiresAt};
const countdown = document.getElementById('ban-countdown');
function updateBanCountdown() {
    const seconds = Math.max(0, Math.ceil((banExpiresAt - Date.now()) / 1000));
    const minutes = Math.floor(seconds / 60);
    const hours = Math.floor(minutes / 60);
    const days = Math.floor(hours / 24);
    const display = days > 0
        ? days + 'd ' + (hours % 24) + 'h ' + (minutes % 60) + 'm ' + (seconds % 60) + 's'
        : hours > 0
            ? hours + 'h ' + (minutes % 60) + 'm ' + (seconds % 60) + 's'
            : minutes > 0
                ? minutes + 'm ' + (seconds % 60) + 's'
                : seconds + ' seconds';
    countdown.textContent = display;
    if (seconds === 0) window.location.reload();
    else window.setTimeout(updateBanCountdown, 1000);
}
updateBanCountdown();
</script></body></html>`);
});

app.get('/stock_predictor.html', async (req, res) => {
    res.set('Cache-Control', 'no-store');
    try {
        const result = getLatestPredictions();
        if (!result) {
            triggerBackgroundInference();
            return res.status(200).send(
                '<html><body style="background:#0b0f19;color:#f1f5f9;font-family:sans-serif;padding:40px;text-align:center;">' +
                '<h2>Warming up predictions…</h2><p>This page will refresh automatically in a few seconds.</p>' +
                '<script>setTimeout(() => location.reload(), 4000);</script>' +
                '</body></html>'
            );
        }
        const html = buildPredictionHtmlReport(result);
        res.set('Content-Type', 'text/html');
        res.send(html);
    } catch (err) {
        console.error('Failed to render stock predictor page:', err);
        res.status(500).send('Failed to generate predictor page');
    }
});

app.use(express.static(path.join(__dirname, 'public')));

function persistBans() {
    const currentBans = Object.fromEntries(
        [...activeBans].filter(([, expiresAt]) => expiresAt > Date.now())
    );
    const temporaryPath = `${adminBanStorePath}.tmp`;
    fs.writeFileSync(temporaryPath, JSON.stringify(currentBans, null, 2));
    fs.renameSync(temporaryPath, adminBanStorePath);
}

function createAdminSession(username) {
    const payload = Buffer.from(JSON.stringify({
        username,
        expiresAt: Date.now() + adminSessionDuration
    })).toString('base64url');
    const signature = crypto.createHmac('sha256', sessionSecret).update(payload).digest('base64url');
    return `${payload}.${signature}`;
}

function getAdminSession(req) {
    const cookieHeader = req.headers.cookie || '';
    const cookie = cookieHeader.split(';').map(value => value.trim())
        .find(value => value.startsWith(`${adminCookieName}=`));
    if (!cookie) return null;

    const token = cookie.slice(adminCookieName.length + 1);
    const [payload, suppliedSignature] = token.split('.');
    if (!payload || !suppliedSignature) return null;

    const expectedSignature = crypto.createHmac('sha256', sessionSecret)
        .update(payload)
        .digest();
    let actualSignature;
    try {
        actualSignature = Buffer.from(suppliedSignature, 'base64url');
    } catch {
        return null;
    }
    if (actualSignature.length !== expectedSignature.length ||
        !crypto.timingSafeEqual(actualSignature, expectedSignature)) return null;

    try {
        const session = JSON.parse(Buffer.from(payload, 'base64url').toString());
        if (session.expiresAt <= Date.now() || !Object.hasOwn(adminCredentials, session.username)) return null;
        return session;
    } catch {
        return null;
    }
}

function renderAdminLogin(res) {
    res.set('Cache-Control', 'no-store').type('html').send(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Admin sign in</title><style>
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;background:#101616;color:#eef2ed;font:16px/1.5 system-ui,sans-serif}
main{width:min(100%,380px);padding:32px;border:1px solid #35413b;background:#19211d}h1{margin:0 0 8px;font-size:26px}p{margin:0 0 24px;color:#aebbb2}label{display:block;margin:16px 0 6px;font-size:14px}input{width:100%;padding:12px;border:1px solid #46534b;background:#111814;color:#fff;font:inherit}button{width:100%;margin-top:22px;padding:12px;border:0;background:#b9e36a;color:#14200d;font:700 15px system-ui,sans-serif;cursor:pointer}button:hover{background:#c9f47a}
</style></head><body><main><h1>Admin sign in</h1><p>Restricted access</p><form method="post" action="/admin/login">
<label for="username">Username</label><input id="username" name="username" autocomplete="username" required>
<label for="password">Password</label><input id="password" name="password" type="password" autocomplete="current-password" required>
<button type="submit">Sign in</button></form></main></body></html>`);
}

function renderAccessDenied(res) {
    res.status(403).set('Cache-Control', 'no-store').type('html').send(`<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Access denied</title><style>
*{box-sizing:border-box}body{margin:0;min-height:100vh;display:grid;place-items:center;padding:24px;background:#171311;color:#f5eee9;font:16px/1.5 system-ui,sans-serif}main{width:min(100%,480px);padding:36px;border-left:4px solid #ee765c;background:#241c19}h1{margin:0 0 10px;font-size:30px}p{margin:0 0 24px;color:#c5b7b0}a{color:#f4a28e}
</style></head><body><main><h1>Access denied</h1><p>This account is not authorized to access the admin panel.</p><a href="/admin">Return to sign in</a></main></body></html>`);
}

app.get('/admin', (req, res) => {
    if (!getAdminSession(req)) return renderAdminLogin(res);
    res.set('Cache-Control', 'no-store').sendFile(path.join(__dirname, 'admin.html'));
});

app.post('/admin/login', (req, res) => {
    const { username, password } = req.body || {};
    const expectedPassword = typeof username === 'string' ? adminCredentials[username] : null;
    if (typeof expectedPassword !== 'string' || typeof password !== 'string' ||
        !safeStringEqual(password, expectedPassword)) return renderAccessDenied(res);

    res.cookie(adminCookieName, createAdminSession(username), {
        httpOnly: true,
        sameSite: 'strict',
        secure: process.env.NODE_ENV === 'production',
        path: '/admin',
        maxAge: adminSessionDuration
    });
    res.redirect(303, '/admin');
});

app.post('/admin/logout', (req, res) => {
    res.clearCookie(adminCookieName, { httpOnly: true, sameSite: 'strict', path: '/admin' });
    res.redirect(303, '/admin');
});

app.get('/admin/users', (req, res) => {
    if (!getAdminSession(req)) return res.status(403).json({ error: 'Access denied' });
    const users = [...io.sockets.sockets.values()].map(socket => ({
        id: socket.id,
        name: normalizeName(socket.data.name),
        site: getSocketSite(socket),
        bannedSites: getBannedSites(getSocketAddress(socket)),
        siteBanExpiries: getSiteBanExpiries(getSocketAddress(socket)),
        connectedAt: socket.data.connectedAt
    }));
    res.set('Cache-Control', 'no-store').json({ users, sites: adminSites });
});

app.post('/admin/users/:id/site-bans', (req, res) => {
    if (!getAdminSession(req)) return res.status(403).json({ error: 'Access denied' });
    const socket = io.sockets.sockets.get(req.params.id);
    if (!socket) return res.status(404).json({ error: 'User is no longer online' });

    const { siteBans } = req.body || {};
    if (!Array.isArray(siteBans) || siteBans.some(ban =>
        !ban || !adminSitePaths.has(ban.site) || !Number.isInteger(ban.duration) ||
        ban.duration < 1 || ban.duration > 30 * 24 * 60 * 60
    )) {
        return res.status(400).json({ error: 'Choose valid sites and ban durations' });
    }

    const address = getSocketAddress(socket);
    if (!address) return res.status(400).json({ error: 'Could not identify this connection' });
    const uniqueSiteBans = [...new Map(siteBans.map(ban => [ban.site, ban])).values()];
    const previousBans = new Map(activeBans);

    activeBans.delete(address);
    for (const site of adminSites) activeBans.delete(getSiteBanKey(address, site.path));

    const expiresAtBySite = new Map();
    for (const ban of uniqueSiteBans) {
        const expiresAt = Date.now() + ban.duration * 1000;
        activeBans.set(getSiteBanKey(address, ban.site), expiresAt);
        expiresAtBySite.set(ban.site, expiresAt);
    }

    try {
        persistBans();
    } catch (error) {
        activeBans.clear();
        for (const [key, expiry] of previousBans) activeBans.set(key, expiry);
        console.error('Failed to persist site bans:', error.message || error);
        return res.status(500).json({ error: 'Could not save site bans' });
    }

    for (const candidate of io.sockets.sockets.values()) {
        const duration = uniqueSiteBans.find(ban => ban.site === getSocketSite(candidate))?.duration;
        if (getSocketAddress(candidate) !== address || !duration) continue;
        io.to(candidate.id).emit('ban', duration);
        candidate.disconnect(true);
    }

    return res.json({ siteBanExpiries: Object.fromEntries(expiresAtBySite) });
});

app.post('/admin/users/:id/ban', (req, res) => {
    if (!getAdminSession(req)) return res.status(403).json({ error: 'Access denied' });
    const socket = io.sockets.sockets.get(req.params.id);
    if (!socket) return res.status(404).json({ error: 'User is no longer online' });

    const requestedDuration = req.body?.duration;
    const duration = Number.isInteger(requestedDuration) && requestedDuration > 0
        ? Math.min(requestedDuration, 30 * 24 * 60 * 60)
        : 30;
    const address = getSocketAddress(socket);
    const site = getSocketSite(socket);
    const scope = req.body?.scope === 'all' ? 'all' : 'page';
    if (!address) return res.status(400).json({ error: 'Could not identify this connection' });

    const expiresAt = Date.now() + duration * 1000;
    activeBans.set(scope === 'all' ? address : getSiteBanKey(address, site), expiresAt);
    try {
        persistBans();
    } catch (error) {
        activeBans.delete(scope === 'all' ? address : getSiteBanKey(address, site));
        console.error('Failed to persist admin ban:', error.message || error);
        return res.status(500).json({ error: 'Could not save the ban' });
    }

    const name = normalizeName(socket.data.name);
    const affectedSockets = [...io.sockets.sockets.values()].filter(candidate =>
        getSocketAddress(candidate) === address &&
        (scope === 'all' || getSocketSite(candidate) === site)
    );
    for (const affectedSocket of affectedSockets) {
        io.to(affectedSocket.id).emit('ban', duration);
        affectedSocket.disconnect(true);
    }
    return res.json({ message: `${name} was banned for ${duration} seconds`, expiresAt, scope, site });
});

const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: '*', methods: ['GET', 'POST'] }
});

io.use((socket, next) => {
    const address = getSocketAddress(socket);
    const site = getSocketSite(socket);
    const expiresAt = getBanExpiry(address) || getBanExpiry(getSiteBanKey(address, site));
    if (!expiresAt) return next();

    const error = new Error('This connection is temporarily banned');
    error.data = {
        code: 'USER_BANNED',
        expiresAt,
        remainingSeconds: Math.ceil((expiresAt - Date.now()) / 1000)
    };
    return next(error);
});

let inferenceInProgress = false;
let chatHistory = [];

function normalizeName(name) {
    const clean = String(name || '').trim();
    if (!clean) return 'Guest';
    return clean.slice(0, 24);
}

function normalizeMessagePayload(payload) {
    if (typeof payload === 'string') {
        return { name: 'Guest', text: payload, timestamp: Date.now() };
    }

    return {
        name: normalizeName(payload?.name),
        text: String(payload?.text || payload?.message || '').trim(),
        timestamp: Number(payload?.timestamp) || Date.now()
    };
}

function triggerBackgroundInference() {
    if (inferenceInProgress) return;
    inferenceInProgress = true;

    runInference({})
        .then(result => {
            io.emit('predictionsUpdated', result);
        })
        .catch(err => {
            console.error('Background inference failed:', err.message || err);
            io.emit('predictionsError', { error: err.message || 'Inference failed' });
        })
        .finally(() => {
            inferenceInProgress = false;
        });
}

app.get('/api/health', (req, res) => {
    res.json({ status: 'ok', time: new Date().toISOString() });
});

// Kick off inference in the background; never block the response on it.
app.post('/api/run-inference', async (req, res) => {
    triggerBackgroundInference();
    const existing = getLatestPredictions();
    if (existing) {
        // Respond immediately with what we have; the fresh result will arrive via socket 'predictionsUpdated'
        return res.json({ message: 'Inference started; live update will follow', ...existing });
    }
    return res.status(202).json({ message: 'Inference started, no cached predictions yet' });
});

app.get('/api/stock-forecast/:ticker', async (req, res) => {
  try {
    const ticker = String(req.params.ticker || '')
      .trim()
      .toUpperCase();
    const result = getLatestPredictions();
    if (!result) {
      triggerBackgroundInference();
      return res
        .status(202)
        .json({ error: 'Predictions still initializing, try again shortly' });
    }
    const forecast = await getStockForecast(ticker, result);
    if (!forecast) {
      return res
        .status(404)
        .json({ error: `No stock forecast available for ${ticker}` });
    }
    return res.json(forecast);
  } catch (err) {
    console.error('Stock forecast request failed:', err.message || err);
    return res
      .status(500)
      .json({ error: 'Unable to load stock forecast data' });
  }
});

io.on('connection', (socket) => {
    socket.data.connectedAt = new Date().toISOString();
    console.log(`Client connected: ${socket.id}`);

    socket.emit('welcome', { message: 'Connected to Socket.IO server', id: socket.id });
    socket.emit('chatHistory', chatHistory);

    socket.on('setName', (name) => {
        const cleanName = normalizeName(name);
        socket.data.name = cleanName;
        socket.emit('nameSet', { name: cleanName });
        io.emit('systemMessage', {
            type: 'system',
            message: `${cleanName} joined the chat`,
            timestamp: Date.now()
        });
    });

    socket.on('requestPredictions', () => {
        const result = getLatestPredictions();
        if (result) {
            socket.emit('predictionsUpdated', result);
        } else {
            triggerBackgroundInference();
            socket.emit('predictionsError', { error: 'Predictions still initializing' });
        }
    });

    socket.on('message', (payload) => {
        const message = normalizeMessagePayload(payload);
        if (!message.text) return;

        const finalMessage = {
            name: message.name || socket.data.name || 'Guest',
            text: message.text,
            timestamp: message.timestamp || Date.now(),
            senderId: socket.id
        };

        chatHistory = [...chatHistory, finalMessage].slice(-100);
        io.emit('message', finalMessage);
    });

    socket.on('banUser', (user) => {
        io.emit('userBanned', user);
    })

    socket.on('disconnect', (reason) => {
        console.log(`Client disconnected: ${socket.id} (${reason})`);
    });
});

const PORT = process.env.PORT || 10000;
const HOST = '0.0.0.0';

server.listen(PORT, HOST, () => {
    console.log(`Socket.IO server running on port ${PORT}`);
    // Warm up predictions in the background worker — fire-and-forget, never blocks startup
    triggerBackgroundInference();
});
