const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const path = require('path');

const { runInference, getLatestPredictions } = require('./inference-manager');
const { getStockForecast } = require('./forecast-utils');
const { buildPredictionHtmlReport } = require('./html-report');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const server = http.createServer(app);
const io = new Server(server, {
    cors: { origin: '*', methods: ['GET', 'POST'] }
});

let inferenceInProgress = false;

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

app.get('/stock_predictor.html', async (req, res) => {
    try {
        let result = getLatestPredictions();
        if (!result) {
            triggerBackgroundInference();
            return res.status(200).send(
                '<html><body style="background:#0b0f19;color:#f1f5f9;font-family:sans-serif;padding:40px;text-align:center;">' +
                '<h2>Warming up predictions…</h2><p>This page will refresh automatically in a few seconds.</p>' +
                '<script>setTimeout(() => location.reload(), 4000);</script>' +
                '</body></html>'
            );
        }
        const html = buildPredictionHtmlReport(
            result.predictions,
            result.topK / result.universeSize || 0.10,
            result.marketSpread
        );
        res.set('Content-Type', 'text/html');
        res.send(html);
    } catch (err) {
        console.error('Failed to render stock predictor page:', err);
        res.status(500).send('Failed to generate predictor page');
    }
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
        const result = getLatestPredictions();
        if (!result) {
            triggerBackgroundInference();
            return res.status(202).json({ error: 'Predictions still initializing, try again shortly' });
        }
        const forecast = await getStockForecast(req.params.ticker, result);
        if (!forecast) {
            return res.status(404).json({ error: 'No stock forecast available yet' });
        }
        return res.json(forecast);
    } catch (err) {
        console.error('Stock forecast request failed:', err.message || err);
        return res.status(500).json({ error: 'Unable to load stock forecast data' });
    }
});

io.on('connection', (socket) => {
    console.log(`Client connected: ${socket.id}`);

    socket.emit('welcome', { message: 'Connected to Socket.IO server', id: socket.id });

    socket.on('requestPredictions', () => {
        const result = getLatestPredictions();
        if (result) {
            socket.emit('predictionsUpdated', result);
        } else {
            triggerBackgroundInference();
            socket.emit('predictionsError', { error: 'Predictions still initializing' });
        }
    });

    socket.on('message', (data) => {
        console.log(`Received message from ${socket.id}:`, data);
        io.emit('message', { sender: socket.id, data, timestamp: new Date().toISOString() });
    });

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
