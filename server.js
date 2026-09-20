const express = require('express');
const http = require('http');
const { Server } = require('socket.io');
const cors = require('cors');
const path = require('path');
const {
  runModelInference,
  getLatestPredictions,
  getStockForecast,
  buildPredictionHtmlReport
} = require('./inference');

const app = express();
app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: '*',
    methods: ['GET', 'POST']
  }
});

app.get('/api/health', (req, res) => {
  res.json({ status: 'ok', time: new Date().toISOString() });
});

app.get('/stock_predictor.html', (req, res) => {
  res.sendFile(path.join(__dirname, 'public', 'stock_predictor.html'));
});

app.post('/api/run-inference', async (req, res) => {
  try {
    const result = await runModelInference({ autoOpen: false, persistHtml: false });
    io.emit('predictionsUpdated', result);
    res.json(result);
  } catch (err) {
    console.error('Inference request failed:', err);
    res.status(500).json({ error: err.message || 'Inference failed' });
  }
});

app.get('/api/stock-forecast/:ticker', async (req, res) => {
  try {
    const result = await getStockForecast(req.params.ticker);
    if (!result) {
      return res.status(404).json({ error: 'No stock forecast available yet' });
    }
    return res.json(result);
  } catch (err) {
    console.error('Stock forecast request failed:', err.message || err);
    return res.status(500).json({ error: 'Unable to load stock forecast data' });
  }
});

io.on('connection', (socket) => {
  console.log(`Client connected: ${socket.id}`);

  socket.emit('welcome', {
    message: 'Connected to Socket.IO server',
    id: socket.id
  });

  socket.on('requestPredictions', async () => {
    try {
      let result = getLatestPredictions();
      if (!result) {
        result = await runModelInference({ autoOpen: false, persistHtml: false });
      }
      socket.emit('predictionsUpdated', result);
    } catch (err) {
      console.error('requestPredictions failed:', err.message || err);
      socket.emit('predictionsError', { error: err.message || 'Unable to load stock predictions' });
    }
  });

  socket.on('message', (data) => {
    console.log(`Received message from ${socket.id}:`, data);
    io.emit('message', {
      sender: socket.id,
      data,
      timestamp: new Date().toISOString()
    });
  });

  socket.on('disconnect', (reason) => {
    console.log(`Client disconnected: ${socket.id} (${reason})`);
  });
});

async function initializePredictor() {
  try {
    const result = await runModelInference({ autoOpen: false, persistHtml: false });
    console.log(`Initial stock inference ready: ${result.universeSize} stocks | spread=${result.marketSpread.toFixed(4)} | latency=${result.latency}ms`);
  } catch (err) {
    console.error('Initial inference failed:', err.message || err);
  }
}

const PORT = process.env.PORT || 10000;
const HOST = '0.0.0.0'
server.listen(PORT, HOST, () => {
  console.log(`Socket.IO server running on port ${PORT}`);
  setImmediate(() => {
    //initializePredictor();
  });
});
