const path = require('path');
const { Worker } = require('worker_threads');

let worker = null;
let workerReady = null; // promise that resolves once the worker sends 'ready'
const pending = new Map(); // requestId -> { resolve, reject }
let requestCounter = 0;
let inflightRun = null; // single-flight: reuse an in-progress run instead of starting a new one

let latestPredictions = null;

function spawnWorker() {
    const w = new Worker(path.join(__dirname, 'inference-worker.js'));

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
        console.error('Inference worker error:', err);
        for (const [, entry] of pending) entry.reject(err);
        pending.clear();
        worker = null;
        workerReady = null;
    });

    w.on('exit', (code) => {
        if (code !== 0) {
            console.error(`Inference worker stopped unexpectedly with exit code ${code}`);
        }
        for (const [, entry] of pending) {
            entry.reject(new Error(`Inference worker exited with code ${code}`));
        }
        pending.clear();
        worker = null;
        workerReady = null;
    });

    return w;
}

function getWorker() {
    if (!worker) {
        worker = spawnWorker();
    }
    return worker;
}

async function runInference(options = {}) {
    // If a run is already in progress, piggyback on it rather than starting a duplicate
    if (inflightRun) return inflightRun;

    const w = getWorker();
    await workerReady; // make sure the worker has finished booting before we message it

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

module.exports = {
    runInference,
    getLatestPredictions
};