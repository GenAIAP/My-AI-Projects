const { parentPort } = require('worker_threads');
const { runModelInference } = require('./inference-core');

if (!parentPort) {
    throw new Error('inference-worker.js must be run as a worker_threads Worker');
}

parentPort.on('message', async (msg) => {
    if (!msg || msg.type !== 'run') return;

    try {
        const result = await runModelInference(msg.options || {});
        parentPort.postMessage({ type: 'result', requestId: msg.requestId, result });
    } catch (err) {
        parentPort.postMessage({
            type: 'error',
            requestId: msg.requestId,
            error: (err && err.message) || String(err)
        });
    }
});

// Let the main thread know the worker booted successfully
parentPort.postMessage({ type: 'ready' });