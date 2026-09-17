/*
 * infer-lab control centre (Node.js, zero dependencies).
 *
 * One page, one port, every view:
 *
 *   Live        scraped Prometheus metrics, rendered as live charts
 *   Metrics     the raw exposition text, as Prometheus sees it
 *   Stats       engine/scheduler/KV internals from /stats
 *   Kernels     which kernel backends this machine actually has
 *   Playground  send a prompt, see the tokens and per-request timings
 *   Load        built-in traffic generator -- no second terminal needed
 *
 * The server proxies the Python API so the browser only ever talks to one
 * origin. That removes the CORS problem and, more importantly, means you do not
 * have to keep a second window open just to put load on the engine.
 *
 * Dependency-free on purpose: `node dashboard.js` must work on a locked-down
 * machine with no npm install and no network access.
 */

'use strict';

const http = require('http');
const fs = require('fs');
const path = require('path');

const TARGET = process.env.INFER_LAB_URL || 'http://127.0.0.1:8000';
const PORT = parseInt(process.env.DASHBOARD_PORT || '3000', 10);
const POLL_MS = parseInt(process.env.POLL_MS || '1000', 10);
const HISTORY = 180;

const targetUrl = new URL(TARGET);

const state = {
  connected: false,
  lastError: null,
  updatedAt: null,
  metrics: {},
  histograms: {},
  rawMetrics: '',
  history: [],
  derived: { tokensPerSecond: 0 },
};

let previous = null;

/* ----------------------------------------------------------- upstream helper */
function upstream(method, urlPath, body) {
  return new Promise((resolve, reject) => {
    const payload = body === undefined ? null : Buffer.from(JSON.stringify(body));
    const headers = {};
    if (payload) {
      headers['Content-Type'] = 'application/json';
      headers['Content-Length'] = payload.length;
    }
    const req = http.request(
      {
        hostname: targetUrl.hostname,
        port: targetUrl.port,
        path: urlPath,
        method,
        headers,
        timeout: 180000,
      },
      (res) => {
        let data = '';
        res.setEncoding('utf8');
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => resolve({ status: res.statusCode, body: data }));
      },
    );
    req.on('timeout', () => req.destroy(new Error('upstream timeout')));
    req.on('error', reject);
    if (payload) req.write(payload);
    req.end();
  });
}

/* ------------------------------------------------------- Prometheus parsing */
function parsePrometheus(text) {
  const scalars = {};
  const histograms = {};

  for (const rawLine of text.split('\n')) {
    const line = rawLine.trim();
    if (!line || line.startsWith('#')) continue;

    const lastSpace = line.lastIndexOf(' ');
    if (lastSpace < 0) continue;

    const left = line.slice(0, lastSpace);
    const value = parseFloat(line.slice(lastSpace + 1));
    if (!Number.isFinite(value)) continue;

    const braceIndex = left.indexOf('{');
    const name = braceIndex < 0 ? left : left.slice(0, braceIndex);
    const labelBlob = braceIndex < 0 ? '' : left.slice(braceIndex + 1, left.lastIndexOf('}'));

    if (name.endsWith('_bucket')) {
      const family = name.slice(0, -'_bucket'.length);
      const le = /le="([^"]+)"/.exec(labelBlob);
      histograms[family] = histograms[family] || { buckets: [] };
      histograms[family].buckets.push({ le: le ? le[1] : '+Inf', count: value });
    } else if (name.endsWith('_sum') || name.endsWith('_count')) {
      const suffix = name.endsWith('_sum') ? '_sum' : '_count';
      const family = name.slice(0, -suffix.length);
      histograms[family] = histograms[family] || { buckets: [] };
      histograms[family][suffix.slice(1)] = value;
    } else {
      const key = labelBlob ? `${name}{${labelBlob}}` : name;
      scalars[key] = value;
      if (labelBlob) scalars[name] = (scalars[name] || 0) + value;
    }
  }
  return { scalars, histograms };
}

/*
 * Linear interpolation inside the matching bucket -- the same thing Prometheus'
 * histogram_quantile() does. The result is only as good as the bucket layout,
 * which is why the server picks buckets that bracket the SLO.
 */
function quantileFromBuckets(histogram, q) {
  if (!histogram || !histogram.count) return null;
  const target = q * histogram.count;
  const buckets = histogram.buckets
    .slice()
    .sort((a, b) => parseFloat(a.le) - parseFloat(b.le));

  let previousLe = 0;
  let previousCount = 0;
  for (const bucket of buckets) {
    if (bucket.count >= target) {
      const upper = parseFloat(bucket.le);
      if (!Number.isFinite(upper)) return previousLe;
      const span = bucket.count - previousCount;
      if (span <= 0) return upper;
      return previousLe + ((target - previousCount) / span) * (upper - previousLe);
    }
    previousLe = parseFloat(bucket.le);
    previousCount = bucket.count;
  }
  return previousLe;
}

/* -------------------------------------------------------------------- scrape */
async function scrape() {
  try {
    const response = await upstream('GET', '/metrics');
    if (response.status !== 200) {
      state.connected = false;
      state.lastError = `HTTP ${response.status}`;
      return;
    }
    const parsed = parsePrometheus(response.body);
    const now = Date.now();

    const decodeTokens = parsed.scalars['infer_lab_tokens_total{phase="decode"}'] || 0;
    if (previous && now > previous.at) {
      const deltaTokens = decodeTokens - previous.decodeTokens;
      const deltaSeconds = (now - previous.at) / 1000;
      state.derived.tokensPerSecond =
        deltaSeconds > 0 ? Math.max(0, deltaTokens / deltaSeconds) : 0;
    }
    previous = { at: now, decodeTokens };

    state.connected = true;
    state.lastError = null;
    state.updatedAt = new Date(now).toISOString();
    state.metrics = parsed.scalars;
    state.histograms = parsed.histograms;
    state.rawMetrics = response.body;

    state.history.push({
      t: now,
      tokensPerSecond: state.derived.tokensPerSecond,
      running: parsed.scalars['infer_lab_running_sequences'] || 0,
      waiting: parsed.scalars['infer_lab_waiting_sequences'] || 0,
      kv: (parsed.scalars['infer_lab_kv_cache_utilization_ratio'] || 0) * 100,
      p50: quantileFromBuckets(parsed.histograms['infer_lab_e2e_milliseconds'], 0.5),
      p99: quantileFromBuckets(parsed.histograms['infer_lab_e2e_milliseconds'], 0.99),
    });
    if (state.history.length > HISTORY) state.history.shift();
  } catch (error) {
    state.connected = false;
    state.lastError = error.message;
  }
}

/* ----------------------------------------------------------- load generator */
const load = {
  running: false,
  concurrency: 0,
  maxTokens: 64,
  promptChars: 200,
  sent: 0,
  completed: 0,
  failed: 0,
  tokens: 0,
  startedAt: null,
  workers: 0,
  latencies: [],
};

const SHARED_PREFIX =
  'You are a helpful assistant operating under strict latency budgets. ';

function makePrompt() {
  // A shared prefix exercises RadixAttention; the random tail keeps the whole
  // prompt from being a trivial cache hit, which would be unrealistic.
  const unique = Math.random().toString(36).slice(2);
  const fillerLength = Math.max(0, load.promptChars - SHARED_PREFIX.length - unique.length);
  return SHARED_PREFIX + unique + ' ' + 'x'.repeat(fillerLength);
}

async function loadWorker() {
  load.workers += 1;
  try {
    while (load.running) {
      const startedAt = Date.now();
      load.sent += 1;
      try {
        const response = await upstream('POST', '/generate', {
          prompt: makePrompt(),
          max_tokens: load.maxTokens,
          temperature: 0.0,
          ignore_eos: true,
        });
        if (response.status === 200) {
          const parsed = JSON.parse(response.body);
          load.completed += 1;
          load.tokens += parsed.output_tokens || 0;
          load.latencies.push(Date.now() - startedAt);
          if (load.latencies.length > 500) load.latencies.shift();
        } else {
          load.failed += 1;
        }
      } catch (error) {
        load.failed += 1;
        if (!load.running) break;
        // Back off briefly so a dead server does not spin the event loop.
        await new Promise((resolve) => setTimeout(resolve, 250));
      }
    }
  } finally {
    load.workers -= 1;
  }
}

function startLoad(concurrency, maxTokens, promptChars) {
  stopLoad();
  load.running = true;
  load.concurrency = concurrency;
  load.maxTokens = maxTokens;
  load.promptChars = promptChars;
  load.sent = 0;
  load.completed = 0;
  load.failed = 0;
  load.tokens = 0;
  load.latencies = [];
  load.startedAt = Date.now();
  for (let i = 0; i < concurrency; i++) {
    // An unhandled rejection terminates the Node process by default, which would
    // kill the whole dashboard because one request failed. Contain it here.
    loadWorker().catch((error) => {
      load.failed += 1;
      state.lastError = `load worker crashed: ${error.message}`;
    });
  }
}

function stopLoad() {
  load.running = false;
  load.concurrency = 0;
}

function percentile(values, q) {
  if (!values.length) return null;
  const sorted = values.slice().sort((a, b) => a - b);
  const rank = Math.min(sorted.length, Math.max(1, Math.ceil((q / 100) * sorted.length)));
  return sorted[rank - 1];
}

function loadStatus() {
  const seconds = load.startedAt ? (Date.now() - load.startedAt) / 1000 : 0;
  return {
    running: load.running,
    concurrency: load.concurrency,
    maxTokens: load.maxTokens,
    workers: load.workers,
    sent: load.sent,
    completed: load.completed,
    failed: load.failed,
    tokens: load.tokens,
    elapsed: seconds,
    requestsPerSecond: seconds > 0 ? load.completed / seconds : 0,
    tokensPerSecond: seconds > 0 ? load.tokens / seconds : 0,
    p50: percentile(load.latencies, 50),
    p90: percentile(load.latencies, 90),
    p99: percentile(load.latencies, 99),
  };
}

/* -------------------------------------------------------------------- routes */
function sendJson(res, payload, status = 200) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    'Content-Type': 'application/json; charset=utf-8',
    'Cache-Control': 'no-store',
  });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve) => {
    let data = '';
    req.setEncoding('utf8');
    req.on('data', (chunk) => { data += chunk; });
    req.on('end', () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch (error) {
        resolve({});
      }
    });
  });
}

function liveState() {
  const s = state.metrics;
  return {
    connected: state.connected,
    lastError: state.lastError,
    updatedAt: state.updatedAt,
    target: TARGET,
    summary: {
      tokensPerSecond: state.derived.tokensPerSecond,
      running: s['infer_lab_running_sequences'] || 0,
      waiting: s['infer_lab_waiting_sequences'] || 0,
      kvUtilization: (s['infer_lab_kv_cache_utilization_ratio'] || 0) * 100,
      kvBlocksFree: s['infer_lab_kv_blocks_free'] || 0,
      requests: s['infer_lab_requests_total'] || 0,
      preemptions: s['infer_lab_preemptions_total'] || 0,
      prefixCacheTokens: s['infer_lab_prefix_cache_hit_tokens_total'] || 0,
      engineSteps: s['infer_lab_engine_steps_total'] || 0,
      prefillTokens: s['infer_lab_tokens_total{phase="prefill"}'] || 0,
      decodeTokens: s['infer_lab_tokens_total{phase="decode"}'] || 0,
      ttftP50: quantileFromBuckets(state.histograms['infer_lab_ttft_milliseconds'], 0.5),
      ttftP99: quantileFromBuckets(state.histograms['infer_lab_ttft_milliseconds'], 0.99),
      e2eP50: quantileFromBuckets(state.histograms['infer_lab_e2e_milliseconds'], 0.5),
      e2eP99: quantileFromBuckets(state.histograms['infer_lab_e2e_milliseconds'], 0.99),
      stepP50: quantileFromBuckets(
        state.histograms['infer_lab_engine_step_milliseconds'], 0.5),
    },
    history: state.history,
    load: loadStatus(),
  };
}

const server = http.createServer(async (req, res) => {
  const url = new URL(req.url, `http://127.0.0.1:${PORT}`);
  const route = url.pathname;

  try {
    if (route === '/api/state') return sendJson(res, liveState());

    if (route === '/api/raw-metrics') {
      res.writeHead(200, { 'Content-Type': 'text/plain; charset=utf-8' });
      return res.end(state.rawMetrics || '# no metrics scraped yet');
    }

    // Straight proxies -- the browser only ever talks to this origin.
    if (route === '/api/stats' || route === '/api/kernels' || route === '/api/info') {
      const upstreamPath = { '/api/stats': '/stats', '/api/kernels': '/kernels',
                             '/api/info': '/info' }[route];
      const response = await upstream('GET', upstreamPath);
      res.writeHead(response.status, { 'Content-Type': 'application/json; charset=utf-8' });
      return res.end(response.body);
    }

    if (route === '/api/generate' && req.method === 'POST') {
      const body = await readBody(req);
      const response = await upstream('POST', '/generate', body);
      res.writeHead(response.status, { 'Content-Type': 'application/json; charset=utf-8' });
      return res.end(response.body);
    }

    if (route === '/api/load/start' && req.method === 'POST') {
      const body = await readBody(req);
      const concurrency = Math.min(64, Math.max(1, parseInt(body.concurrency, 10) || 8));
      const maxTokens = Math.min(512, Math.max(1, parseInt(body.maxTokens, 10) || 64));
      const promptChars = Math.min(4000, Math.max(20, parseInt(body.promptChars, 10) || 200));
      startLoad(concurrency, maxTokens, promptChars);
      return sendJson(res, loadStatus());
    }

    if (route === '/api/load/stop' && req.method === 'POST') {
      stopLoad();
      return sendJson(res, loadStatus());
    }

    if (route === '/api/load/status') return sendJson(res, loadStatus());

    if (route === '/healthz') return sendJson(res, { status: 'ok' });

    // Everything else serves the single-page app.
    const file = path.join(__dirname, 'public', 'index.html');
    return fs.readFile(file, (error, content) => {
      if (error) {
        res.writeHead(500, { 'Content-Type': 'text/plain' });
        return res.end('dashboard template missing');
      }
      res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
      res.end(content);
    });
  } catch (error) {
    return sendJson(res, { error: error.message }, 502);
  }
});

server.listen(PORT, '127.0.0.1', () => {
  console.log(`infer-lab control centre on http://127.0.0.1:${PORT} (upstream ${TARGET})`);
  scrape();
  setInterval(scrape, POLL_MS);
});

function shutdown() {
  stopLoad();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 2000).unref();
}
process.on('SIGINT', shutdown);
process.on('SIGTERM', shutdown);
