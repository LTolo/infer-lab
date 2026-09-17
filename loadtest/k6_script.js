/*
 * k6 load test for infer-lab.
 *
 * k6 is a single static binary -- no runtime, no container, no admin rights.
 * Download it, put it on PATH, then:
 *
 *   k6 run loadtest/k6_script.js
 *   k6 run -e BASE_URL=http://127.0.0.1:8000 -e MAX_TOKENS=64 loadtest/k6_script.js
 *
 * The load profile is staged rather than flat on purpose: a flat load tells you
 * whether the server survives, a staged ramp tells you *where* it stops scaling.
 * Watch the point where token throughput plateaus but p99 keeps climbing -- that
 * is the KV cache saturating and the scheduler starting to queue.
 */

import http from 'k6/http';
import { check, sleep } from 'k6';
import { Trend, Counter, Rate } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'http://127.0.0.1:8000';
const MAX_TOKENS = parseInt(__ENV.MAX_TOKENS || '32', 10);
const PROMPT_CHARS = parseInt(__ENV.PROMPT_CHARS || '200', 10);
const SHARED_PREFIX = 'You are a helpful assistant operating under strict latency budgets. ';

const ttft = new Trend('infer_lab_ttft_ms', true);
const e2e = new Trend('infer_lab_e2e_ms', true);
const tpot = new Trend('infer_lab_tpot_ms', true);
const outputTokens = new Counter('infer_lab_output_tokens');
const errorRate = new Rate('infer_lab_errors');

export const options = {
  scenarios: {
    staged_ramp: {
      executor: 'ramping-vus',
      startVUs: 1,
      stages: [
        { duration: '15s', target: 1 },   // baseline: one sequence, no batching
        { duration: '30s', target: 4 },   // batching starts to pay off
        { duration: '30s', target: 16 },  // throughput should scale here
        { duration: '30s', target: 32 },  // KV pressure becomes visible
        { duration: '20s', target: 1 },   // recovery: does p99 come back down?
      ],
      gracefulRampDown: '10s',
    },
  },
  thresholds: {
    // These are SLOs, not hopes: a failing threshold fails the k6 run.
    'http_req_failed': ['rate<0.01'],
    'infer_lab_e2e_ms': ['p(95)<10000', 'p(99)<20000'],
    'infer_lab_errors': ['rate<0.01'],
  },
  summaryTrendStats: ['min', 'med', 'avg', 'p(90)', 'p(95)', 'p(99)', 'max'],
};

function makePrompt() {
  // A shared prefix on most requests exercises RadixAttention; the random tail
  // prevents the whole prompt from being a cache hit, which would be unrealistic.
  const unique = Math.random().toString(36).slice(2);
  const filler = 'x'.repeat(Math.max(0, PROMPT_CHARS - SHARED_PREFIX.length - unique.length));
  return `${SHARED_PREFIX}${unique} ${filler}`;
}

export function setup() {
  const probe = http.get(`${BASE_URL}/health`, { timeout: '10s' });
  if (probe.status !== 200) {
    throw new Error(`server not healthy at ${BASE_URL} (status ${probe.status})`);
  }
  return { startedAt: Date.now() };
}

export default function () {
  const payload = JSON.stringify({
    prompt: makePrompt(),
    max_tokens: MAX_TOKENS,
    temperature: 0.0,
    ignore_eos: true,
  });

  const response = http.post(`${BASE_URL}/generate`, payload, {
    headers: { 'Content-Type': 'application/json' },
    timeout: '120s',
    tags: { endpoint: 'generate' },
  });

  const ok = check(response, {
    'status is 200': (r) => r.status === 200,
    'body has tokens': (r) => {
      try {
        return JSON.parse(r.body).output_tokens > 0;
      } catch (e) {
        return false;
      }
    },
  });

  errorRate.add(!ok);

  if (ok) {
    const body = JSON.parse(response.body);
    outputTokens.add(body.output_tokens);
    // The server reports its own internal timings, which excludes network and
    // client overhead -- comparing them against k6's wall time isolates where
    // latency is actually spent.
    if (body.metrics) {
      if (body.metrics.ttft_ms >= 0) ttft.add(body.metrics.ttft_ms);
      if (body.metrics.e2e_ms >= 0) e2e.add(body.metrics.e2e_ms);
      if (body.metrics.tpot_ms >= 0) tpot.add(body.metrics.tpot_ms);
    }
  }

  sleep(0.1);
}

export function teardown() {
  const stats = http.get(`${BASE_URL}/stats`, { timeout: '10s' });
  if (stats.status === 200) {
    const engine = JSON.parse(stats.body).engine;
    console.log(`scheduler: ${JSON.stringify(engine.scheduler)}`);
    console.log(`kv cache : ${JSON.stringify(engine.kv_cache)}`);
  }
}
