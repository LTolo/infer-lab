/*
 * infer-lab concurrent load-test client (Java).
 *
 * Purpose: make the continuous-batching effect *visible*. A single-threaded
 * client measures one sequence decoding alone. This client fires N concurrent
 * requests, so the server's scheduler batches them into shared forward passes.
 *
 * The signature you are looking for in the output: as concurrency rises,
 * per-request latency grows only slowly while aggregate throughput grows
 * almost linearly. That gap is exactly the weight-load amortisation that
 * continuous batching buys. With static batching the p99 would explode instead.
 *
 * Build: javac LoadTestClient.java
 * Run:   java LoadTestClient [baseUrl] [concurrency] [requestsPerThread] [maxTokens]
 */

import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;

public class LoadTestClient {

    private static final AtomicInteger FAILURES = new AtomicInteger();

    private record Sample(double latencyMs, int outputTokens) {}

    public static void main(String[] args) throws Exception {
        String baseUrl = args.length > 0 ? args[0] : "http://127.0.0.1:8000";
        int concurrency = args.length > 1 ? Integer.parseInt(args[1]) : 8;
        int perThread = args.length > 2 ? Integer.parseInt(args[2]) : 5;
        int maxTokens = args.length > 3 ? Integer.parseInt(args[3]) : 32;

        System.out.printf("infer-lab Java load test -> %s%n", baseUrl);
        System.out.printf("  concurrency=%d requests/thread=%d max_tokens=%d%n%n",
                concurrency, perThread, maxTokens);

        HttpClient client = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .version(HttpClient.Version.HTTP_1_1)
                .build();

        if (!waitForServer(client, baseUrl)) {
            System.err.println("server is not reachable at " + baseUrl);
            System.exit(1);
        }

        // Warm up so JIT compilation is not counted as server latency.
        fire(client, baseUrl, 4);

        ExecutorService pool = Executors.newFixedThreadPool(concurrency);
        List<Callable<List<Sample>>> tasks = new ArrayList<>();
        for (int t = 0; t < concurrency; t++) {
            final int threadId = t;
            tasks.add(() -> {
                List<Sample> local = new ArrayList<>();
                for (int i = 0; i < perThread; i++) {
                    local.add(fire(client, baseUrl, maxTokens));
                }
                return local;
            });
        }

        long startNanos = System.nanoTime();
        List<Future<List<Sample>>> futures = pool.invokeAll(tasks);
        List<Sample> samples = new ArrayList<>();
        for (Future<List<Sample>> future : futures) {
            samples.addAll(future.get());
        }
        double wallSeconds = (System.nanoTime() - startNanos) / 1_000_000_000.0;
        pool.shutdown();
        pool.awaitTermination(30, TimeUnit.SECONDS);

        report(samples, wallSeconds, concurrency);
    }

    private static boolean waitForServer(HttpClient client, String baseUrl) {
        for (int attempt = 0; attempt < 30; attempt++) {
            try {
                HttpRequest request = HttpRequest.newBuilder()
                        .uri(URI.create(baseUrl + "/health"))
                        .timeout(Duration.ofSeconds(3))
                        .GET().build();
                if (client.send(request, HttpResponse.BodyHandlers.ofString())
                        .statusCode() == 200) {
                    return true;
                }
            } catch (Exception ignored) {
                // server not up yet
            }
            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                Thread.currentThread().interrupt();
                return false;
            }
        }
        return false;
    }

    private static Sample fire(HttpClient client, String baseUrl, int maxTokens) {
        String payload = String.format(
                "{\"prompt\":\"java load test\",\"max_tokens\":%d,\"temperature\":0.0}",
                maxTokens);
        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/generate"))
                .timeout(Duration.ofSeconds(120))
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(payload))
                .build();

        long t0 = System.nanoTime();
        try {
            HttpResponse<String> response =
                    client.send(request, HttpResponse.BodyHandlers.ofString());
            double ms = (System.nanoTime() - t0) / 1_000_000.0;
            if (response.statusCode() != 200) {
                FAILURES.incrementAndGet();
                return new Sample(ms, 0);
            }
            return new Sample(ms, extractInt(response.body(), "output_tokens"));
        } catch (Exception e) {
            FAILURES.incrementAndGet();
            return new Sample((System.nanoTime() - t0) / 1_000_000.0, 0);
        }
    }

    /* Minimal JSON field extraction -- avoids pulling in a JSON dependency. */
    private static int extractInt(String json, String field) {
        String key = "\"" + field + "\":";
        int idx = json.indexOf(key);
        if (idx < 0) {
            return 0;
        }
        int start = idx + key.length();
        int end = start;
        while (end < json.length() && (Character.isDigit(json.charAt(end)))) {
            end++;
        }
        return end > start ? Integer.parseInt(json.substring(start, end)) : 0;
    }

    private static double percentile(List<Double> sorted, double q) {
        if (sorted.isEmpty()) {
            return 0.0;
        }
        int rank = (int) Math.ceil(q / 100.0 * sorted.size());
        rank = Math.max(1, Math.min(sorted.size(), rank));
        return sorted.get(rank - 1);
    }

    private static void report(List<Sample> samples, double wallSeconds, int concurrency) {
        List<Double> latencies = new ArrayList<>();
        int totalTokens = 0;
        for (Sample sample : samples) {
            latencies.add(sample.latencyMs());
            totalTokens += sample.outputTokens();
        }
        Collections.sort(latencies);

        double mean = latencies.stream().mapToDouble(Double::doubleValue).average().orElse(0);
        double p50 = percentile(latencies, 50);
        double p99 = percentile(latencies, 99);

        System.out.println("results");
        System.out.printf("  requests          : %d (%d failed)%n",
                samples.size(), FAILURES.get());
        System.out.printf("  wall time         : %.2f s%n", wallSeconds);
        System.out.printf("  request throughput: %.2f req/s%n", samples.size() / wallSeconds);
        System.out.printf("  token throughput  : %.1f tok/s%n", totalTokens / wallSeconds);
        System.out.printf("  latency mean      : %.2f ms%n", mean);
        System.out.printf("  latency p50       : %.2f ms%n", p50);
        System.out.printf("  latency p90       : %.2f ms%n", percentile(latencies, 90));
        System.out.printf("  latency p99       : %.2f ms%n", p99);
        System.out.printf("  tail ratio p99/p50: %.2f%n", p50 > 0 ? p99 / p50 : 0.0);
        System.out.printf("%n  interpretation: with continuous batching, raising concurrency%n");
        System.out.printf("  from 1 to %d should multiply token throughput while latency%n",
                concurrency);
        System.out.printf("  grows far more slowly. A tail ratio above ~5 indicates the%n");
        System.out.printf("  scheduler is queueing rather than batching.%n");
    }
}
