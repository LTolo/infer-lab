// infer-lab enterprise client (C#, .NET 8).
//
// This is the client an enterprise integration would actually look like, which
// is a different problem from the raw latency probe in clients/c:
//
//   * async/await throughout -- no thread is blocked while the GPU works
//   * bounded concurrency via SemaphoreSlim rather than unbounded task spam
//   * retry with exponential backoff + jitter on 429/503/timeout
//   * a distinction between retryable and terminal failures
//   * cancellation support so a shutdown does not strand in-flight work
//
// Build: dotnet build
// Run:   dotnet run -- [baseUrl] [totalRequests] [concurrency] [maxTokens]

using System.Diagnostics;
using System.Net;
using System.Net.Http.Json;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace InferLab.Client;

public sealed record GenerateRequest(
    [property: JsonPropertyName("prompt")] string Prompt,
    [property: JsonPropertyName("max_tokens")] int MaxTokens,
    [property: JsonPropertyName("temperature")] double Temperature = 0.0);

public sealed record GenerateResponse(
    [property: JsonPropertyName("request_id")] string RequestId,
    [property: JsonPropertyName("text")] string Text,
    [property: JsonPropertyName("output_tokens")] int OutputTokens,
    [property: JsonPropertyName("finish_reason")] string? FinishReason);

/// <summary>Transient-fault-handling client for the infer-lab HTTP API.</summary>
public sealed class InferLabClient : IDisposable
{
    private static readonly HttpStatusCode[] RetryableStatuses =
    {
        HttpStatusCode.TooManyRequests,
        HttpStatusCode.ServiceUnavailable,
        HttpStatusCode.GatewayTimeout,
        HttpStatusCode.BadGateway,
    };

    private readonly HttpClient _http;
    private readonly int _maxAttempts;
    private readonly Random _jitter = new();

    public InferLabClient(string baseUrl, int maxAttempts = 4, TimeSpan? timeout = null)
    {
        _http = new HttpClient
        {
            BaseAddress = new Uri(baseUrl),
            Timeout = timeout ?? TimeSpan.FromSeconds(120),
        };
        _maxAttempts = maxAttempts;
    }

    public async Task<bool> WaitForReadyAsync(TimeSpan timeout, CancellationToken ct = default)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline && !ct.IsCancellationRequested)
        {
            try
            {
                using var response = await _http.GetAsync("/ready", ct).ConfigureAwait(false);
                if (response.IsSuccessStatusCode) return true;
            }
            catch (Exception) when (!ct.IsCancellationRequested)
            {
                // server not up yet -- keep polling
            }
            await Task.Delay(500, ct).ConfigureAwait(false);
        }
        return false;
    }

    /// <summary>Sends one generation request, retrying only on transient faults.</summary>
    public async Task<GenerateResponse> GenerateAsync(
        GenerateRequest request, CancellationToken ct = default)
    {
        Exception? last = null;

        for (var attempt = 1; attempt <= _maxAttempts; attempt++)
        {
            try
            {
                using var response = await _http
                    .PostAsJsonAsync("/generate", request, ct)
                    .ConfigureAwait(false);

                if (response.IsSuccessStatusCode)
                {
                    var payload = await response.Content
                        .ReadFromJsonAsync<GenerateResponse>(cancellationToken: ct)
                        .ConfigureAwait(false);
                    return payload ?? throw new InvalidOperationException("empty response body");
                }

                // 4xx other than 429 is our fault -- retrying cannot help.
                if (!RetryableStatuses.Contains(response.StatusCode))
                {
                    var body = await response.Content.ReadAsStringAsync(ct).ConfigureAwait(false);
                    throw new HttpRequestException(
                        $"terminal failure {(int)response.StatusCode}: {body}");
                }

                last = new HttpRequestException($"retryable status {(int)response.StatusCode}");
            }
            catch (TaskCanceledException ex) when (!ct.IsCancellationRequested)
            {
                last = ex; // request timeout, not caller cancellation
            }
            catch (HttpRequestException ex) when (!ex.Message.StartsWith("terminal"))
            {
                last = ex;
            }

            if (attempt < _maxAttempts)
            {
                // Exponential backoff with jitter: without jitter, every client
                // retries in lockstep and re-creates the overload it is backing off from.
                var backoff = TimeSpan.FromMilliseconds(
                    Math.Pow(2, attempt) * 100 + _jitter.Next(0, 100));
                await Task.Delay(backoff, ct).ConfigureAwait(false);
            }
        }

        throw new HttpRequestException(
            $"request failed after {_maxAttempts} attempts", last);
    }

    public void Dispose() => _http.Dispose();
}

public static class Program
{
    public static async Task<int> Main(string[] args)
    {
        var baseUrl = args.Length > 0 ? args[0] : "http://127.0.0.1:8000";
        var total = args.Length > 1 ? int.Parse(args[1]) : 20;
        var concurrency = args.Length > 2 ? int.Parse(args[2]) : 4;
        var maxTokens = args.Length > 3 ? int.Parse(args[3]) : 32;

        Console.WriteLine($"infer-lab C# client -> {baseUrl}");
        Console.WriteLine($"  requests={total} concurrency={concurrency} max_tokens={maxTokens}\n");

        using var cts = new CancellationTokenSource();
        Console.CancelKeyPress += (_, e) => { e.Cancel = true; cts.Cancel(); };

        using var client = new InferLabClient(baseUrl);
        if (!await client.WaitForReadyAsync(TimeSpan.FromSeconds(30), cts.Token))
        {
            Console.Error.WriteLine("server did not become ready");
            return 1;
        }

        // Bounded concurrency: the server has a finite KV cache, so an unbounded
        // fan-out just converts throughput into queueing delay.
        using var gate = new SemaphoreSlim(concurrency);
        var latencies = new System.Collections.Concurrent.ConcurrentBag<double>();
        var failures = 0;
        var tokens = 0;

        var stopwatch = Stopwatch.StartNew();
        var tasks = Enumerable.Range(0, total).Select(async i =>
        {
            await gate.WaitAsync(cts.Token).ConfigureAwait(false);
            try
            {
                var sw = Stopwatch.StartNew();
                var response = await client.GenerateAsync(
                    new GenerateRequest($"csharp request {i}", maxTokens), cts.Token);
                sw.Stop();
                latencies.Add(sw.Elapsed.TotalMilliseconds);
                Interlocked.Add(ref tokens, response.OutputTokens);
            }
            catch (Exception ex)
            {
                Interlocked.Increment(ref failures);
                Console.Error.WriteLine($"  request {i} failed: {ex.Message}");
            }
            finally
            {
                gate.Release();
            }
        });

        await Task.WhenAll(tasks).ConfigureAwait(false);
        stopwatch.Stop();

        var sorted = latencies.OrderBy(x => x).ToArray();
        static double Percentile(double[] values, double q)
        {
            if (values.Length == 0) return 0;
            var rank = Math.Clamp((int)Math.Ceiling(q / 100.0 * values.Length), 1, values.Length);
            return values[rank - 1];
        }

        var seconds = stopwatch.Elapsed.TotalSeconds;
        Console.WriteLine("\nresults");
        Console.WriteLine($"  succeeded         : {sorted.Length} / {total} ({failures} failed)");
        Console.WriteLine($"  wall time         : {seconds:F2} s");
        Console.WriteLine($"  request throughput: {sorted.Length / seconds:F2} req/s");
        Console.WriteLine($"  token throughput  : {tokens / seconds:F1} tok/s");
        Console.WriteLine($"  latency p50       : {Percentile(sorted, 50):F2} ms");
        Console.WriteLine($"  latency p90       : {Percentile(sorted, 90):F2} ms");
        Console.WriteLine($"  latency p99       : {Percentile(sorted, 99):F2} ms");

        return failures == 0 ? 0 : 1;
    }
}
