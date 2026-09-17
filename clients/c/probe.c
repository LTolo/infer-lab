/*
 * infer-lab raw latency probe (C).
 *
 * Why a C client at all: every higher-level HTTP client adds its own buffering,
 * connection pooling and GC pauses, so when you measure "server latency" with
 * Python you are partly measuring Python. This probe speaks HTTP/1.1 directly
 * over a socket with TCP_NODELAY, so what it reports is as close to the wire
 * time as you can get from user space.
 *
 * Build:
 *   Linux/macOS : cc -O2 -std=c11 probe.c -o probe
 *   Windows/MSVC: cl /nologo /O2 probe.c ws2_32.lib
 *
 * Usage:
 *   ./probe [host] [port] [iterations] [max_tokens]
 */

/* POSIX feature macros must precede every include: with a strict -std=c11
   the libc headers otherwise hide clock_gettime() and getaddrinfo(). */
#if !defined(_WIN32)
#  define _POSIX_C_SOURCE 200809L
#  define _DEFAULT_SOURCE
#endif

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   typedef SOCKET sock_t;
#  define CLOSESOCK closesocket
#  define SOCK_INVALID INVALID_SOCKET
#else
#  include <arpa/inet.h>
#  include <netdb.h>
#  include <netinet/in.h>
#  include <netinet/tcp.h>
#  include <sys/socket.h>
#  include <time.h>
#  include <unistd.h>
   typedef int sock_t;
#  define CLOSESOCK close
#  define SOCK_INVALID (-1)
#endif

#define RECV_BUF 65536

static double now_ms(void) {
#ifdef _WIN32
    LARGE_INTEGER freq, counter;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart * 1000.0 / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000.0 + ts.tv_nsec / 1.0e6;
#endif
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* nearest-rank percentile, same definition as the Python harness */
static double percentile(double *sorted, int n, double q) {
    if (n <= 0) return 0.0;
    int rank = (int)((q / 100.0) * n + 0.9999);
    if (rank < 1) rank = 1;
    if (rank > n) rank = n;
    return sorted[rank - 1];
}

static sock_t connect_to(const char *host, const char *port) {
    struct addrinfo hints, *res = NULL;
    memset(&hints, 0, sizeof(hints));
    hints.ai_family = AF_INET;
    hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host, port, &hints, &res) != 0) return SOCK_INVALID;

    sock_t fd = socket(res->ai_family, res->ai_socktype, res->ai_protocol);
    if (fd == SOCK_INVALID) { freeaddrinfo(res); return SOCK_INVALID; }

    if (connect(fd, res->ai_addr, (int)res->ai_addrlen) != 0) {
        CLOSESOCK(fd); freeaddrinfo(res); return SOCK_INVALID;
    }
    freeaddrinfo(res);

    /* Nagle would batch our small request with the previous ACK and add up to
       40 ms of pure measurement artefact. Disable it. */
    int one = 1;
    setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, (const char *)&one, sizeof(one));
    return fd;
}

/* Reads a full HTTP/1.1 response: headers, then Content-Length bytes of body. */
static int read_response(sock_t fd, char *buf, int cap) {
    int total = 0;
    int header_end = -1, content_length = -1;

    while (total < cap - 1) {
        int n = (int)recv(fd, buf + total, cap - 1 - total, 0);
        if (n <= 0) break;
        total += n;
        buf[total] = '\0';

        if (header_end < 0) {
            char *p = strstr(buf, "\r\n\r\n");
            if (p) {
                header_end = (int)(p - buf) + 4;
                char *cl = strstr(buf, "Content-Length:");
                if (!cl) cl = strstr(buf, "content-length:");
                if (cl) content_length = atoi(cl + 15);
            }
        }
        if (header_end >= 0 && content_length >= 0 &&
            total >= header_end + content_length) {
            break;
        }
    }
    return total;
}

int main(int argc, char **argv) {
    const char *host = argc > 1 ? argv[1] : "127.0.0.1";
    const char *port = argc > 2 ? argv[2] : "8000";
    int iterations   = argc > 3 ? atoi(argv[3]) : 50;
    int max_tokens   = argc > 4 ? atoi(argv[4]) : 16;

    if (iterations <= 0) iterations = 50;

#ifdef _WIN32
    WSADATA wsa;
    if (WSAStartup(MAKEWORD(2, 2), &wsa) != 0) {
        fprintf(stderr, "WSAStartup failed\n");
        return 1;
    }
#endif

    char body[256];
    int body_len = snprintf(body, sizeof(body),
        "{\"prompt\":\"latency probe\",\"max_tokens\":%d,\"temperature\":0.0}",
        max_tokens);

    char request[1024];
    int request_len = snprintf(request, sizeof(request),
        "POST /generate HTTP/1.1\r\n"
        "Host: %s:%s\r\n"
        "Content-Type: application/json\r\n"
        "Content-Length: %d\r\n"
        "Connection: close\r\n"
        "\r\n%s",
        host, port, body_len, body);

    double *samples = (double *)malloc(sizeof(double) * (size_t)iterations);
    if (!samples) { fprintf(stderr, "out of memory\n"); return 1; }
    char *buf = (char *)malloc(RECV_BUF);
    if (!buf) { free(samples); fprintf(stderr, "out of memory\n"); return 1; }

    printf("infer-lab C probe -> %s:%s  (%d iterations, max_tokens=%d)\n",
           host, port, iterations, max_tokens);

    int ok_count = 0;
    for (int i = 0; i < iterations; i++) {
        double t0 = now_ms();
        sock_t fd = connect_to(host, port);
        if (fd == SOCK_INVALID) {
            fprintf(stderr, "  iteration %d: connect failed\n", i);
            continue;
        }
        if (send(fd, request, request_len, 0) != request_len) {
            fprintf(stderr, "  iteration %d: short send\n", i);
            CLOSESOCK(fd);
            continue;
        }
        int n = read_response(fd, buf, RECV_BUF);
        CLOSESOCK(fd);
        if (n <= 0) {
            fprintf(stderr, "  iteration %d: empty response\n", i);
            continue;
        }
        if (strstr(buf, "200 OK") == NULL) {
            fprintf(stderr, "  iteration %d: non-200 response\n", i);
            continue;
        }
        samples[ok_count++] = now_ms() - t0;
    }

    if (ok_count == 0) {
        fprintf(stderr, "no successful requests -- is the server running?\n");
        free(samples); free(buf);
#ifdef _WIN32
        WSACleanup();
#endif
        return 1;
    }

    qsort(samples, (size_t)ok_count, sizeof(double), cmp_double);
    double sum = 0.0;
    for (int i = 0; i < ok_count; i++) sum += samples[i];

    printf("\n  successful : %d / %d\n", ok_count, iterations);
    printf("  mean       : %8.3f ms\n", sum / ok_count);
    printf("  min        : %8.3f ms\n", samples[0]);
    printf("  p50        : %8.3f ms\n", percentile(samples, ok_count, 50));
    printf("  p90        : %8.3f ms\n", percentile(samples, ok_count, 90));
    printf("  p99        : %8.3f ms\n", percentile(samples, ok_count, 99));
    printf("  max        : %8.3f ms\n", samples[ok_count - 1]);

    free(samples);
    free(buf);
#ifdef _WIN32
    WSACleanup();
#endif
    return 0;
}
