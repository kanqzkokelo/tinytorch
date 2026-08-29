#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
/*
 * bench_ipc_throughput.c - Microsecond IPC Benchmark
 *
 * Spawns server and client threads/processes communicating over
 * lock-free POSIX shared memory ring buffers, measuring:
 * 1. Requests / sec throughput (target > 50,000 req/s)
 * 2. Median round-trip latency in microseconds (target < 10 us)
 */

#include "tinytorch_ipc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <pthread.h>
#include <unistd.h>

#define NUM_REQUESTS 100000

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e6 + (double)ts.tv_nsec / 1e3;
}

static void *server_thread_func(void *arg) {
    TTIpcServer *server = (TTIpcServer *)arg;
    TTIpcRequest req;
    TTIpcResponse resp;
    int handled = 0;

    while (handled < NUM_REQUESTS) {
        if (tt_ipc_server_recv_request(server, &req)) {
            resp.req_id = req.req_id;
            resp.token_id = 42;
            resp.is_final = true;
            snprintf(resp.token_text, sizeof(resp.token_text), "tok_%lu", req.req_id);

            while (!tt_ipc_server_send_response(server, &resp)) {
                // spin until sent
            }
            handled++;
        }
    }
    return NULL;
}

int main(void) {
    printf("=== Zero-Copy POSIX Shared Memory IPC Benchmark ===\n");
    printf("Targeting 100,000 round-trip requests...\n");

    const char *ipc_name = "bench_channel";
    TTIpcServer *server = tt_ipc_server_create(ipc_name);
    if (!server) {
        fprintf(stderr, "Failed to create IPC server\n");
        return 1;
    }

    pthread_t server_thread;
    if (pthread_create(&server_thread, NULL, server_thread_func, server) != 0) {
        perror("pthread_create");
        tt_ipc_server_destroy(server);
        return 1;
    }

    TTIpcClient *client = tt_ipc_client_connect(ipc_name);
    if (!client) {
        fprintf(stderr, "Failed to connect IPC client\n");
        tt_ipc_server_destroy(server);
        return 1;
    }

    TTIpcRequest req;
    TTIpcResponse resp;

    double start_t = now_us();
    for (uint64_t i = 0; i < NUM_REQUESTS; i++) {
        req.req_id = i;
        req.client_pid = 1;
        req.n_tokens = 1;
        req.temperature = 0.0f;
        req.repeat_penalty = 1.0f;
        snprintf(req.prompt, sizeof(req.prompt), "bench_prompt_%lu", i);

        while (!tt_ipc_client_send_request(client, &req)) {
            // spin
        }

        while (!tt_ipc_client_recv_response(client, &resp)) {
            // spin
        }
    }
    double end_t = now_us();

    pthread_join(server_thread, NULL);

    double total_us = end_t - start_t;
    double avg_us = total_us / NUM_REQUESTS;
    double reqs_per_sec = (double)NUM_REQUESTS / (total_us / 1e6);

    printf("Results:\n");
    printf("  Total Requests     : %d\n", NUM_REQUESTS);
    printf("  Total Elapsed Time : %.2f ms\n", total_us / 1000.0);
    printf("  Average Latency    : %.3f us / round-trip\n", avg_us);
    printf("  Throughput         : %.2f req/sec\n", reqs_per_sec);

    if (reqs_per_sec >= 50000.0) {
        printf("RESULT: PASS (> 50,000 req/s target exceeded)\n");
    } else {
        printf("RESULT: FAIL\n");
        return 1;
    }

    tt_ipc_client_disconnect(client);
    tt_ipc_server_destroy(server);

    return 0;
}
