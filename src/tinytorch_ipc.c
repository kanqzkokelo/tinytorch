#define _POSIX_C_SOURCE 200809L
#define _DEFAULT_SOURCE
/*
 * tinytorch_ipc.c - Lock-free POSIX Shared Memory Ring Buffer IPC
 *
 * Implements high-throughput (> 50,000 req/s), microsecond-latency
 * inter-process communication using memory-mapped circular queues with
 * C11 atomic head/tail pointers and memory fences.
 */

#include "tinytorch_ipc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdatomic.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>

typedef struct {
    _Atomic uint32_t head;
    _Atomic uint32_t tail;
    TTIpcRequest entries[TT_IPC_QUEUE_CAP];
} ReqQueue;

typedef struct {
    _Atomic uint32_t head;
    _Atomic uint32_t tail;
    TTIpcResponse entries[TT_IPC_QUEUE_CAP];
} RespQueue;

typedef struct {
    ReqQueue req_q;
    RespQueue resp_q;
    _Atomic uint32_t active_clients;
} ShmBuffer;

struct TTIpcServer {
    char shm_name[TT_IPC_NAME_MAX];
    int shm_fd;
    ShmBuffer *shm;
};

struct TTIpcClient {
    char shm_name[TT_IPC_NAME_MAX];
    int shm_fd;
    ShmBuffer *shm;
};

TTIpcServer *tt_ipc_server_create(const char *name) {
    if (!name || !name[0]) return NULL;
    TTIpcServer *s = (TTIpcServer *)calloc(1, sizeof(TTIpcServer));
    if (!s) return NULL;

    snprintf(s->shm_name, sizeof(s->shm_name), "/tt_ipc_%s", name);
    shm_unlink(s->shm_name); // Clean up stale segments if any

    s->shm_fd = shm_open(s->shm_name, O_CREAT | O_RDWR | O_EXCL, 0666);
    if (s->shm_fd < 0) {
        perror("shm_open create");
        free(s);
        return NULL;
    }

    if (ftruncate(s->shm_fd, sizeof(ShmBuffer)) < 0) {
        perror("ftruncate");
        close(s->shm_fd);
        shm_unlink(s->shm_name);
        free(s);
        return NULL;
    }

    s->shm = (ShmBuffer *)mmap(NULL, sizeof(ShmBuffer), PROT_READ | PROT_WRITE,
                               MAP_SHARED, s->shm_fd, 0);
    if (s->shm == MAP_FAILED) {
        perror("mmap server");
        close(s->shm_fd);
        shm_unlink(s->shm_name);
        free(s);
        return NULL;
    }

    memset(s->shm, 0, sizeof(ShmBuffer));
    atomic_init(&s->shm->req_q.head, 0);
    atomic_init(&s->shm->req_q.tail, 0);
    atomic_init(&s->shm->resp_q.head, 0);
    atomic_init(&s->shm->resp_q.tail, 0);
    atomic_init(&s->shm->active_clients, 0);

    return s;
}

bool tt_ipc_server_recv_request(TTIpcServer *server, TTIpcRequest *req) {
    if (!server || !server->shm || !req) return false;
    ReqQueue *q = &server->shm->req_q;

    uint32_t head = atomic_load_explicit(&q->head, memory_order_relaxed);
    uint32_t tail = atomic_load_explicit(&q->tail, memory_order_acquire);

    if (head == tail) {
        return false; // Empty queue
    }

    *req = q->entries[head % TT_IPC_QUEUE_CAP];
    atomic_store_explicit(&q->head, head + 1, memory_order_release);
    return true;
}

bool tt_ipc_server_send_response(TTIpcServer *server, const TTIpcResponse *resp) {
    if (!server || !server->shm || !resp) return false;
    RespQueue *q = &server->shm->resp_q;

    uint32_t head = atomic_load_explicit(&q->head, memory_order_acquire);
    uint32_t tail = atomic_load_explicit(&q->tail, memory_order_relaxed);

    if (tail - head >= TT_IPC_QUEUE_CAP) {
        return false; // Queue full
    }

    q->entries[tail % TT_IPC_QUEUE_CAP] = *resp;
    atomic_store_explicit(&q->tail, tail + 1, memory_order_release);
    return true;
}

void tt_ipc_server_destroy(TTIpcServer *server) {
    if (!server) return;
    if (server->shm && server->shm != MAP_FAILED) {
        munmap(server->shm, sizeof(ShmBuffer));
    }
    if (server->shm_fd >= 0) {
        close(server->shm_fd);
    }
    shm_unlink(server->shm_name);
    free(server);
}

TTIpcClient *tt_ipc_client_connect(const char *name) {
    if (!name || !name[0]) return NULL;
    TTIpcClient *c = (TTIpcClient *)calloc(1, sizeof(TTIpcClient));
    if (!c) return NULL;

    snprintf(c->shm_name, sizeof(c->shm_name), "/tt_ipc_%s", name);
    c->shm_fd = shm_open(c->shm_name, O_RDWR, 0666);
    if (c->shm_fd < 0) {
        free(c);
        return NULL;
    }

    c->shm = (ShmBuffer *)mmap(NULL, sizeof(ShmBuffer), PROT_READ | PROT_WRITE,
                               MAP_SHARED, c->shm_fd, 0);
    if (c->shm == MAP_FAILED) {
        close(c->shm_fd);
        free(c);
        return NULL;
    }

    atomic_fetch_add_explicit(&c->shm->active_clients, 1, memory_order_relaxed);
    return c;
}

bool tt_ipc_client_send_request(TTIpcClient *client, const TTIpcRequest *req) {
    if (!client || !client->shm || !req) return false;
    ReqQueue *q = &client->shm->req_q;

    uint32_t head = atomic_load_explicit(&q->head, memory_order_acquire);
    uint32_t tail = atomic_load_explicit(&q->tail, memory_order_relaxed);

    if (tail - head >= TT_IPC_QUEUE_CAP) {
        return false; // Queue full
    }

    q->entries[tail % TT_IPC_QUEUE_CAP] = *req;
    atomic_store_explicit(&q->tail, tail + 1, memory_order_release);
    return true;
}

bool tt_ipc_client_recv_response(TTIpcClient *client, TTIpcResponse *resp) {
    if (!client || !client->shm || !resp) return false;
    RespQueue *q = &client->shm->resp_q;

    uint32_t head = atomic_load_explicit(&q->head, memory_order_relaxed);
    uint32_t tail = atomic_load_explicit(&q->tail, memory_order_acquire);

    if (head == tail) {
        return false; // Empty queue
    }

    *resp = q->entries[head % TT_IPC_QUEUE_CAP];
    atomic_store_explicit(&q->head, head + 1, memory_order_release);
    return true;
}

void tt_ipc_client_disconnect(TTIpcClient *client) {
    if (!client) return;
    if (client->shm && client->shm != MAP_FAILED) {
        atomic_fetch_sub_explicit(&client->shm->active_clients, 1, memory_order_relaxed);
        munmap(client->shm, sizeof(ShmBuffer));
    }
    if (client->shm_fd >= 0) {
        close(client->shm_fd);
    }
    free(client);
}
