#include "async_printer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <pthread.h>
#include <stdatomic.h>
#include <time.h>
#include <unistd.h>

#define ASYNC_BUF_SIZE 16384
#define ASYNC_QUEUE_CAP 1024

struct AsyncPrinter {
    const char *queue_str[ASYNC_QUEUE_CAP];
    int queue_len[ASYNC_QUEUE_CAP];
    _Atomic int head;
    _Atomic int tail;
    _Atomic int done;
    pthread_t thread;
};

static void *async_printer_worker(void *arg) {
    AsyncPrinter *ap = (AsyncPrinter *)arg;
    char io_buf[ASYNC_BUF_SIZE];
    int buf_len = 0;

    while (!atomic_load(&ap->done) || atomic_load(&ap->head) != atomic_load(&ap->tail)) {
        int head = atomic_load(&ap->head);
        int tail = atomic_load(&ap->tail);

        if (head != tail) {
            const char *str = ap->queue_str[head];
            int len = ap->queue_len[head];
            atomic_store(&ap->head, (head + 1) % ASYNC_QUEUE_CAP);

            if (str && len > 0) {
                if (buf_len + len < (int)sizeof(io_buf)) {
                    memcpy(io_buf + buf_len, str, len);
                    buf_len += len;
                } else {
                    write(STDOUT_FILENO, io_buf, buf_len);
                    buf_len = 0;
                    memcpy(io_buf + buf_len, str, len);
                    buf_len += len;
                }
            }
        } else {
            if (buf_len > 0) {
                write(STDOUT_FILENO, io_buf, buf_len);
                buf_len = 0;
            }
            struct timespec req = {0, 100000}; // 0.1 ms sleep
            nanosleep(&req, NULL);
        }
    }

    if (buf_len > 0) {
        write(STDOUT_FILENO, io_buf, buf_len);
    }
    return NULL;
}

AsyncPrinter *async_printer_start(void) {
    AsyncPrinter *ap = (AsyncPrinter *)calloc(1, sizeof(AsyncPrinter));
    atomic_store(&ap->head, 0);
    atomic_store(&ap->tail, 0);
    atomic_store(&ap->done, 0);

    pthread_create(&ap->thread, NULL, async_printer_worker, ap);
    return ap;
}

void async_printer_push(AsyncPrinter *ap, const char *str, int len) {
    if (!ap || !str || len <= 0) return;
    int tail = atomic_load(&ap->tail);
    ap->queue_str[tail] = str;
    ap->queue_len[tail] = len;
    atomic_store(&ap->tail, (tail + 1) % ASYNC_QUEUE_CAP);
}

void async_printer_stop_and_flush(AsyncPrinter *ap) {
    if (!ap) return;
    atomic_store(&ap->done, 1);
    pthread_join(ap->thread, NULL);
    free(ap);
}
