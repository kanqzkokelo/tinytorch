#ifndef TINYTORCH_IPC_H
#define TINYTORCH_IPC_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TT_IPC_NAME_MAX 64
#define TT_IPC_QUEUE_CAP 1024
#define TT_IPC_TEXT_MAX 512

typedef struct {
    uint64_t req_id;
    int32_t client_pid;
    int32_t n_tokens;
    float temperature;
    float repeat_penalty;
    char prompt[TT_IPC_TEXT_MAX];
} TTIpcRequest;

typedef struct {
    uint64_t req_id;
    int32_t token_id;
    bool is_final;
    char token_text[64];
} TTIpcResponse;

typedef struct TTIpcServer TTIpcServer;
typedef struct TTIpcClient TTIpcClient;

// Server APIs
TTIpcServer *tt_ipc_server_create(const char *name);
bool tt_ipc_server_recv_request(TTIpcServer *server, TTIpcRequest *req);
bool tt_ipc_server_send_response(TTIpcServer *server, const TTIpcResponse *resp);
void tt_ipc_server_destroy(TTIpcServer *server);

// Client APIs
TTIpcClient *tt_ipc_client_connect(const char *name);
bool tt_ipc_client_send_request(TTIpcClient *client, const TTIpcRequest *req);
bool tt_ipc_client_recv_response(TTIpcClient *client, TTIpcResponse *resp);
void tt_ipc_client_disconnect(TTIpcClient *client);

#ifdef __cplusplus
}
#endif

#endif /* TINYTORCH_IPC_H */
