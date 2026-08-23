#ifndef LOADER_GGUF_H
#define LOADER_GGUF_H

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

// GGUF tensor data types
typedef enum {
    GGUF_TYPE_F32  = 0,
    GGUF_TYPE_F16  = 1,
    GGUF_TYPE_Q4_0 = 2,
    GGUF_TYPE_Q4_1 = 3,
    GGUF_TYPE_Q8_0 = 8,
} GGUFType;

// Block layout for Q4_0: 32 FP16 values stored in 16 bytes of nibbles + 1 FP16 scale d
typedef struct {
    uint16_t d;        // FP16 scale factor (2 bytes)
    uint8_t qs[16];    // 32 nibbles (16 bytes)
} BlockQ4_0;

typedef struct {
    char name[128];
    GGUFType type;
    int ndim;
    int64_t shape[4];
    size_t size_bytes;
    uint64_t offset;
    void *data;       // Pointer to host or mmap data
    void *gpu_data;   // Pointer to device CUDA memory
} GGUFTensor;

typedef struct {
    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int max_seq_len;
    float rms_norm_eps;
    float rope_freq_base;
    int tensor_count;
    GGUFTensor *tensors;
    void *mmap_addr;
    size_t mmap_size;
} GGUFModel;

GGUFModel *gguf_load(const char *filepath);
GGUFTensor *gguf_get_tensor(GGUFModel *model, const char *name);
void gguf_free(GGUFModel *model);

#ifdef __cplusplus
}
#endif

#endif // LOADER_GGUF_H
