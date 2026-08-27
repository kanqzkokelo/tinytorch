// Find f16 tensors in a GGUF. Usage: build/probe_f16 <path>
#include <stdio.h>
#include "loader_gguf.h"

int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s path.gguf\n", argv[0]); return 1; }
    GGUFModel *m = gguf_load(argv[1]);
    if (!m) { fprintf(stderr, "load fail\n"); return 1; }
    int f16 = 0, bf16 = 0, f32 = 0;
    for (int i = 0; i < m->tensor_count; i++) {
        GGUFTensor *t = &m->tensors[i];
        if ((int)t->type == 1) { f16++; if (f16 <= 5) printf("F16  %s shape=[%ld,%ld]\n", t->name, (long)t->shape[0], (long)t->shape[1]); }
        if ((int)t->type == 30) { bf16++; if (bf16 <= 5) printf("BF16 %s shape=[%ld,%ld]\n", t->name, (long)t->shape[0], (long)t->shape[1]); }
        if ((int)t->type == 0) { f32++; }
    }
    printf("F16=%d BF16=%d F32=%d total=%d\n", f16, bf16, f32, m->tensor_count);
    gguf_free(m);
    return 0;
}
