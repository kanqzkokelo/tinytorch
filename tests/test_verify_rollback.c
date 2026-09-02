// Verify-speculative rollback test (Fix2): confirms that rolling back
// rejected tail tokens makes the next eager step bit-exact.
// Build: nvcc -O3 -gencode arch=compute_86,code=sm_86 -I../include -Isrc -o build/test_verify_rollback tests/test_verify_rollback.c src/loader_gguf.c src/arch_registry.c src/dequant_ref.c src/tokenizer_bpe.c kernels/gemv_q4_cuda.cu kernels/gemv_typed.cu kernels/qwen2_cuda.cu -L$HOME/mmcuda/lib -lcudart
// Run: ./build/test_verify_rollback
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>
#include "loader_gguf.h"
#include "qwen2_engine.h"

#define MAX_CTX 1024
#define PREFILL_N 8

#define CUDA_OK(x) do { cudaError_t e_=(x); if(e_!=cudaSuccess){fprintf(stderr,"CUDA %s\n",cudaGetErrorString(e_)); exit(1);} }while(0)

static Qwen2Engine *make_prefilled(const TTConfig *cfg, GGUFModel *m, const int *prompt, int n){
    Qwen2Engine *e = qwen2_engine_create(cfg, m);
    if(!e){fprintf(stderr,"create fail\n"); exit(1);}
    if(qwen2_engine_prefill(e, prompt, n)){fprintf(stderr,"prefill fail\n"); exit(1);}
    return e;
}

int main(void){
    const char *model_path = getenv("TT_MODEL") ? getenv("TT_MODEL") : "data/models/qwen2.5-0.5b-instruct-q4_0.gguf";
    fprintf(stderr,"[rollback-test] model %s\n", model_path);
    GGUFModel *model = gguf_load(model_path);
    if(!model){fprintf(stderr,"gguf_load fail\n"); return 1;}
    TTConfig cfg = tt_config_from_gguf(model, MAX_CTX);
    GGUFTensor *tembd = gguf_get_tensor(model, "token_embd.weight");
    if(!tembd){fprintf(stderr,"embd missing\n"); return 1;}
    cfg.vocab = (int)tembd->shape[tembd->ndim-1];
    int prompt[PREFILL_N];
    for(int i=0;i<PREFILL_N;i++) prompt[i]=100+i;
    const long vocab_f = (long)cfg.vocab;
    // Golden eager: prefill + token 1 + token 5
    Qwen2Engine *eA = make_prefilled(&cfg, model, prompt, PREFILL_N);
    int pos0 = qwen2_engine_pos(eA);
    float *logitsA = (float*)malloc(vocab_f * sizeof(float));
    float *logitsA2 = (float*)malloc(vocab_f * sizeof(float));
    if(qwen2_engine_step_logits(eA, 1, logitsA)){fprintf(stderr,"A step1 fail\n"); return 1;}
    int posA1 = qwen2_engine_pos(eA);
    if(qwen2_engine_step_logits(eA, 5, logitsA2)){fprintf(stderr,"A step5 fail\n"); return 1;}
    int posA2 = qwen2_engine_pos(eA);
    fprintf(stderr,"[rollback-test] eager: pos %d -> %d -> %d\n", pos0, posA1, posA2);

    // Verify path: prefill + verify [1,2,3,4] (4 candidates) -> rollback to pos0+1, then step 5
    Qwen2Engine *eB = make_prefilled(&cfg, model, prompt, PREFILL_N);
    int posB0 = qwen2_engine_pos(eB);
    int candidates[4] = {1,2,3,4};
    float *d_logits=NULL;
    CUDA_OK(cudaMalloc(&d_logits, 4*vocab_f*sizeof(float)));
    int vrc = qwen2_engine_verify_speculative(eB, candidates, 4, d_logits);
    if(vrc){fprintf(stderr,"verify fail %d\n", vrc); return 1;}
    int posB1 = qwen2_engine_pos(eB);
    fprintf(stderr,"[rollback-test] verify: pos %d -> %d (delta %d, expected 4)\n", posB0, posB1, posB1-posB0);
    if(posB1 - posB0 != 4){fprintf(stderr,"FAIL pos delta\n"); return 1;}
    // Simulate reject of last 3: rollback to pos0+1
    int target = posB0 + 1;
    qwen2_engine_rollback(eB, target);
    int posB1r = qwen2_engine_pos(eB);
    fprintf(stderr,"[rollback-test] rollback to %d -> pos %d\n", target, posB1r);
    if(posB1r != target){fprintf(stderr,"FAIL rollback pos\n"); return 1;}
    float *logitsB2 = (float*)malloc(vocab_f * sizeof(float));
    if(qwen2_engine_step_logits(eB, 5, logitsB2)){fprintf(stderr,"B step5 fail\n"); return 1;}
    int posB2 = qwen2_engine_pos(eB);
    fprintf(stderr,"[rollback-test] B after rollback+step5 pos %d\n", posB2);
    // Compare logitsB2 vs logitsA2 : must be bit-exact
    int mism=0; double max_abs=0;
    for(long i=0;i<vocab_f;i++){
        float a=logitsA2[i], b=logitsB2[i];
        double d=fabs((double)a-(double)b);
        if(d>max_abs) max_abs=d;
        union{float f; unsigned int u;} ua={a}, ub={b};
        if(ua.u != ub.u) mism++;
    }
    fprintf(stderr,"[rollback-test] max_abs %.3e mismatched %d/%ld\n", max_abs, mism, vocab_f);
    cudaFree(d_logits);
    free(logitsA); free(logitsA2); free(logitsB2);
    qwen2_engine_free(eA); qwen2_engine_free(eB);
    gguf_free(model);
    if(mism!=0){
        fprintf(stderr,"[rollback-test] FAIL: after rollback not bit-exact (%d mism, max_abs %.3e)\n", mism, max_abs);
        return 1;
    }
    fprintf(stderr,"[rollback-test] PASS: rollback restores bit-exact eager path\n");
    return 0;
}
