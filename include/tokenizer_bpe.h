#ifndef TOKENIZER_BPE_H
#define TOKENIZER_BPE_H

#include "loader_gguf.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int vocab_size;
    char **tokens;        // Flat array of string pointers
    int *token_lens;      // Array of token string lengths
    float *scores;        // Token scores for BPE merge ranking
    int bos_id;
    int eos_id;
} BPETokenizer;

BPETokenizer *bpe_tokenizer_init(const GGUFModel *model);
const char *bpe_decode_token(const BPETokenizer *tok, int token_id, int *out_len);
int bpe_encode(const BPETokenizer *tok, const char *text, int *out_tokens, int max_tokens);
void bpe_tokenizer_free(BPETokenizer *tok);

#ifdef __cplusplus
}
#endif

#endif // TOKENIZER_BPE_H
