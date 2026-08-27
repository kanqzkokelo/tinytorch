#ifndef NGRAM_LOOKUP_H
#define NGRAM_LOOKUP_H

#include <stdint.h>
#include <stddef.h>

#define MAX_DRAFT_K 4

typedef struct {
    int window_size; // 2 or 3
    int max_draft;   // K candidate tokens to return
} NgramConfig;

// Search sequence history tokens[0..N-1] for a match of the last `window_size` tokens.
// If found, copy up to `max_draft` continuation tokens into `out_draft` and return count.
// Returns 0 if no match found.
int ngram_lookup_draft(const int *tokens, int N, int window_size, int max_draft, int *out_draft);

#endif // NGRAM_LOOKUP_H
