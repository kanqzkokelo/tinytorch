#include "ngram_lookup.h"
#include <string.h>

int ngram_lookup_draft(const int *tokens, int N, int window_size, int max_draft, int *out_draft) {
    if (N <= window_size + 1 || window_size <= 0 || max_draft <= 0) return 0;

    const int *target = tokens + N - window_size;

    // Search backward from N - window_size - 1 down to 0 for a matching window
    for (int i = N - window_size - 1; i >= 0; i--) {
        int match = 1;
        for (int j = 0; j < window_size; j++) {
            if (tokens[i + j] != target[j]) {
                match = 0;
                break;
            }
        }
        if (match) {
            // Found match starting at i! Continuation starts at i + window_size
            int start_cont = i + window_size;
            int available = (N - window_size) - start_cont; // don't overlap with the current target window itself
            if (available <= 0) continue;

            int count = available < max_draft ? available : max_draft;
            for (int k = 0; k < count; k++) {
                out_draft[k] = tokens[start_cont + k];
            }
            return count;
        }
    }
    return 0;
}
