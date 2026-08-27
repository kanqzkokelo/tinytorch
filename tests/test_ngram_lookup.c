#include "ngram_lookup.h"
#include <stdio.h>
#include <assert.h>

int main() {
    // Pattern: "The quick brown fox jumps over the lazy dog. The quick brown"
    int tokens[] = {1, 10, 20, 30, 40, 50, 60, 70, 80, 90, 1, 10, 20};
    int N = sizeof(tokens)/sizeof(tokens[0]);
    int draft[4];

    // Window = 2 ("10, 20"), expected continuation: "30, 40, 50"
    int count = ngram_lookup_draft(tokens, N, 2, 3, draft);
    printf("Draft count: %d\n", count);
    assert(count == 3);
    assert(draft[0] == 30);
    assert(draft[1] == 40);
    assert(draft[2] == 50);

    // Edge: N too small (N <= window_size + 1)
    int small[] = {1, 2, 3};
    int c0 = ngram_lookup_draft(small, 3, 2, 2, draft);
    assert(c0 == 0);

    // Edge: no match in history
    int nomatch[] = {1, 2, 3, 4, 5, 99, 98, 97};
    int c1 = ngram_lookup_draft(nomatch, 8, 2, 3, draft);
    assert(c1 == 0);

    // Edge: match at index 0 (early occurrence should still be found)
    // window = 2 ("9, 8"), target at [6..7], match at i=0, continuation [7,0,0,0]
    int early[] = {9, 8, 7, 0, 0, 0, 9, 8};
    int c2 = ngram_lookup_draft(early, 8, 2, 4, draft);
    assert(c2 == 4);
    assert(draft[0] == 7);
    assert(draft[1] == 0);
    assert(draft[2] == 0);
    assert(draft[3] == 0);

    // Edge: max_draft caps available continuation
    int cap[] = {1, 2, 3, 4, 5, 1, 2};
    int c3 = ngram_lookup_draft(cap, 7, 2, 2, draft);
    assert(c3 == 2);
    assert(draft[0] == 3);
    assert(draft[1] == 4);

    // Edge: window_size = 3, target [1,2,3] at end, match at i=0,
    // continuation [4,5,9,9], capped at max_draft=4 -> draft = [4,5,9,9]
    int win3[] = {1, 2, 3, 4, 5, 9, 9, 1, 2, 3};
    int c4 = ngram_lookup_draft(win3, 10, 3, 4, draft);
    assert(c4 == 4);
    assert(draft[0] == 4);
    assert(draft[1] == 5);
    assert(draft[2] == 9);
    assert(draft[3] == 9);

    // Edge: invalid params
    int c5 = ngram_lookup_draft(tokens, N, 0, 3, draft);
    assert(c5 == 0);
    int c6 = ngram_lookup_draft(tokens, N, 2, 0, draft);
    assert(c6 == 0);

    // Edge: immediate neighbor match but continuation overlaps target window
    // tokens "1, 2, 3, 1, 2, 3" with window=3, target=[1,2,3], match at i=0,
    // start_cont=3, N-window_size=3, available=0 -> skip, no other match -> 0
    int overlap[] = {1, 2, 3, 1, 2, 3};
    int c7 = ngram_lookup_draft(overlap, 6, 3, 2, draft);
    assert(c7 == 0);

    printf("PASS: test_ngram_lookup\n");
    return 0;
}
