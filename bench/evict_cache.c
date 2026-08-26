/* evict_cache.c — best-effort drop page cache for a file.
 * Builds without nvcc, just gcc. Usage: ./evict_cache <path>
 * Strategy: open RO, fadvise DONTNEED on the full file, close.
 * Kernel may ignore for shared mappings, but combined with dd+other file
 * reads it usually evicts. We print mincore residency if mappable. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <file>\n", argv[0]); return 1; }
    int fd = open(argv[1], O_RDONLY);
    if (fd < 0) { perror("open"); return 1; }
    struct stat st; fstat(fd, &st);
    size_t n = (size_t)st.st_size;
    /* short mmap, sample residency, madvise DONTNEED, unmap, fadvise */
    void *p = mmap(NULL, n, PROT_READ, MAP_SHARED, fd, 0);
    if (p != MAP_FAILED) {
        size_t pg = (size_t)getpagesize();
        size_t npg = n / pg;
        unsigned char *vec = (unsigned char*)malloc(npg);
        if (mincore(p, n, vec) == 0) {
            size_t res = 0;
            for (size_t i = 0; i < npg; i++) res += (vec[i] & 1);
            fprintf(stderr, "[evict] before: %.1f%% resident (%zu/%zu pages)\n",
                    100.0 * res / npg, res, npg);
        }
        free(vec);
        madvise(p, n, MADV_DONTNEED);
        munmap(p, n);
    }
    posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);
    close(fd);
    fprintf(stderr, "[evict] done for %s (%zu bytes)\n", argv[1], n);
    return 0;
}
