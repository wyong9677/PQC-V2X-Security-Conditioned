#define _POSIX_C_SOURCE 200809L
#include <oqs/oqs.h>

#include <ctype.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0) {
        perror("clock_gettime");
        exit(2);
    }
    return (double)ts.tv_sec * 1.0e6 + (double)ts.tv_nsec / 1.0e3;
}

static int cmp_double(const void *a, const void *b) {
    const double da = *(const double *)a;
    const double db = *(const double *)b;
    return (da > db) - (da < db);
}

static void normalize_name(const char *src, char *dst, size_t cap) {
    size_t j = 0;
    for (size_t i = 0; src[i] != '\0' && j + 1 < cap; ++i) {
        unsigned char c = (unsigned char)src[i];
        if (isalnum(c)) {
            dst[j++] = (char)tolower(c);
        }
    }
    dst[j] = '\0';
}

static int normalized_equal(const char *a, const char *b) {
    char na[256], nb[256];
    normalize_name(a, na, sizeof(na));
    normalize_name(b, nb, sizeof(nb));
    return strcmp(na, nb) == 0;
}

static int normalized_contains(const char *haystack, const char *needle) {
    char h[256], n[256];
    normalize_name(haystack, h, sizeof(h));
    normalize_name(needle, n, sizeof(n));
    return strstr(h, n) != NULL || strstr(n, h) != NULL;
}

static const char *resolve_algorithm(const char *const *candidates, size_t ncand) {
    for (size_t c = 0; c < ncand; ++c) {
        for (size_t i = 0; i < OQS_SIG_alg_count(); ++i) {
            const char *id = OQS_SIG_alg_identifier(i);
            if (id != NULL && normalized_equal(id, candidates[c])) {
                OQS_SIG *sig = OQS_SIG_new(id);
                if (sig != NULL) {
                    OQS_SIG_free(sig);
                    return id;
                }
            }
        }
    }
    for (size_t c = 0; c < ncand; ++c) {
        for (size_t i = 0; i < OQS_SIG_alg_count(); ++i) {
            const char *id = OQS_SIG_alg_identifier(i);
            if (id != NULL && normalized_contains(id, candidates[c])) {
                OQS_SIG *sig = OQS_SIG_new(id);
                if (sig != NULL) {
                    OQS_SIG_free(sig);
                    return id;
                }
            }
        }
    }
    return NULL;
}

static int benchmark_one(const char *label, const char *alg_name, size_t iterations) {
    static const size_t msg_sizes[] = {64, 256, 1024};
    OQS_SIG *sig = OQS_SIG_new(alg_name);
    if (sig == NULL) {
        fprintf(stderr, "ERROR=OQS_SIG_new_failed algorithm=%s\n", alg_name);
        return 2;
    }

    uint8_t *pk = (uint8_t *)malloc(sig->length_public_key);
    uint8_t *sk = (uint8_t *)malloc(sig->length_secret_key);
    uint8_t *signature = (uint8_t *)malloc(sig->length_signature);
    double *samples = (double *)malloc(iterations * sizeof(double));
    if (pk == NULL || sk == NULL || signature == NULL || samples == NULL) {
        fprintf(stderr, "ERROR=allocation_failed algorithm=%s\n", alg_name);
        free(pk); free(sk); free(signature); free(samples); OQS_SIG_free(sig);
        return 2;
    }

    if (OQS_SIG_keypair(sig, pk, sk) != OQS_SUCCESS) {
        fprintf(stderr, "ERROR=keypair_failed algorithm=%s\n", alg_name);
        free(pk); free(sk); free(signature); free(samples); OQS_SIG_free(sig);
        return 2;
    }

    printf("ARTIFACT_ROW algorithm=%s liboqs_name=%s pk_bytes=%zu sk_bytes=%zu sig_bytes=%zu\n",
           label, alg_name, sig->length_public_key, sig->length_secret_key,
           sig->length_signature);

    for (size_t m = 0; m < sizeof(msg_sizes) / sizeof(msg_sizes[0]); ++m) {
        size_t msg_len = msg_sizes[m];
        uint8_t *msg = (uint8_t *)malloc(msg_len);
        if (msg == NULL) {
            fprintf(stderr, "ERROR=message_allocation_failed\n");
            free(pk); free(sk); free(signature); free(samples); OQS_SIG_free(sig);
            return 2;
        }
        for (size_t j = 0; j < msg_len; ++j) {
            msg[j] = (uint8_t)((j * 131u + msg_len * 17u + 29u) & 0xffu);
        }

        size_t sig_len = 0;
        if (OQS_SIG_sign(sig, signature, &sig_len, msg, msg_len, sk) != OQS_SUCCESS) {
            fprintf(stderr, "ERROR=sign_failed algorithm=%s msg=%zu\n", alg_name, msg_len);
            free(msg); free(pk); free(sk); free(signature); free(samples); OQS_SIG_free(sig);
            return 2;
        }
        if (OQS_SIG_verify(sig, msg, msg_len, signature, sig_len, pk) != OQS_SUCCESS) {
            fprintf(stderr, "ERROR=preverify_failed algorithm=%s msg=%zu\n", alg_name, msg_len);
            free(msg); free(pk); free(sk); free(signature); free(samples); OQS_SIG_free(sig);
            return 2;
        }

        for (size_t i = 0; i < iterations; ++i) {
            double t0 = now_us();
            OQS_STATUS rc = OQS_SIG_verify(sig, msg, msg_len, signature, sig_len, pk);
            double t1 = now_us();
            if (rc != OQS_SUCCESS) {
                fprintf(stderr, "ERROR=verify_failed algorithm=%s msg=%zu iteration=%zu\n",
                        alg_name, msg_len, i);
                free(msg); free(pk); free(sk); free(signature); free(samples); OQS_SIG_free(sig);
                return 2;
            }
            samples[i] = t1 - t0;
        }

        qsort(samples, iterations, sizeof(double), cmp_double);
        double median;
        if ((iterations & 1u) == 0u) {
            median = 0.5 * (samples[iterations / 2 - 1] + samples[iterations / 2]);
        } else {
            median = samples[iterations / 2];
        }
        size_t p95_index = (size_t)ceil(0.95 * (double)iterations);
        if (p95_index == 0) p95_index = 1;
        if (p95_index > iterations) p95_index = iterations;
        double p95 = samples[p95_index - 1];
        double maxv = samples[iterations - 1];

        printf("BENCH_ROW algorithm=%s liboqs_name=%s msg=%zu iterations=%zu median_us=%.6f p95_us=%.6f max_us=%.6f\n",
               label, alg_name, msg_len, iterations, median, p95, maxv);
        free(msg);
    }

    OQS_MEM_cleanse(sk, sig->length_secret_key);
    free(pk); free(sk); free(signature); free(samples); OQS_SIG_free(sig);
    return 0;
}

int main(int argc, char **argv) {
    size_t iterations = 10000;
    if (argc == 3 && strcmp(argv[1], "--iterations") == 0) {
        char *end = NULL;
        unsigned long n = strtoul(argv[2], &end, 10);
        if (end == NULL || *end != '\0' || n < 10) {
            fprintf(stderr, "ERROR=invalid_iterations\n");
            return 2;
        }
        iterations = (size_t)n;
    } else if (argc != 1) {
        fprintf(stderr, "usage: %s [--iterations N]\n", argv[0]);
        return 2;
    }

    const char *ml_candidates[] = {"ML-DSA-65", "ML_DSA_65"};
    const char *slh_candidates[] = {
        "SLH_DSA_PURE_SHA2_192S",
        "SLH-DSA-SHA2-192s",
        "SLH_DSA_SHA2_192S"
    };

    printf("LIBOQS_VERSION=%s\n", OQS_version());
    printf("OQS_SIG_ALG_COUNT=%zu\n", OQS_SIG_alg_count());

    const char *ml = resolve_algorithm(ml_candidates, sizeof(ml_candidates)/sizeof(ml_candidates[0]));
    const char *slh = resolve_algorithm(slh_candidates, sizeof(slh_candidates)/sizeof(slh_candidates[0]));
    if (ml == NULL) {
        fprintf(stderr, "ERROR=ML_DSA_65_NOT_FOUND\n");
        return 3;
    }
    if (slh == NULL) {
        fprintf(stderr, "ERROR=SLH_DSA_SHA2_192S_NOT_FOUND\n");
        return 3;
    }

    printf("ALGORITHM_RESOLVED algorithm=ML-DSA-65 liboqs_name=%s\n", ml);
    printf("ALGORITHM_RESOLVED algorithm=SLH-DSA-SHA2-192s liboqs_name=%s\n", slh);

    int rc = benchmark_one("ML-DSA-65", ml, iterations);
    if (rc != 0) return rc;
    rc = benchmark_one("SLH-DSA-SHA2-192s", slh, iterations);
    if (rc != 0) return rc;

    printf("NATIVE_PQC_BENCH=PASS iterations=%zu\n", iterations);
    return 0;
}
