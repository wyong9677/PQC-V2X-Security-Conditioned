#include <oqs/oqs.h>
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec * 1e6 + (double) ts.tv_nsec / 1e3;
}

static int cmp_double(const void *a, const void *b) {
    const double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

static double quantile_sorted(double *x, size_t n, double q) {
    if (n == 0) return NAN;
    double pos = q * (double)(n - 1);
    size_t lo = (size_t) floor(pos), hi = (size_t) ceil(pos);
    if (lo == hi) return x[lo];
    double w = pos - (double)lo;
    return x[lo] * (1.0 - w) + x[hi] * w;
}

static OQS_SIG *new_first_available(const char **names, size_t n, const char **chosen) {
    for (size_t i = 0; i < n; ++i) {
        OQS_SIG *s = OQS_SIG_new(names[i]);
        if (s != NULL) {
            *chosen = names[i];
            return s;
        }
    }
    return NULL;
}

static int benchmark_alg(FILE *fp, const char *report_name, const char **aliases, size_t n_aliases, size_t iters) {
    const char *chosen = NULL;
    OQS_SIG *sig = new_first_available(aliases, n_aliases, &chosen);
    if (sig == NULL) {
        fprintf(stderr, "BENCH_ALGORITHM_UNAVAILABLE=%s\n", report_name);
        return 1;
    }

    const size_t msg_sizes[] = {64, 256, 1024};
    uint8_t *pk = OQS_MEM_malloc(sig->length_public_key);
    uint8_t *sk = OQS_MEM_malloc(sig->length_secret_key);
    uint8_t *signature = OQS_MEM_malloc(sig->length_signature);
    double *times = malloc(iters * sizeof(double));
    if (!pk || !sk || !signature || !times) {
        fprintf(stderr, "allocation failure\n");
        OQS_MEM_insecure_free(pk); OQS_MEM_secure_free(sk, sig->length_secret_key);
        OQS_MEM_insecure_free(signature); free(times); OQS_SIG_free(sig);
        return 2;
    }
    if (OQS_SIG_keypair(sig, pk, sk) != OQS_SUCCESS) {
        fprintf(stderr, "keypair failure %s\n", report_name);
        return 3;
    }

    for (size_t mi = 0; mi < sizeof(msg_sizes)/sizeof(msg_sizes[0]); ++mi) {
        size_t mlen = msg_sizes[mi];
        uint8_t *msg = OQS_MEM_malloc(mlen);
        if (!msg) return 4;
        OQS_randombytes(msg, mlen);
        size_t siglen = 0;
        if (OQS_SIG_sign(sig, signature, &siglen, msg, mlen, sk) != OQS_SUCCESS) return 5;
        if (OQS_SIG_verify(sig, msg, mlen, signature, siglen, pk) != OQS_SUCCESS) return 6;
        for (size_t w = 0; w < 64; ++w) {
            if (OQS_SIG_verify(sig, msg, mlen, signature, siglen, pk) != OQS_SUCCESS) return 7;
        }
        for (size_t i = 0; i < iters; ++i) {
            double t0 = now_us();
            OQS_STATUS rc = OQS_SIG_verify(sig, msg, mlen, signature, siglen, pk);
            double t1 = now_us();
            if (rc != OQS_SUCCESS) return 8;
            times[i] = t1 - t0;
        }
        qsort(times, iters, sizeof(double), cmp_double);
        double med = quantile_sorted(times, iters, 0.50);
        double p95 = quantile_sorted(times, iters, 0.95);
        double mx = times[iters - 1];
        fprintf(fp, "%s,%zu,%zu,%zu,%zu,%zu,%.9f,%.9f,%.9f,%zu\n",
            report_name, mlen, sig->length_public_key, sig->length_secret_key,
            sig->length_signature, siglen, med, p95, mx, iters);
        fprintf(stderr, "BENCH_ROW algorithm=%s liboqs_name=%s msg=%zu median_us=%.3f p95_us=%.3f max_us=%.3f\n",
            report_name, chosen, mlen, med, p95, mx);
        OQS_MEM_insecure_free(msg);
    }

    OQS_MEM_insecure_free(pk);
    OQS_MEM_secure_free(sk, sig->length_secret_key);
    OQS_MEM_insecure_free(signature);
    free(times);
    OQS_SIG_free(sig);
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s OUTPUT.csv [iterations]\n", argv[0]);
        return 64;
    }
    size_t iters = 5000;
    if (argc >= 3) {
        char *end = NULL;
        errno = 0;
        unsigned long v = strtoul(argv[2], &end, 10);
        if (errno || end == argv[2] || *end || v < 100) return 65;
        iters = (size_t)v;
    }
    FILE *fp = fopen(argv[1], "w");
    if (!fp) { perror("fopen"); return 66; }
    fprintf(fp, "algorithm,message_bytes,public_key_bytes,secret_key_bytes,signature_capacity_bytes,signature_actual_bytes,verify_median_us,verify_p95_us,verify_max_us,iterations\n");

    OQS_init();
    fprintf(stderr, "LIBOQS_VERSION=%s\n", OQS_version());
    const char *ml[] = {"ML-DSA-65"};
    const char *slh[] = {"SLH-DSA-SHA2-192s", "SLH_DSA_PURE_SHA2_192S", "SPHINCS+-SHA2-192s-simple"};
    int r1 = benchmark_alg(fp, "ML-DSA-65", ml, sizeof(ml)/sizeof(ml[0]), iters);
    int r2 = benchmark_alg(fp, "SLH-DSA-SHA2-192s", slh, sizeof(slh)/sizeof(slh[0]), iters);
    OQS_destroy();
    fclose(fp);
    if (r1 || r2) {
        fprintf(stderr, "NATIVE_PQC_BENCH=PARTIAL_OR_FAIL ml=%d slh=%d\n", r1, r2);
        return 2;
    }
    fprintf(stderr, "NATIVE_PQC_BENCH=PASS iterations=%zu output=%s\n", iters, argv[1]);
    return 0;
}
