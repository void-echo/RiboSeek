/*
 * nw_align.c - C-accelerated Needleman-Wunsch / Smith-Waterman alignment
 * for RNA structural-alphabet sequences.
 *
 * Compiled by setuptools as a CPython extension (so wheels build for every
 * supported Python version), but the Python side loads it via ctypes
 * because the data path is plain int* / double* buffers, not PyObjects.
 *
 * To make this work on every platform:
 *   - core functions are annotated EXPORT so MSVC also exports them
 *   - a trivial PyInit__nw_align stub is provided so setuptools is happy
 */

#include <stdlib.h>
#include <string.h>
#include <math.h>

#ifdef _WIN32
#  define EXPORT __declspec(dllexport)
#else
#  define EXPORT __attribute__((visibility("default")))
#endif


/* ------------------------------------------------------------------ */
/*  Needleman-Wunsch (global) alignment.                              */
/*  Returns raw_score / max(n, m).                                    */
/* ------------------------------------------------------------------ */
EXPORT double nw_align(const int *seq1, int n, const int *seq2, int m,
                       const double *score_matrix, int K, double gap_penalty) {
    if (n == 0 || m == 0) return 0.0;

    double *prev = (double *)malloc((m + 1) * sizeof(double));
    double *curr = (double *)malloc((m + 1) * sizeof(double));
    if (!prev || !curr) { free(prev); free(curr); return 0.0; }

    for (int j = 0; j <= m; j++) prev[j] = j * gap_penalty;

    for (int i = 1; i <= n; i++) {
        curr[0] = i * gap_penalty;
        int si = seq1[i - 1];
        for (int j = 1; j <= m; j++) {
            double match = prev[j - 1] + score_matrix[si * K + seq2[j - 1]];
            double del   = prev[j]     + gap_penalty;
            double ins   = curr[j - 1] + gap_penalty;
            double best  = match;
            if (del > best) best = del;
            if (ins > best) best = ins;
            curr[j] = best;
        }
        double *tmp = prev; prev = curr; curr = tmp;
    }

    double raw_score = prev[m];
    free(prev); free(curr);
    int maxlen = n > m ? n : m;
    return raw_score / maxlen;
}


/* ------------------------------------------------------------------ */
/*  Smith-Waterman (local) alignment.                                 */
/*  Returns max_score / min(n, m).                                    */
/* ------------------------------------------------------------------ */
EXPORT double sw_align(const int *seq1, int n, const int *seq2, int m,
                       const double *score_matrix, int K, double gap_penalty) {
    if (n == 0 || m == 0) return 0.0;

    double *prev = (double *)calloc(m + 1, sizeof(double));
    double *curr = (double *)calloc(m + 1, sizeof(double));
    if (!prev || !curr) { free(prev); free(curr); return 0.0; }

    double max_score = 0.0;

    for (int i = 1; i <= n; i++) {
        curr[0] = 0.0;
        int si = seq1[i - 1];
        for (int j = 1; j <= m; j++) {
            double match = prev[j - 1] + score_matrix[si * K + seq2[j - 1]];
            double del   = prev[j]     + gap_penalty;
            double ins   = curr[j - 1] + gap_penalty;
            double best  = 0.0;
            if (match > best) best = match;
            if (del > best)   best = del;
            if (ins > best)   best = ins;
            curr[j] = best;
            if (best > max_score) max_score = best;
        }
        double *tmp = prev; prev = curr; curr = tmp;
    }

    free(prev); free(curr);
    int minlen = n < m ? n : m;
    return minlen > 0 ? max_score / minlen : 0.0;
}


/* ------------------------------------------------------------------ */
/*  Batch: align one query against many targets.                      */
/* ------------------------------------------------------------------ */
EXPORT void batch_nw_align(const int *query, int query_len,
                           const int *targets, const int *target_offsets,
                           const int *target_lengths, int n_targets,
                           const double *score_matrix, int K, double gap_penalty,
                           double *scores_out) {
    for (int t = 0; t < n_targets; t++) {
        const int *target = targets + target_offsets[t];
        int target_len = target_lengths[t];
        scores_out[t] = nw_align(query, query_len, target, target_len,
                                 score_matrix, K, gap_penalty);
    }
}


/* ------------------------------------------------------------------ */
/*  Batch: align a list of (i, j) pairs from a sequence collection.   */
/* ------------------------------------------------------------------ */
EXPORT void batch_pairwise_align(const int *pairs, int n_pairs,
                                 const int *all_seqs, const int *seq_offsets,
                                 const int *seq_lengths,
                                 const double *score_matrix, int K, double gap_penalty,
                                 double *scores_out) {
    for (int p = 0; p < n_pairs; p++) {
        int i = pairs[2 * p];
        int j = pairs[2 * p + 1];
        const int *s1 = all_seqs + seq_offsets[i];
        const int *s2 = all_seqs + seq_offsets[j];
        scores_out[p] = nw_align(s1, seq_lengths[i], s2, seq_lengths[j],
                                 score_matrix, K, gap_penalty);
    }
}


/* ------------------------------------------------------------------ */
/*  CPython init stub.                                                */
/*                                                                    */
/*  We don't expose anything via the Python C API — Python loads the  */
/*  compiled .so / .pyd via ctypes and calls the EXPORT'd functions   */
/*  above directly. The PyInit symbol is here only so setuptools can  */
/*  package this as a regular extension module.                       */
/* ------------------------------------------------------------------ */
#include <Python.h>

static struct PyModuleDef _nw_align_module = {
    PyModuleDef_HEAD_INIT,
    "_nw_align",
    "Compiled NW/SW alignment kernels for riboseek (used via ctypes).",
    -1,
    NULL, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__nw_align(void) {
    return PyModule_Create(&_nw_align_module);
}
