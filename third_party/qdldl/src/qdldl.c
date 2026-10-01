/*
 * This file is part of QDLDL, a library for performing the LDL^T factorization
 * of a symmetric indefinite matrix.
 *
 * QDLDL is part of the OSQP project, and is available at https://github.com/osqp/qdldl.
 *
 * Copyright 2018, Paul Goulart, Bartolomeo Stellato, Goran Banjac, The OSQP developers
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 * SPDX-ExternalRef: PACKAGE_MANAGER purl pkg:github/osqp/qdldl
 */
#include "qdldl.h"

#define QDLDL_UNKNOWN (-1)
#define QDLDL_USED (1)
#define QDLDL_UNUSED (0)

/* Compute the elimination tree for a quasidefinite matrix
 * in compressed sparse column form.
 */
QDLDL_int QDLDL_etree(const QDLDL_int n, const QDLDL_int* Ap, const QDLDL_int* Ai,
                      QDLDL_int* work, QDLDL_int* Lnz, QDLDL_int* etree) {
    QDLDL_int i = 0;
    QDLDL_int j = 0;
    QDLDL_int p = 0;
    QDLDL_int sumLnz = 0;

    for (i = 0; i < n; i++) {
        /* Zero out Lnz and work. Set all etree values to unknown. */
        work[i] = 0;
        Lnz[i] = 0;
        etree[i] = QDLDL_UNKNOWN;
        /* Abort if A doesn't have at least one entry in every column. */
        if (Ap[i] == Ap[i + 1]) {
            return -1;
        }
    }

    for (j = 0; j < n; j++) {
        work[j] = j;
        for (p = Ap[j]; p < Ap[j + 1]; p++) {
            i = Ai[p];
            /* Abort if entries on lower triangle. */
            if (i > j) {
                return -1;
            }
            while (work[i] != j) {
                if (etree[i] == QDLDL_UNKNOWN) {
                    etree[i] = j;
                }
                Lnz[i]++;
                work[i] = j;
                i = etree[i];
            }
        }
    }

    /* Compute the total nonzeros in L. */
    sumLnz = 0;
    for (i = 0; i < n; i++) {
        if (sumLnz > QDLDL_INT_MAX - Lnz[i]) {
            sumLnz = -2;
            break;
        } else {
            sumLnz += Lnz[i];
        }
    }
    return sumLnz;
}

QDLDL_int QDLDL_factor(const QDLDL_int n, const QDLDL_int* Ap, const QDLDL_int* Ai,
                       const QDLDL_float* Ax, QDLDL_int* Lp, QDLDL_int* Li,
                       QDLDL_float* Lx, QDLDL_float* D, QDLDL_float* Dinv,
                       const QDLDL_int* Lnz, const QDLDL_int* etree,
                       QDLDL_bool* bwork, QDLDL_int* iwork, QDLDL_float* fwork) {
    QDLDL_int i = 0;
    QDLDL_int j = 0;
    QDLDL_int k = 0;
    QDLDL_int nnzY = 0;
    QDLDL_int bidx = 0;
    QDLDL_int cidx = 0;
    QDLDL_int nextIdx = 0;
    QDLDL_int nnzE = 0;
    QDLDL_int tmpIdx = 0;
    QDLDL_int* yIdx;
    QDLDL_int* elimBuffer;
    QDLDL_int* LNextSpaceInCol;
    QDLDL_float* yVals;
    QDLDL_float yVals_cidx = 0.0;
    QDLDL_bool* yMarkers;
    QDLDL_int positiveValuesInD = 0;

    /* Partition working memory into pieces. */
    yMarkers = bwork;
    yIdx = iwork;
    elimBuffer = iwork + n;
    LNextSpaceInCol = iwork + n * 2;
    yVals = fwork;

    Lp[0] = 0;
    for (i = 0; i < n; i++) {
        Lp[i + 1] = Lp[i] + Lnz[i];
        yMarkers[i] = QDLDL_UNUSED;
        yVals[i] = 0.0;
        D[i] = 0.0;
        LNextSpaceInCol[i] = Lp[i];
    }

    /* First element of the diagonal D. */
    D[0] = Ax[0];
    if (D[0] == 0.0) {
        return -1;
    }
    if (D[0] > 0.0) {
        positiveValuesInD++;
    }
    Dinv[0] = 1 / D[0];

    for (k = 1; k < n; k++) {
        nnzY = 0;
        tmpIdx = Ap[k + 1];
        for (i = Ap[k]; i < tmpIdx; i++) {
            bidx = Ai[i];
            if (bidx == k) {
                D[k] = Ax[i];
                continue;
            }
            yVals[bidx] = Ax[i];
            nextIdx = bidx;
            if (yMarkers[nextIdx] == QDLDL_UNUSED) {
                yMarkers[nextIdx] = QDLDL_USED;
                elimBuffer[0] = nextIdx;
                nnzE = 1;
                nextIdx = etree[bidx];
                while (nextIdx != QDLDL_UNKNOWN && nextIdx < k) {
                    if (yMarkers[nextIdx] == QDLDL_USED) {
                        break;
                    }
                    yMarkers[nextIdx] = QDLDL_USED;
                    elimBuffer[nnzE] = nextIdx;
                    nnzE++;
                    nextIdx = etree[nextIdx];
                }
                while (nnzE) {
                    yIdx[nnzY++] = elimBuffer[--nnzE];
                }
            }
        }

        for (i = (nnzY - 1); i >= 0; i--) {
            cidx = yIdx[i];
            tmpIdx = LNextSpaceInCol[cidx];
            yVals_cidx = yVals[cidx];
            for (j = Lp[cidx]; j < tmpIdx; j++) {
                yVals[Li[j]] -= Lx[j] * yVals_cidx;
            }
            Li[tmpIdx] = k;
            Lx[tmpIdx] = yVals_cidx * Dinv[cidx];
            D[k] -= yVals_cidx * Lx[tmpIdx];
            LNextSpaceInCol[cidx]++;
            yVals[cidx] = 0.0;
            yMarkers[cidx] = QDLDL_UNUSED;
        }

        if (D[k] == 0.0) {
            return -1;
        }
        if (D[k] > 0.0) {
            positiveValuesInD++;
        }
        Dinv[k] = 1 / D[k];
    }

    return positiveValuesInD;
}

void QDLDL_Lsolve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                  const QDLDL_float* Lx, QDLDL_float* x) {
    QDLDL_int i = 0;
    QDLDL_int j = 0;
    for (i = 0; i < n; i++) {
        QDLDL_float val = x[i];
        for (j = Lp[i]; j < Lp[i + 1]; j++) {
            x[Li[j]] -= Lx[j] * val;
        }
    }
}

void QDLDL_Ltsolve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                   const QDLDL_float* Lx, QDLDL_float* x) {
    QDLDL_int i = 0;
    QDLDL_int j = 0;
    for (i = n - 1; i >= 0; i--) {
        QDLDL_float val = x[i];
        for (j = Lp[i]; j < Lp[i + 1]; j++) {
            val -= Lx[j] * x[Li[j]];
        }
        x[i] = val;
    }
}

void QDLDL_solve(const QDLDL_int n, const QDLDL_int* Lp, const QDLDL_int* Li,
                 const QDLDL_float* Lx, const QDLDL_float* Dinv, QDLDL_float* x) {
    QDLDL_int i = 0;
    QDLDL_Lsolve(n, Lp, Li, Lx, x);
    for (i = 0; i < n; i++) {
        x[i] *= Dinv[i];
    }
    QDLDL_Ltsolve(n, Lp, Li, Lx, x);
}
