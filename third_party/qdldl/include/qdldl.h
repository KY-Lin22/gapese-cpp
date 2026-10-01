/*
 * Embedded QDLDL interface for Gapese.
 * QDLDL: https://github.com/osqp/qdldl
 * SPDX-License-Identifier: Apache-2.0
 */
#ifndef QDLDL_H
#define QDLDL_H

#include "qdldl_types.h"
#include "qdldl_version.h"

#ifdef __cplusplus
extern "C" {
#endif

QDLDL_int QDLDL_etree(
    QDLDL_int n,
    const QDLDL_int* Ap,
    const QDLDL_int* Ai,
    QDLDL_int* work,
    QDLDL_int* Lnz,
    QDLDL_int* etree);

QDLDL_int QDLDL_factor(
    QDLDL_int n,
    const QDLDL_int* Ap,
    const QDLDL_int* Ai,
    const QDLDL_float* Ax,
    QDLDL_int* Lp,
    QDLDL_int* Li,
    QDLDL_float* Lx,
    QDLDL_float* D,
    QDLDL_float* Dinv,
    const QDLDL_int* Lnz,
    const QDLDL_int* etree,
    QDLDL_bool* bwork,
    QDLDL_int* iwork,
    QDLDL_float* fwork);

void QDLDL_solve(
    QDLDL_int n,
    const QDLDL_int* Lp,
    const QDLDL_int* Li,
    const QDLDL_float* Lx,
    const QDLDL_float* Dinv,
    QDLDL_float* x);

void QDLDL_Lsolve(
    QDLDL_int n,
    const QDLDL_int* Lp,
    const QDLDL_int* Li,
    const QDLDL_float* Lx,
    QDLDL_float* x);

void QDLDL_Ltsolve(
    QDLDL_int n,
    const QDLDL_int* Lp,
    const QDLDL_int* Li,
    const QDLDL_float* Lx,
    QDLDL_float* x);

#ifdef __cplusplus
}
#endif

#endif
