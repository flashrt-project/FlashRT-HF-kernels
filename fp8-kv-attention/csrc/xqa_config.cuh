// SPDX-License-Identifier: Apache-2.0
#pragma once

#define NDEBUG 1
#define BEAM_WIDTH 1
#define USE_INPUT_KV 0
#define USE_CUSTOM_BARRIER 1
#define MLA_WRAPPER 0
#define USE_SM90_MHA 0
#define INPUT_FP16 0
#define DTYPE __nv_bfloat16
#define CACHE_ELEM_ENUM 2
#define TOKENS_PER_PAGE 128
#define HEAD_ELEMS 256
#define HEAD_GRP_SIZE 6
#define SLIDING_WINDOW 0
#define LOW_PREC_OUTPUT 0
#define SPEC_DEC 1
