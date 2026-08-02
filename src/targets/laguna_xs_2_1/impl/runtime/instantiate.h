#pragma once

// Laguna-specific runtime instantiation.
// Laguna does not support GDN, Vision, MTP, or DFlash. We include only the
// files that can compile with Laguna's type layout; the rest are guarded
// behind feature checks that evaluate to false at compile time.

#include "targets/qwen3_6/impl/runtime/layouts.h"
#include "targets/qwen3_6/impl/runtime/schedule.h"
#include "targets/qwen3_6/impl/runtime/program.h"
#include "targets/qwen3_6/impl/runtime/api_impl.h"

// Laguna-specific implementations of the shared runtime components.
// Order matters: text_prefill → speculative → graph → decode → text_context → layouts → program → request_plan
#include "targets/laguna_xs_2_1/impl/runtime/text_prefill_impl.h"
#include "targets/laguna_xs_2_1/impl/runtime/speculative_impl.h"
#include "targets/laguna_xs_2_1/impl/runtime/graph_impl.h"
#include "targets/laguna_xs_2_1/impl/runtime/decode_impl.h"
#include "targets/laguna_xs_2_1/impl/runtime/text_context_impl.h"
#include "targets/laguna_xs_2_1/impl/runtime/layouts_impl.h"
#include "targets/laguna_xs_2_1/impl/runtime/program_impl.h"
#include "targets/laguna_xs_2_1/impl/runtime/request_plan_impl.h"