#pragma once

// Laguna XS 2.1 sequence planning: KV cache layout, arena allocation, and
// per-layer geometry dispatch for the mixed global/SWA attention pattern.
//
// This header provides plan_laguna_kv_cache for use during conversion
// and validation. The runtime SequencePlan is provided by runtime/layouts.h.

#include "targets/laguna_xs_2_1/impl/config.h"
#include "targets/laguna_xs_2_1/impl/kv_cache.h"

#include "core/layout.h"

#include <cstdint>