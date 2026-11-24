// SPDX-License-Identifier: Apache-2.0
#pragma once
#include <cupqc/ntt.hpp>

namespace bytecrt::warp_ntt {
template<unsigned J, cupqc::nttDirection Direction>
using InnerNtt = decltype(cupqc::Algorithm<cupqc::algorithm::NTT>() +
    cupqc::Precision<uint32_t>() + cupqc::Direction<Direction>() +
    cupqc::Size<J>() + cupqc::Block() + cupqc::BlockDim<256>());

} // namespace bytecrt::warp_ntt
