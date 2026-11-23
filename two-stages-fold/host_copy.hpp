#pragma once
#include "fherma.h"
#include <cstring>

namespace host_copy {
inline void upload_bytes(uint32_t* pinned, const std::vector<uint32_t>& source) {
    if (!source.empty()) std::memcpy(pinned, source.data(), source.size() * sizeof(uint32_t));
}
inline fherma::Outputs download_bytes(fherma::Outputs out, const uint32_t* pinned, size_t words) {
    if (!out.c.data.empty()) throw std::runtime_error("download target must be empty");
    if (words) {
        out.c.data.reserve(words);
        out.c.data.insert(out.c.data.end(), pinned, pinned + words);
    }
    return out;
}
} // namespace host_copy
