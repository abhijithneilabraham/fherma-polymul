#pragma once
#include "radix4.hpp"
namespace four_inverse {
using namespace radix4;
struct Field {
    Mod mod;
    Twiddle inverse;
    Twiddle recovery[4];
};

}
