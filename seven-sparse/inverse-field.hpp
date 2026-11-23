#pragma once
#include "rader7.hpp"
namespace seven_inverse {
using namespace rader7;
struct Field {
    Mod mod;
    Weights inverse;
    Twiddle recovery[7];
};

}
