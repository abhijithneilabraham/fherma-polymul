#pragma once

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <utility>

// Host-only arithmetic for public point constants. No operand is accepted here.
namespace setup {
using Words = std::vector<uint32_t>;

inline int compare(const Words& a, const Words& b) {
    for (size_t j = a.size(); j-- > 0;) {
        if (a[j] != b[j]) return a[j] < b[j] ? -1 : 1;
    }
    return 0;
}

inline uint32_t subtract(Words& a, const Words& b) {
    uint64_t borrow = 0;
    for (size_t j = 0; j < a.size(); ++j) {
        const uint64_t rhs = uint64_t(b[j]) + borrow;
        const uint64_t lhs = a[j];
        a[j] = uint32_t(lhs - rhs);
        borrow = lhs < rhs;
    }
    return uint32_t(borrow);
}

struct Field {
    Words q, one, r, r2;
    uint32_t negative_inverse;

    explicit Field(Words modulus) : q(std::move(modulus)), one(q.size(), 0) {
        if (q.empty() || !(q[0] & 1)) throw std::runtime_error("odd modulus required");
        one[0] = 1;
        if (compare(q, one) <= 0) throw std::runtime_error("modulus must exceed one");
        uint32_t inverse = 1;
        for (unsigned j = 0; j < 5; ++j) inverse *= 2u - q[0] * inverse;
        negative_inverse = 0u - inverse;
        r = one;
        for (size_t j = 0; j < 32 * q.size(); ++j) r = add(r, r);
        r2 = r;
        for (size_t j = 0; j < 32 * q.size(); ++j) r2 = add(r2, r2);
    }

    Words add(const Words& a, const Words& b) const {
        Words out(q.size());
        uint64_t carry = 0;
        for (size_t j = 0; j < q.size(); ++j) {
            const uint64_t sum = uint64_t(a[j]) + b[j] + carry;
            out[j] = uint32_t(sum);
            carry = sum >> 32;
        }
        if (carry || compare(out, q) >= 0) subtract(out, q);
        return out;
    }

    Words sub(const Words& a, const Words& b) const {
        Words out = a;
        if (subtract(out, b)) {
            uint64_t carry = 0;
            for (size_t j = 0; j < q.size(); ++j) {
                const uint64_t sum = uint64_t(out[j]) + q[j] + carry;
                out[j] = uint32_t(sum);
                carry = sum >> 32;
            }
        }
        return out;
    }

    // Coarsely integrated operand scanning. Inputs are canonical residues.
    Words mont(const Words& a, const Words& b) const {
        const size_t n = q.size();
        Words t(n + 2, 0);
        for (size_t i = 0; i < n; ++i) {
            uint64_t carry = 0;
            for (size_t j = 0; j < n; ++j) {
                const uint64_t v = uint64_t(a[j]) * b[i] + t[j] + carry;
                t[j] = uint32_t(v);
                carry = v >> 32;
            }
            uint64_t v = uint64_t(t[n]) + carry;
            t[n] = uint32_t(v);
            t[n + 1] += uint32_t(v >> 32);
            const uint32_t m = t[0] * negative_inverse;
            carry = 0;
            for (size_t j = 0; j < n; ++j) {
                v = uint64_t(m) * q[j] + t[j] + carry;
                if (j) t[j - 1] = uint32_t(v);
                carry = v >> 32;
            }
            v = uint64_t(t[n]) + carry;
            t[n - 1] = uint32_t(v);
            t[n] = uint32_t(v >> 32) + t[n + 1];
            t[n + 1] = 0;
        }
        const bool high = t[n] != 0;
        t.resize(n);
        if (high || compare(t, q) >= 0) subtract(t, q);
        return t;
    }

    Words encode(const Words& a) const { return mont(a, r2); }
    Words decode(const Words& a) const { return mont(a, one); }

    Words power(Words base, const Words& exponent) const {
        Words out = r;
        size_t significant = exponent.size();
        while (significant && !exponent[significant - 1]) --significant;
        for (size_t j = 0; j < significant; ++j) {
            unsigned limit = 32;
            if (j + 1 == significant) {
                limit = 0;
                for (uint32_t top = exponent[j]; top; top >>= 1) ++limit;
            }
            for (unsigned bit = 0; bit < limit; ++bit) {
                if ((exponent[j] >> bit) & 1) out = mont(out, base);
                base = mont(base, base);
            }
        }
        return out;
    }
};

inline Words shifted(const Words& value, unsigned bits) {
    Words out(value.size(), 0);
    const size_t word = bits / 32;
    const unsigned shift = bits % 32;
    for (size_t j = 0; j + word < value.size(); ++j) {
        uint64_t v = value[j + word];
        if (shift && j + word + 1 < value.size()) v |= uint64_t(value[j + word + 1]) << 32;
        out[j] = uint32_t(v >> shift);
    }
    return out;
}

struct Plan {
    uint32_t n;
    unsigned log_n;
    Field field;
    Words forward, inverse, twist_r2, scale_normal;

    Plan(uint32_t count, Words modulus) : n(count), log_n(0), field(std::move(modulus)) {
        if (!n || (n & (n - 1)) || n > (1u << 30)) throw std::runtime_error("invalid NTT length");
        while ((1u << log_n) < n) ++log_n;
        Words qm1 = field.q;
        subtract(qm1, field.one);
        if ((uint64_t(field.q[0]) - 1) % (uint64_t(n) * 2)) throw std::runtime_error("q is not 1 mod 2N");
        const Words exponent = shifted(qm1, log_n + 1);
        Words n_exponent(field.q.size(), 0);
        n_exponent[0] = n;
        const Words minus_one = field.sub(Words(field.q.size(), 0), field.r);
        Words psi;
        for (uint32_t candidate = 2; candidate != 0; ++candidate) {
            Words base(field.q.size(), 0);
            base[0] = candidate;
            if (compare(base, field.q) >= 0) break;
            Words value = field.power(field.encode(base), exponent);
            if (field.power(value, n_exponent) == minus_one) { psi = std::move(value); break; }
        }
        if (psi.empty()) throw std::runtime_error("no primitive 2N-th root found");
        Words inverse_exponent(field.q.size(), 0);
        inverse_exponent[0] = 2 * n - 1;
        const Words inverse_psi = field.power(psi, inverse_exponent);
        const Words omega = field.mont(psi, psi);
        const Words inverse_omega = field.mont(inverse_psi, inverse_psi);
        const size_t limbs = field.q.size();
        forward.resize(size_t(n / 2) * limbs);
        inverse.resize(size_t(n / 2) * limbs);
        twist_r2.resize(size_t(n) * limbs);
        scale_normal.resize(size_t(n) * limbs);
        const Words n_inverse = field.sub(Words(limbs, 0), shifted(qm1, log_n));
        const Words encoded_n_inverse = field.encode(n_inverse);
        Words fp = field.r, ip = field.r, tp = field.r, up = field.r;
        for (uint32_t i = 0; i < n; ++i) {
            if (i < n / 2) {
                std::copy(fp.begin(), fp.end(), forward.begin() + size_t(i) * limbs);
                std::copy(ip.begin(), ip.end(), inverse.begin() + size_t(i) * limbs);
                fp = field.mont(fp, omega);
                ip = field.mont(ip, inverse_omega);
            }
            const Words t = field.mont(tp, field.r2);
            const Words u = field.decode(field.mont(encoded_n_inverse, up));
            std::copy(t.begin(), t.end(), twist_r2.begin() + size_t(i) * limbs);
            std::copy(u.begin(), u.end(), scale_normal.begin() + size_t(i) * limbs);
            tp = field.mont(tp, psi);
            up = field.mont(up, inverse_psi);
        }
    }
};
}  // namespace setup
