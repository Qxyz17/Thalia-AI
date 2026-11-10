// Thalia - HDC implementation
#include "hdc.h"
#include <openssl/sha.h>
#include <algorithm>
#include <cstring>
#include <random>

namespace thalia {

HdcVec::HdcVec() {
    bits_.fill(0);
}

// --- SHA256 ---
std::array<uint8_t, 32> sha256(const std::string& data) {
    std::array<uint8_t, 32> out{};
    SHA256(reinterpret_cast<const unsigned char*>(data.data()),
           data.size(), out.data());
    return out;
}

std::string sha256_hex(const std::string& data) {
    static const char* hex = "0123456789abcdef";
    auto h = sha256(data);
    std::string s(64, ' ');
    for (size_t i = 0; i < 32; ++i) {
        s[2*i]     = hex[h[i] >> 4];
        s[2*i + 1] = hex[h[i] & 0xF];
    }
    return s;
}

// --- Seeded vector generation ---
// Expand SHA256(seed) into 1250 bytes via repeated SHA256(seed || counter).
// This gives a deterministic, well-mixed bit pattern from any seed string.
static std::array<uint8_t, HDC_DIM / 8> expand_seed(const std::string& seed) {
    std::array<uint8_t, HDC_DIM / 8> out{};
    uint32_t counter = 0;
    size_t filled = 0;
    while (filled < out.size()) {
        std::string msg = seed + std::string(reinterpret_cast<const char*>(&counter), 4);
        auto h = sha256(msg);  // 32 bytes
        size_t to_copy = std::min<size_t>(32, out.size() - filled);
        std::memcpy(out.data() + filled, h.data(), to_copy);
        filled += to_copy;
        ++counter;
    }
    return out;
}

HdcVec HdcVec::from_seed(const std::string& seed) {
    HdcVec v;
    v.bits_ = expand_seed(seed);
    return v;
}

HdcVec HdcVec::from_bytes(const uint8_t* data, size_t n) {
    // Hash the data, then expand. Two different inputs -> different vectors
    // with overwhelming probability (collision resistance of SHA256).
    auto h = sha256(std::string(reinterpret_cast<const char*>(data), n));
    std::string seed(h.begin(), h.end());
    return from_seed(seed);
}

HdcVec HdcVec::bind(const HdcVec& a, const HdcVec& b) {
    HdcVec r;
    for (size_t i = 0; i < r.bits_.size(); ++i) {
        r.bits_[i] = a.bits_[i] ^ b.bits_[i];
    }
    return r;
}

HdcVec HdcVec::bundle(const std::vector<HdcVec>& vs) {
    HdcVec r;
    if (vs.empty()) return r;
    // Per-bit majority vote. HDC_DIM is divisible by 8.
    for (size_t byte = 0; byte < r.bits_.size(); ++byte) {
        for (int bit = 0; bit < 8; ++bit) {
            int sum = 0;
            for (const auto& v : vs) {
                sum += (v.bits_[byte] >> bit) & 1 ? +1 : -1;
            }
            if (sum > 0) r.bits_[byte] |= (1 << bit);
        }
    }
    return r;
}

HdcVec HdcVec::bundle(const HdcVec& a, const HdcVec& b, int wa, int wb) {
    HdcVec r;
    for (size_t byte = 0; byte < r.bits_.size(); ++byte) {
        for (int bit = 0; bit < 8; ++bit) {
            int sa = (a.bits_[byte] >> bit) & 1 ? +1 : -1;
            int sb = (b.bits_[byte] >> bit) & 1 ? +1 : -1;
            int sum = sa * wa + sb * wb;
            if (sum > 0) r.bits_[byte] |= (1 << bit);
        }
    }
    return r;
}

HdcVec HdcVec::bundle_weighted(const std::vector<std::pair<HdcVec, double>>& vs) {
    HdcVec r;
    if (vs.empty()) return r;
    for (size_t byte = 0; byte < r.bits_.size(); ++byte) {
        for (int bit = 0; bit < 8; ++bit) {
            double sum = 0;
            for (const auto& [v, w] : vs) {
                sum += w * ((v.bits_[byte] >> bit) & 1 ? +1.0 : -1.0);
            }
            if (sum > 0) r.bits_[byte] |= (1 << bit);
        }
    }
    return r;
}

double HdcVec::sim(const HdcVec& a, const HdcVec& b) {
    int agree = 0;
    for (size_t i = 0; i < a.bits_.size(); ++i) {
        uint8_t x = a.bits_[i] ^ b.bits_[i];
        // count zeros in x = agreements
        agree += 8 - __builtin_popcount(x);
    }
    return static_cast<double>(agree) / HDC_DIM;
}

void HdcVec::nudge(const HdcVec& other, double alpha) {
    // Soft mix: for each bit, with prob alpha take other's bit, else keep self's.
    // Deterministic via counter-based PRNG seeded from a constant (we want
    // reproducible nudges, not stochastic).
    // Implement by: bundle(self, other, 1-alpha, alpha) approximated to integers.
    int wa = static_cast<int>(1000 * (1.0 - alpha));
    int wb = 1000 - wa;
    *this = bundle(*this, other, wa, wb);
}

std::string HdcVec::serialize() const {
    return std::string(reinterpret_cast<const char*>(bits_.data()), bits_.size());
}

HdcVec HdcVec::deserialize(const std::string& raw) {
    HdcVec v;
    if (raw.size() == v.bits_.size()) {
        std::memcpy(v.bits_.data(), raw.data(), v.bits_.size());
    }
    return v;
}

}  // namespace thalia
