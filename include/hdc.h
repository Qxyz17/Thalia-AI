// Thalia - Hyperdimensional Computing layer
// 10000-dim bipolar (+1/-1) vectors, SHA256-seeded, no training
#pragma once
#include <array>
#include <cstdint>
#include <string>
#include <vector>

namespace thalia {

constexpr size_t HDC_DIM = 10000;

// A bipolar vector stored as packed bits: each byte holds 8 dims.
// +1 -> bit set, -1 -> bit clear. This makes XOR = binding trivially.
class HdcVec {
public:
    HdcVec();                                  // random zero state
    static HdcVec from_seed(const std::string& seed);     // SHA256(seed) -> deterministic
    static HdcVec from_bytes(const uint8_t* data, size_t n);

    // Ops
    static HdcVec bind(const HdcVec& a, const HdcVec& b);      // XOR
    static HdcVec bundle(const std::vector<HdcVec>& vs);       // majority vote
    static HdcVec bundle(const HdcVec& a, const HdcVec& b, int wa = 1, int wb = 1);

    // Weighted superposition: per-bit sum of w * (+1/-1), majority sign
    // wins. Ties resolve to bit clear (deterministic). Empty input ->
    // zero vector. Used by the Semantics layer to fold co-occurrence
    // counts into a vector.
    static HdcVec bundle_weighted(const std::vector<std::pair<HdcVec, double>>& vs);

    // Similarity in [0,1]: 1 = identical, 0.5 = uncorrelated, 0 = inverse
    static double sim(const HdcVec& a, const HdcVec& b);

    // Soft update: (1-alpha)*self + alpha*other, then re-binarize
    void nudge(const HdcVec& other, double alpha);

    // Access
    int at(size_t i) const { return (bits_[i >> 3] >> (i & 7)) & 1 ? +1 : -1; }
    const std::array<uint8_t, HDC_DIM / 8>& bits() const { return bits_; }
    void set_bits(const std::array<uint8_t, HDC_DIM / 8>& b) { bits_ = b; }

    // Serialization (raw 1250 bytes)
    std::string serialize() const;
    static HdcVec deserialize(const std::string& raw);

private:
    std::array<uint8_t, HDC_DIM / 8> bits_;  // 1250 bytes
};

// SHA256 wrapper
std::array<uint8_t, 32> sha256(const std::string& data);
std::string sha256_hex(const std::string& data);

}  // namespace thalia
