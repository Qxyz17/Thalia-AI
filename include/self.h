// Thalia - Self vector and chemical state
// The "self" is a slowly-evolving HDC vector. Every perception is bound
// with self before being stored, so all memories carry the "I" tint.
#pragma once
#include "hdc.h"
#include "memory.h"
#include <string>

namespace thalia {

class Self {
public:
    // Initialize from a stable seed (e.g. birthdate + name + random nonce).
    explicit Self(const std::string& identity_seed);

    // Bind an arbitrary HDC vector with self: produces self XOR vec.
    // This is how perceptions get the "I am perceiving X" tint.
    HdcVec tint(const HdcVec& v) const;

    // Self evolves slowly toward the bound version of recent salient events.
    void evolve(const HdcVec& recent_salient, double alpha);

    // Chemical state lives with self (they're the body's modulators).
    ChemState& chem() { return chem_; }
    const ChemState& chem() const { return chem_; }

    // Reward / stress API
    void reward(double amount);          // raises dopamine, lowers cortisol
    void stress(double amount);          // raises cortisol
    void focus(double amount);           // raises serotonin
    void urgent(double amount);          // raises adrenaline

    // Decays chemicals toward baseline each tick.
    void tick();

    const HdcVec& vec() const { return vec_; }

private:
    HdcVec vec_;          // current self
    ChemState chem_;
};

}  // namespace thalia
