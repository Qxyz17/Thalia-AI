// Thalia - Self implementation
#include "self.h"
#include <algorithm>
#include <chrono>

namespace thalia {

Self::Self(const std::string& identity_seed)
    : vec_(HdcVec::from_seed("self:" + identity_seed)) {}

HdcVec Self::tint(const HdcVec& v) const {
    return HdcVec::bind(vec_, v);
}

void Self::evolve(const HdcVec& recent_salient, double alpha) {
    // The salient event is "what just happened to me" — bind it with current
    // self to form "I-experienced-X", then nudge self toward that.
    HdcVec experience = HdcVec::bind(vec_, recent_salient);
    vec_.nudge(experience, alpha);
}

void Self::reward(double amount) {
    chem_.dopamine = std::min(1.0, chem_.dopamine + amount);
    chem_.cortisol = std::max(0.0, chem_.cortisol - amount * 0.3);
}
void Self::stress(double amount) {
    chem_.cortisol = std::min(1.0, chem_.cortisol + amount);
}
void Self::focus(double amount) {
    chem_.serotonin = std::min(1.0, chem_.serotonin + amount);
}
void Self::urgent(double amount) {
    chem_.adrenaline = std::min(1.0, chem_.adrenaline + amount);
}

void Self::tick() {
    // Decay toward baseline (0.5 dopamine / 0.3 cortisol / 0.5 serotonin / 0.2 adrenaline).
    chem_.dopamine   += (0.5 - chem_.dopamine)   * 0.05;
    chem_.cortisol   += (0.3 - chem_.cortisol)   * 0.05;
    chem_.serotonin  += (0.5 - chem_.serotonin)  * 0.05;
    chem_.adrenaline += (0.2 - chem_.adrenaline) * 0.08;
}

}  // namespace thalia
