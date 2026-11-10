// Thalia - Content-addressable memory store
// Each event = SHA256-keyed record with HDC value + metadata.
// No training. Pure write/retrieve/forget.
#pragma once
#include "hdc.h"
#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <vector>

namespace thalia {

struct ChemState {
    double dopamine  = 0.5;   // reward signal; high -> boost related memory weights
    double cortisol  = 0.3;   // stress; high -> lower retrieval threshold (more divergent)
    double serotonin = 0.5;   // high -> focused single-line, low -> broad association
    double adrenaline = 0.2;  // urgency; high -> skip deep retrieval
};

struct MemoryRecord {
    std::string key;          // SHA256 hex of (raw_data + timestamp + seq)
    std::string kind;         // "percept" | "heard" | "image" | "thought" | "action" | "self-update"
    std::string raw_preview;  // first 64 chars (display only)
    std::string raw_full;     // full raw text (language material for the Speaker)
    HdcVec     vec;           // EXACT route: surface + identity anchor (episodic fingerprint)
    HdcVec     avec;          // ASSOCIATION route: surface + co-occurrence semantics
    double     weight;        // current weight, decays over time, boosted by dopamine
    int64_t    ts_ms;         // creation timestamp
    int        hits = 0;      // retrieval count
    std::vector<std::string> links;  // related keys (graph edges)
};

class Memory {
public:
    explicit Memory(const std::string& store_dir);

    // Record a new event. Returns its key.
    std::string record(const std::string& kind,
                       const std::string& raw,
                       const HdcVec& vec,
                       const HdcVec& avec,
                       const std::vector<std::string>& links = {});

    // Top-k nearest neighbors over BOTH routes. Each hit reports the
    // effective similarity (max of the two routes, chemically modulated)
    // plus each route's raw value, so callers can tell "I remember this
    // exactly" (sim_exact high) from "this reminds me of something"
    // (sim_assoc high). k and threshold modulated by chem.
    struct Hit {
        std::string key;
        double sim;         // effective (max route, boosted)
        double sim_exact;   // raw exact-route similarity
        double sim_assoc;   // raw association-route similarity
        const MemoryRecord* rec;
    };
    std::vector<Hit> retrieve(const HdcVec& q, const HdcVec& qa,
                              size_t k, const ChemState& chem);

    // Decay pass: weights decay, low-weight+low-hit records get pruned.
    void decay(double factor);

    // Offline consolidation ("sleep"): replay the whole store, merge very
    // similar records into prototypes, reinforce frequently-hit ones, prune
    // weak unused ones. Inspired by systems consolidation (hippocampal
    // replay during sleep -> long-term neocortical traces).
    // Returns a report of what happened.
    struct SleepReport {
        size_t considered = 0;
        size_t merged     = 0;   // victims absorbed into survivors
        size_t reinforced = 0;   // high-hit records got stronger
        size_t pruned     = 0;   // weak unused records deleted
        size_t re_encoded = 0;   // set by Engine::sleep: vectors re-derived
                                 // with the current semantic model
    };
    SleepReport consolidate(double merge_threshold);

    // Persistence: append-only log + periodic compact.
    void flush();
    void load();

    // Diagnostics
    size_t size() const { return records_.size(); }
    const std::map<std::string, MemoryRecord>& all() const { return records_; }

    // Update an existing record's weight (used by dopamine feedback)
    void boost(const std::string& key, double delta);

    // Reconsolidation hook: rewrite records in place (keys and metadata
    // stay; typically only `vec` is re-derived). Used by Engine::sleep to
    // re-encode text memories with the CURRENT semantic model -- old
    // episodes re-understood with today's vocabulary.
    void reencode(const std::function<void(MemoryRecord&)>& fn);

private:
    std::string dir_;
    std::map<std::string, MemoryRecord> records_;
    std::string log_path_;
};

}  // namespace thalia
