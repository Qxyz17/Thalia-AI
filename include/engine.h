// Thalia - Engine: orchestrates Self + Memory + Encoder.
// The "think" loop:
//   1. encode input -> vec_in
//   2. tint with self -> vec_tinted
//   3. retrieve top-k from memory (k & threshold modulated by chem)
//   4. compose a response: bundle of top-k vecs, cleaned up against a small
//      lexicon of known concepts; or echo a direct hit if similarity is high
//   5. record (input, response) into memory with link to retrieved hits
//   6. evolve self from this round's salient vector
#pragma once
#include "encoder.h"
#include "hdc.h"
#include "memory.h"
#include "self.h"
#include "semantics.h"
#include "speaker.h"
#include <string>
#include <vector>

namespace thalia {

struct Response {
    std::string text;
    std::vector<std::string> cited_keys;   // memory keys that fired
    double     best_sim;
    bool       direct_hit;                 // true if best_sim > DIRECT_HIT_THRESHOLD
    std::string trace;                     // debug-ish: short reasoning trace
};

class Engine {
public:
    Engine(const std::string& store_dir, const std::string& identity_seed);

    // Main entry. kind = "text" | "image". Returns a Response.
    Response perceive(const std::string& kind, const std::string& raw);

    // Maintenance
    void tick();
    void flush();

    // Offline consolidation ("sleep"). Merges similar memories into
    // prototypes, reinforces replayed ones, prunes weak ones, resets
    // chemicals to rested baseline, RE-ENCODES every text memory with the
    // current semantic model (reconsolidation: old episodes re-understood
    // with today's vocabulary), rebuilds the language tables, persists.
    // Returns the report.
    Memory::SleepReport sleep(double merge_threshold = 0.80);

    // Feed language: read a text file, one sentence per line, and record
    // each line as a "heard" memory (self-tinted). Rebuilds the voice.
    // Returns the number of lines learned.
    size_t learn_file(const std::string& path);

    // The voice (n-gram speaker built from memory).
    Speaker& voice() { return voice_; }

    // The lexical statistics (co-occurrence semantics).
    Semantics& semantics() { return sem_; }

    Self&    self()    { return self_; }
    Memory&  memory()  { return memory_; }

    // A small lexicon of named concepts for clean-up. Loaded from a file
    // at startup; each line is "name|seed_string". Lets the engine translate
    // a computed HDC vector back to a label.
    void load_lexicon(const std::string& path);
    void add_lexicon_entry(const std::string& name, const std::string& seed);

    // Direct feedback: human says "this response was good" -> dopamine boost
    void feedback_good(const std::string& key, double amount = 0.3);
    void feedback_bad (const std::string& key, double amount = 0.3);

private:
    Self     self_;
    Memory   memory_;
    Semantics sem_;
    Speaker  voice_;

    struct LexEntry { std::string name; HdcVec vec; };
    std::vector<LexEntry> lexicon_;

    // Cleanup: given a noisy HDC vector, find the closest lexicon entry
    // whose similarity exceeds threshold; return its name (or empty).
    std::string clean_up(const HdcVec& v, double threshold) const;
};

}  // namespace thalia
