// Thalia - Speaker: language generation from its OWN memory corpus.
// No neural network, no gradient descent. The mechanism:
//   1. rebuild(): count word-level n-grams over everything the agent has
//      heard (memory records' raw text). Weights = memory weights, so
//      well-rewarded memories dominate the phrasing.
//   2. speak(): seed a Markov walk from the current context, generate a
//      sentence. Chemicals modulate the walk:
//        - serotonin high  -> greedy picks (focused, repetitive)
//        - serotonin low   -> weighted random (loose association)
//        - cortisol        -> (reserved: more divergence when stuck)
//   The agent can only say what it has heard. Before it learns any
//   language, speak() returns "" (honest silence).
//   Randomness is SELF-random (v0.3): the RNG is seeded from the
//   millisecond wall clock hashed through SHA256 -- no fixed seed, so
//   the same context at a different moment yields a different utterance.
//   The context still steers where the walk starts; entropy only rules
//   how it branches.
#pragma once
#include "memory.h"
#include "tokenize.h"
#include <map>
#include <string>
#include <vector>

namespace thalia {

class Speaker {
public:
    Speaker();  // innate corpus is mixed in at rebuild() with low weight

    // Recount n-grams from all language-bearing memory records
    // (skips kind == "image"; those store file paths, not language).
    void rebuild(const Memory& mem);

    // Generate one utterance. seed_texts bias the starting point (retrieved
    // memories + the user's input). Returns "" if nothing has been learned.
    std::string speak(const std::vector<std::string>& seed_texts,
                      const ChemState& chem, size_t max_tokens);

    bool learned() const { return learned_; }
    size_t vocab() const { return uni_.size(); }

    // Innate "babble" corpus: built-in reflex sentences, mixed at low
    // weight so early speech has something to draw on beyond the corpus.
    static const char* kInnateText;

private:
    void add_text(const std::string& text, double weight);

    // successor tables: context -> (next token -> accumulated weight)
    std::map<std::pair<std::string, std::string>, std::map<std::string, double>> tri_;
    std::map<std::string, std::map<std::string, double>> bi_;
    std::map<std::string, double> uni_;
    std::vector<std::pair<std::string, double>> starts_;  // sentence starters
    bool learned_ = false;   // any real (non-innate) language seen yet
};

}  // namespace thalia
