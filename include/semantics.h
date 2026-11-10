// Thalia - Distributional semantics from corpus co-occurrence statistics.
// The attack on the "semantically hollow HDC" problem (v0.3).
//
// Mechanism (BEAGLE-style holographic semantics, zero gradients):
//   1. Every distinct token gets a deterministic random INDEX vector:
//      HdcVec::from_seed("idx:" + token). No training.
//   2. Every sentence Thalia hears updates co-occurrence counts within a
//      +/- kWindow span. Pure counting.
//   3. A token's SEMANTIC vector is the weighted bundle of the index
//      vectors of its co-occurring words, plus a weak self-index component
//      so the word stays distinguishable from its sentence-mates.
//
// Consequence: "狗" and "犬" never appear together, but if they appear in
// similar company (草地/跑/追/球), their semantic vectors converge. Meaning
// = the company a word keeps. Still no training; the statistics live in
// their own append-only log (lang.log), so lexical knowledge outlives
// episodic memory pruning: you forget the day you learned a word, but not
// the word.
#pragma once
#include "hdc.h"
#include <map>
#include <string>
#include <vector>

namespace thalia {

class Semantics {
public:
    explicit Semantics(const std::string& store_dir);

    // Feed one sentence of natural language into the statistics.
    // weight scales the counts (corpus lines 1.0, innate text lower).
    // Punctuation tokens are ignored: they carry no lexical company.
    void observe(const std::string& text, double weight = 1.0);

    // Semantic vector for a token. NEVER fails: a never-observed word is
    // its own pure index vector ("an unheard word is just an arbitrary
    // symbol"), and context mass grows around it continuously — so the
    // encoding drifts smoothly as the corpus grows instead of jumping
    // between absent/present.
    HdcVec vec_of(const std::string& token) const;

    // Similarity between two tokens (0.5 = unrelated or unknown).
    double token_sim(const std::string& a, const std::string& b) const;

    bool   empty() const { return cooc_.empty(); }
    size_t vocab() const { return cooc_.size(); }
    size_t pairs() const;   // total distinct co-occurring pairs

    // Persistence: append-only sentence log ("weight\ttext" per line),
    // replayed through observe() at startup. flush() writes pending lines.
    void load();
    void flush();

    // Tunables.
    static constexpr int    kWindow     = 5;    // co-occurrence span (+/- tokens)
    static constexpr double kSelfWeight = 0.45; // self-index coefficient:
                                                // sqrt-mass scaling, keeps a
                                                // word distinct from its mates
                                                // while context dominates.

private:
    void observe_tokens(const std::vector<std::string>& toks, double weight);
    const HdcVec& index_vec(const std::string& token) const;  // cached

    std::map<std::string, std::map<std::string, double>> cooc_;  // w -> ctx word -> weight
    std::map<std::string, double> mass_;                         // total ctx weight per word
    std::string dir_;
    std::string log_path_;
    std::vector<std::pair<double, std::string>> pending_;         // not yet on disk

    mutable std::map<std::string, HdcVec> vec_cache_;   // token -> semantic vec
    mutable std::map<std::string, HdcVec> idx_cache_;   // token -> index vec
};

}  // namespace thalia
