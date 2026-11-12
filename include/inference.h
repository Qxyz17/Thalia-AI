// Thalia - Inference: facts extracted from heard language, and the
// deductions that follow from them.
//
// The Speaker can only recombine what it heard. This layer is what lets
// Thalia say something it was never told:
//
//   1. learn(text) pulls subject-predicate relations out of one sentence
//      and stores them as sparse triples:
//        A 是 B      -> IS(A, B)       (identity / class)
//        A 有 B      -> HAS(A, B)      (parts, properties)
//        A 会 V      -> CAN(A, V)      (abilities)
//        A 喜欢 B    -> LIKES(A, B)
//        A 在 B      -> AT(A, B)
//        A 需要 B    -> NEEDS(A, B)
//        A 用 B      -> USES(A, B)
//        A 比 B C    -> MORE(A, B, C)
//      Anything else with a verb-like tail falls back to DOES(A, tail).
//
//   2. ask(subject, relation) answers by chaining what it knows:
//        - direct match
//        - IS-transitivity: A 是 B, B 是 C  =>  A 是 C
//          plus property inheritance: A 是 B, B 会 V => A 会 V
//        - HAS/CAN inheritance through the IS chain
//      Every derived answer records the path that produced it, so the
//      CLI can show the reasoning instead of asserting a fact.
//
// Nothing here is learned by gradient descent: a fact is a string triple,
// and a deduction is a walk over the triples that are already stored.
#pragma once
#include "memory.h"
#include <map>
#include <set>
#include <string>
#include <vector>

namespace thalia {

class Inference {
public:
    // Fact relations. IS_NAME is the copula that also carries inheritance.
    enum Rel {
        IS,     // A 是 B
        HAS,    // A 有 B
        CAN,    // A 会 B
        LIKES,  // A 喜欢 B
        AT,     // A 在 B
        NEEDS,  // A 需要 B
        USES,   // A 用 B
        MORE,   // A 比 B 更/还/要 C -> MORE(A,B,C); the tail is kept
        DOES,   // fallback: A <verb phrase>
    };

    struct Triple {
        std::string a;
        std::string b;
        std::string c;   // only MORE uses it
        Rel rel = IS;
    };

    struct Answer {
        std::string text;                 // the deduced sentence
        std::vector<std::string> path;    // steps that produced it
        bool derived = false;             // true if not a direct fact
    };

    // Pull every relation out of one line of text.
    void learn(const std::string& text);

    // Replay every language-bearing memory record through learn(), so the
    // fact base survives a restart (same pattern as Semantics::load and
    // Speaker::rebuild: the log is the source of truth, not this object).
    void rebuild(const Memory& mem);

    // All answers Thalia can produce about `subject`, direct or derived.
    // Empty if it knows nothing about it.
    std::vector<Answer> ask(const std::string& subject) const;

    size_t facts() const { return triples_.size(); }
    size_t subjects() const;
    bool   known(const std::string& subject) const;

    // Relation name as printed by the CLI.
    static const char* rel_name(Rel rel);

private:
    // Direct triples in insertion order (a fact heard twice stays once).
    std::vector<Triple> triples_;
    std::set<std::string> seen_;                 // dedupe key
    std::map<std::string, std::vector<size_t>> by_subject_;

    void add(const std::string& a, Rel rel, const std::string& b,
             const std::string& c = "");
    std::vector<size_t> outgoing(const std::string& a, Rel rel) const;

    // Follow IS from `a` up to `depth` hops; returns the classes reached,
    // each as a step string ("a是b" for the first hop, "b是c" for the next).
    std::vector<std::string> is_steps(const std::string& a, int depth) const;
};

}  // namespace thalia
