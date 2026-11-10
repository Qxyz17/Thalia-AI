// Thalia - Speaker implementation
#include "speaker.h"
#include "memory.h"
#include "tokenize.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <functional>
#include <random>

namespace thalia {

// Built-in reflex sentences (mixed at low weight). The agent still refuses
// to speak until it has actually learned some language from the world.
const char* Speaker::kInnateText =
    "我是 Thalia 。 我是一个小小的非神经智能体 。 我有一个自我向量 。 "
    "我有四种化学信号 。 我不使用梯度下降 。 我靠积累经验活着 。 "
    "hello . i am thalia . i am a small non neural agent . "
    "i have a self vector and chemical signals . "
    "i do not use gradient descent . i accumulate experience .";

// --- Speaker ---

Speaker::Speaker() {}

void Speaker::add_text(const std::string& text, double weight) {
    auto toks = tokenize(text);
    if (toks.empty()) return;
    starts_.push_back({toks[0], weight});
    for (size_t i = 0; i < toks.size(); ++i) {
        uni_[toks[i]] += weight;
        if (i + 1 < toks.size()) {
            bi_[toks[i]][toks[i + 1]] += weight;
        }
        if (i + 2 < toks.size()) {
            tri_[{toks[i], toks[i + 1]}][toks[i + 2]] += weight;
        }
    }
}

void Speaker::rebuild(const Memory& mem) {
    tri_.clear();
    bi_.clear();
    uni_.clear();
    starts_.clear();
    learned_ = false;

    // 1. Language the agent actually heard (full weight).
    for (const auto& [key, rec] : mem.all()) {
        if (rec.kind == "image") continue;          // file paths, not language
        if (rec.raw_full.empty()) continue;
        double w = std::max(0.2, rec.weight);
        add_text(rec.raw_full, w);
        learned_ = true;
    }

    // 2. Innate reflex sentences (low weight, always present).
    add_text(kInnateText, 0.3);
}

// FNV-1a hash (hashes the extracted digit string into a 64-bit seed).
static uint64_t fnv1a(const std::string& s) {
    uint64_t h = 1469598103934665603ULL;
    for (unsigned char c : s) {
        h ^= c;
        h *= 1099511628211ULL;
    }
    return h;
}

// --- Self-random entropy source (v0.3) ---
// Per spec: no forced seed. Take the local wall clock's millisecond
// timestamp, SHA256 it, pull the decimal digits out of the hex digest,
// and use that digit string as the seed material. A process-wide
// monotonic counter keeps two calls inside the same millisecond distinct
// (same trick as the memory-key seq). Same context, different moment ->
// different utterance.
static uint64_t self_random_seed() {
    namespace sc = std::chrono;
    int64_t ms = sc::duration_cast<sc::milliseconds>(
                     sc::system_clock::now().time_since_epoch()).count();
    static std::atomic<uint64_t> seq{0};
    std::string digest = sha256_hex("voice:" + std::to_string(ms) + ":" +
                                    std::to_string(seq.fetch_add(1)));
    std::string digits;
    for (char c : digest) {
        if (c >= '0' && c <= '9') digits += c;
        if (digits.size() >= 19) break;
    }
    if (digits.size() < 8) digits = digest;   // paranoid fallback
    return fnv1a(digits);
}

std::string Speaker::speak(const std::vector<std::string>& seed_texts,
                            const ChemState& chem, size_t max_tokens) {
    if (!learned_ || uni_.empty() || starts_.empty()) return "";

    // Self-random RNG: entropy comes from the clock, not the context.
    std::mt19937_64 rng(self_random_seed());
    auto unit = [&rng]() { return (rng() >> 11) / 9007199254740992.0; };  // [0,1)

    // --- Pick a starting context ---
    // Preference: a bigram from the tail of a seed text that exists in
    // tri_; then a unigram from a seed tail in bi_; then a random starter.
    std::vector<std::string> ctx;  // 1-2 tokens of context
    bool found = false;
    for (auto it = seed_texts.rbegin(); it != seed_texts.rend() && !found; ++it) {
        auto toks = tokenize(*it);
        if (toks.size() >= 2) {
            std::pair<std::string, std::string> bg{toks[toks.size()-2], toks[toks.size()-1]};
            if (tri_.count(bg)) { ctx = {bg.first, bg.second}; found = true; }
        }
    }
    if (!found) {
        for (auto it = seed_texts.rbegin(); it != seed_texts.rend() && !found; ++it) {
            auto toks = tokenize(*it);
            if (!toks.empty() && bi_.count(toks.back())) { ctx = {toks.back()}; found = true; }
        }
    }
    if (!found) {
        // weighted random starter
        double total = 0;
        for (auto& [t, w] : starts_) total += w;
        double r = unit() * total;
        for (auto& [t, w] : starts_) {
            r -= w;
            if (r <= 0) { ctx = {t}; found = true; break; }
        }
        if (!found) return "";
    }

    // --- Walk ---
    std::vector<std::string> out;
    double greed = std::min(0.95, chem.serotonin * 0.85);
    const size_t MIN_TOKENS = 4;

    while (out.size() < max_tokens) {
        // Choose next token
        const std::map<std::string, double>* succ = nullptr;
        if (ctx.size() == 2) {
            auto it = tri_.find({ctx[0], ctx[1]});
            if (it != tri_.end()) succ = &it->second;
        }
        if (!succ && !ctx.empty()) {
            auto it = bi_.find(ctx.back());
            if (it != bi_.end()) succ = &it->second;
        }
        if (!succ || succ->empty()) break;

        std::string next;
        if (unit() < greed) {
            // greedy: argmax weight
            double best = -1;
            for (auto& [t, w] : *succ) if (w > best) { best = w; next = t; }
        } else {
            double total = 0;
            for (auto& [t, w] : *succ) total += w;
            double r = unit() * total;
            for (auto& [t, w] : *succ) {
                r -= w;
                if (r <= 0) { next = t; break; }
            }
            if (next.empty() && !succ->empty()) next = succ->begin()->first;
        }

        // Anti-stutter: don't repeat the same token 3x in a row.
        size_t n = out.size();
        if (n >= 2 && out[n-1] == next && out[n-2] == next) {
            // resample once, greedily excluding that token
            double best = -1;
            for (auto& [t, w] : *succ) {
                if (t == next) continue;
                if (w > best) { best = w; next = t; }
            }
            if (best < 0) break;
        }

        out.push_back(next);
        if (is_sentence_ender(next) && out.size() >= MIN_TOKENS) break;

        // slide context
        if (ctx.size() == 2) { ctx[0] = ctx[1]; ctx[1] = next; }
        else ctx.push_back(next);
    }

    if (out.size() < 2) return "";

    // Don't start an utterance with a sentence-ender (walk quirk).
    size_t lead = 0;
    while (lead + 1 < out.size() && is_sentence_ender(out[lead])) ++lead;
    if (lead > 0) out.erase(out.begin(), out.begin() + lead);
    if (out.size() < 2) return "";

    // --- Join tokens ---
    std::string speech;
    for (size_t i = 0; i < out.size(); ++i) {
        if (i > 0) {
            auto is_ascii_word_start = [](const std::string& t) {
                if (t.empty()) return false;
                unsigned char c = static_cast<unsigned char>(t[0]);
                return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
                       (c >= '0' && c <= '9') || c == '_' || c == '\'';
            };
            if (is_ascii_word_start(out[i]) && is_ascii_word_start(out[i-1])) {
                speech += ' ';
            }
        }
        speech += out[i];
    }
    return speech;
}

}  // namespace thalia
