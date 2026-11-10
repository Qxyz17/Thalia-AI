// Thalia - Semantics implementation.
#include "semantics.h"
#include "tokenize.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#define thalia_mkdir(p) _mkdir(p)
#else
#define thalia_mkdir(p) ::mkdir((p), 0755)
#endif

namespace thalia {

static void ensure_dir(const std::string& path) {
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) {
            thalia_mkdir(cur.c_str());
        }
    }
}

Semantics::Semantics(const std::string& store_dir) : dir_(store_dir) {
    ensure_dir(dir_);
    log_path_ = dir_ + "/lang.log";
    load();
}

void Semantics::observe_tokens(const std::vector<std::string>& toks, double weight) {
    if (toks.empty() || weight <= 0) return;
    const size_t W = static_cast<size_t>(kWindow);
    for (size_t i = 0; i < toks.size(); ++i) {
        if (!is_word_token(toks[i])) continue;   // punctuation has no company
        size_t lo = (i > W) ? i - W : 0;
        size_t hi = std::min(toks.size() - 1, i + W);
        for (size_t j = lo; j <= hi; ++j) {
            if (j == i || !is_word_token(toks[j])) continue;
            cooc_[toks[i]][toks[j]] += weight;
            mass_[toks[i]] += weight;
        }
    }
    vec_cache_.clear();   // statistics changed; stale vectors must go
}

void Semantics::observe(const std::string& text, double weight) {
    auto toks = tokenize(text);
    if (toks.empty()) return;
    observe_tokens(toks, weight);
    pending_.push_back({weight, text});
}

const HdcVec& Semantics::index_vec(const std::string& token) const {
    auto it = idx_cache_.find(token);
    if (it != idx_cache_.end()) return it->second;
    HdcVec v = HdcVec::from_seed("idx:" + token);
    return idx_cache_.emplace(token, v).first->second;
}

HdcVec Semantics::vec_of(const std::string& token) const {
    auto cached = vec_cache_.find(token);
    if (cached != vec_cache_.end()) return cached->second;

    // sem(w) = kSelfWeight * sqrt(1 + mass) * idx(w)
    //        + sum over co-words v of count(w, v) * idx(v)
    // mass 0 (never heard in company) degrades to the pure index vector.
    // The sqrt keeps the self component sub-linear: heavy context mass
    // still dominates (meaning is the company), but the word never
    // collapses into an exact synonym of its sentence-mates.
    double mass = 0;
    auto mit = mass_.find(token);
    if (mit != mass_.end()) mass = mit->second;
    double w_self = kSelfWeight * std::sqrt(1.0 + mass);

    std::vector<std::pair<HdcVec, double>> parts;
    auto it = cooc_.find(token);
    if (it != cooc_.end()) {
        parts.reserve(it->second.size() + 1);
        for (const auto& [w, c] : it->second) {
            parts.push_back({index_vec(w), c});
        }
    }
    parts.push_back({index_vec(token), w_self});

    HdcVec v = HdcVec::bundle_weighted(parts);
    vec_cache_.emplace(token, v);
    return v;
}

double Semantics::token_sim(const std::string& a, const std::string& b) const {
    if (a == b) return 1.0;
    return HdcVec::sim(vec_of(a), vec_of(b));
}

size_t Semantics::pairs() const {
    size_t n = 0;
    for (const auto& [w, ctx] : cooc_) n += ctx.size();
    return n;
}

void Semantics::load() {
    std::ifstream f(log_path_);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        double w = std::atof(line.substr(0, tab).c_str());
        if (w <= 0) w = 1.0;
        // Count only; replayed lines must not be re-appended to the log.
        auto toks = tokenize(line.substr(tab + 1));
        observe_tokens(toks, w);
    }
}

void Semantics::flush() {
    if (pending_.empty()) return;
    std::ofstream f(log_path_, std::ios::app);
    if (!f) return;
    for (auto& [w, text] : pending_) {
        // Keep the log line-shaped: no tabs/newlines inside the text.
        std::string clean;
        clean.reserve(text.size());
        for (char c : text) {
            clean += (c == '\t' || c == '\n' || c == '\r') ? ' ' : c;
        }
        char wbuf[32];
        std::snprintf(wbuf, sizeof(wbuf), "%.4f", w);
        f << wbuf << '\t' << clean << '\n';
    }
    pending_.clear();
}

}  // namespace thalia
