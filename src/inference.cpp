// Thalia - Inference implementation.
#include "inference.h"
#include "tokenize.h"
#include <algorithm>

namespace thalia {

namespace {

// Marker tokens for the relation cues, compared as whole tokens.
const char* kIsWord    = "是";
const char* kWorseWord = "比";

// A token carries content (as opposed to punctuation).
bool content(const std::string& t) { return is_word_token(t); }

// Split on spaces. The shared tokenizer splits CJK per character, which
// erases word boundaries; the fact layer needs them ("太阳" as one word,
// not 太 + 阳). Space-separated input is the corpus convention.
std::vector<std::string> words(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    for (char c : s) {
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (!cur.empty()) { out.push_back(cur); cur.clear(); }
        } else {
            cur += c;
        }
    }
    if (!cur.empty()) out.push_back(cur);
    return out;
}

// Join tokens back into a phrase, no spaces (Chinese has none between
// characters, and the tokenizer splits CJK per character).
std::string join(const std::vector<std::string>& toks, size_t from, size_t to) {
    std::string s;
    for (size_t i = from; i < to && i < toks.size(); ++i) {
        if (!content(toks[i])) break;
        s += toks[i];
    }
    return s;
}

}  // namespace

const char* Inference::rel_name(Rel rel) {
    switch (rel) {
        case IS:    return "是";
        case HAS:   return "有";
        case CAN:   return "会";
        case LIKES: return "喜欢";
        case AT:    return "在";
        case NEEDS: return "需要";
        case USES:  return "用";
        case MORE:  return "比";
        case DOES:  return "做";
    }
    return "?";
}

void Inference::add(const std::string& a, Rel rel, const std::string& b,
                    const std::string& c) {
    if (a.empty() || b.empty()) return;
    std::string key = a + "|" + std::to_string(static_cast<int>(rel)) + "|" +
                      b + "|" + c;
    if (!seen_.insert(key).second) return;
    size_t idx = triples_.size();
    triples_.push_back({a, b, c, rel});
    by_subject_[a].push_back(idx);
}

size_t Inference::subjects() const { return by_subject_.size(); }

bool Inference::known(const std::string& subject) const {
    return by_subject_.count(subject) != 0;
}

void Inference::learn(const std::string& text) {
    auto toks = words(text);
    if (toks.size() < 3) return;

    // Locate the first relation cue. Everything before it is the subject.
    static const char* kCues[] = {"是", "有", "会", "喜欢", "需要", "用", "在", "比"};
    size_t cut = toks.size();
    std::string cue;
    for (size_t k = 0; k < toks.size(); ++k) {
        for (const char* c : kCues) {
            if (toks[k] == c) { cut = k; cue = c; break; }
        }
        if (cut != toks.size()) break;
    }
    if (cut == toks.size()) return;   // no relation: nothing to store

    // Subject: the content run immediately before the cue.
    size_t s0 = cut;
    while (s0 > 0 && content(toks[s0 - 1])) --s0;
    std::string subj = join(toks, s0, cut);
    if (subj.empty()) return;

    auto tail = [&](size_t from) { return join(toks, from, toks.size()); };
    size_t i = cut;   // index of the cue token

    if (cue == kIsWord) {
        std::string obj = tail(i + 1);
        if (obj.empty()) return;
        // "A 是 B C" where C is 的-phrase is still just identity.
        add(subj, IS, obj);
        return;
    }
    if (cue == "有") {
        std::string obj = tail(i + 1);
        add(subj, HAS, obj);
        return;
    }
    if (cue == "会") {
        std::string obj = tail(i + 1);
        add(subj, CAN, obj);
        return;
    }
    if (cue == "喜欢") {
        std::string obj = tail(i + 1);
        add(subj, LIKES, obj);
        return;
    }
    if (cue == "需要") {
        std::string obj = tail(i + 1);
        add(subj, NEEDS, obj);
        return;
    }
    if (cue == "用") {
        // "A 用 B 做 C" -> USES(A, B)
        std::string obj = join(toks, i + 1, toks.size());
        add(subj, USES, obj);
        return;
    }
    if (cue == "在") {
        std::string obj = tail(i + 1);
        add(subj, AT, obj);
        return;
    }
    if (cue == kWorseWord) {
        // "A 比 B 重" -> MORE(A, B, 重)
        size_t b0 = i + 1;
        size_t j = b0;
        while (j < toks.size() && content(toks[j])) ++j;
        std::string other = join(toks, b0, j);
        std::string attr = tail(j);
        // Drop a leading 更/还/要 filler.
        while (!attr.empty() && (attr.compare(0, 3, "更") == 0 ||
                                 attr.compare(0, 3, "还") == 0 ||
                                 attr.compare(0, 3, "要") == 0)) {
            attr = attr.substr(3);
        }
        add(subj, MORE, other, attr);
        return;
    }

    // Fallback: the rest of the clause is what the subject does.
    std::string tail_txt = tail(i);
    if (!tail_txt.empty()) add(subj, DOES, tail_txt);
}

void Inference::rebuild(const Memory& mem) {
    triples_.clear();
    seen_.clear();
    by_subject_.clear();
    for (const auto& [key, rec] : mem.all()) {
        if (rec.kind == "image") continue;   // file paths, not language
        if (rec.raw_full.empty()) continue;
        learn(rec.raw_full);
    }
}

std::vector<size_t> Inference::outgoing(const std::string& a, Rel rel) const {
    std::vector<size_t> out;
    auto it = by_subject_.find(a);
    if (it == by_subject_.end()) return out;
    for (size_t idx : it->second) {
        if (triples_[idx].rel == rel) out.push_back(idx);
    }
    return out;
}

std::vector<Inference::Answer> Inference::ask(const std::string& subject) const {
    std::vector<Answer> out;
    auto it = by_subject_.find(subject);
    if (it == by_subject_.end()) return out;

    // 1. Direct facts, in the order they were heard.
    for (size_t idx : it->second) {
        const Triple& t = triples_[idx];
        Answer a;
        if (t.rel == MORE) {
            a.text = t.a + "比" + t.b + t.c;
        } else if (t.rel == DOES) {
            a.text = t.a + t.b;
        } else {
            a.text = t.a + rel_name(t.rel) + t.b;
        }
        a.derived = false;
        out.push_back(a);
    }

    // 2. Inheritance: what the classes of `subject` can do / have / are.
    //    Walk the IS chain one hop at a time, so an inferred fact carries
    //    the whole derivation, not just its first step.
    std::set<std::string> seen;
    seen.insert(subject);
    std::string cur = subject;
    for (int hop = 0; hop < 3; ++hop) {
        auto outs = outgoing(cur, IS);
        if (outs.empty()) break;
        std::string cls = triples_[outs.front()].b;
        if (!seen.insert(cls).second) break;

        std::string step = cur + "是" + cls;
        auto cit = by_subject_.find(cls);
        if (cit != by_subject_.end()) {
            for (size_t idx : cit->second) {
                const Triple& t = triples_[idx];
                if (t.rel != CAN && t.rel != HAS && t.rel != IS) continue;
                Answer a;
                a.text = subject + rel_name(t.rel) + t.b;
                if (t.rel == IS) a.text = subject + "是" + t.b;
                a.path.push_back(step);
                a.path.push_back(t.a + (t.rel == DOES ? "" : rel_name(t.rel)) + t.b);
                a.derived = true;
                out.push_back(a);
            }
        }
        cur = cls;
    }
    return out;
}

}  // namespace thalia
