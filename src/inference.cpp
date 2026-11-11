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
    auto toks = tokenize(text);
    if (toks.size() < 3) return;

    // Subject is the leading content run.
    size_t i = 0;
    while (i < toks.size() && !content(toks[i])) ++i;
    size_t s0 = i;
    while (i < toks.size() && content(toks[i])) ++i;
    std::string subj = join(toks, s0, i);
    if (subj.empty() || i >= toks.size()) return;

    // join() stops at the first non-content token, so the trailing period
    // is already excluded.
    auto tail = [&](size_t from) { return join(toks, from, toks.size()); };

    std::string cue = toks[i];

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

std::vector<size_t> Inference::outgoing(const std::string& a, Rel rel) const {
    std::vector<size_t> out;
    auto it = by_subject_.find(a);
    if (it == by_subject_.end()) return out;
    for (size_t idx : it->second) {
        if (triples_[idx].rel == rel) out.push_back(idx);
    }
    return out;
}

std::vector<std::string> Inference::is_chain(const std::string& a, int depth) const {
    std::vector<std::string> chain;
    std::set<std::string> seen{a};
    std::string cur = a;
    for (int d = 0; d < depth; ++d) {
        auto outs = outgoing(cur, IS);
        if (outs.empty()) break;
        std::string next = triples_[outs.front()].b;
        if (!seen.insert(next).second) break;
        chain.push_back(next);
        cur = next;
    }
    return chain;
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
    auto chain = is_chain(subject, 3);
    for (const auto& cls : chain) {
        auto cit = by_subject_.find(cls);
        if (cit == by_subject_.end()) continue;
        for (size_t idx : cit->second) {
            const Triple& t = triples_[idx];
            // Only inheritable relations: properties, not opinions about
            // the class as an individual.
            if (t.rel != CAN && t.rel != HAS && t.rel != IS) continue;
            Answer a;
            a.text = subject + rel_name(t.rel) + t.b;
            a.path.push_back(subject + "是" + cls);
            a.path.push_back(t.a + rel_name(t.rel) + t.b);
            a.derived = true;
            out.push_back(a);
        }
    }
    return out;
}

std::vector<Inference::Answer> Inference::ask(const std::string& subject,
                                              Rel rel) const {
    std::vector<Answer> out;
    auto it = by_subject_.find(subject);
    if (it != by_subject_.end()) {
        for (size_t idx : it->second) {
            const Triple& t = triples_[idx];
            if (t.rel != rel) continue;
            Answer a;
            a.text = (rel == DOES) ? t.a + t.b : t.a + rel_name(rel) + t.b;
            out.push_back(a);
        }
    }
    // Then anything the classes imply for this relation.
    for (const auto& cls : is_chain(subject, 3)) {
        auto cit = by_subject_.find(cls);
        if (cit == by_subject_.end()) continue;
        for (size_t idx : cit->second) {
            const Triple& t = triples_[idx];
            if (t.rel != rel) continue;
            Answer a;
            a.text = subject + rel_name(rel) + t.b;
            a.path.push_back(subject + "是" + cls);
            a.path.push_back(t.a + rel_name(rel) + t.b);
            a.derived = true;
            out.push_back(a);
        }
    }
    return out;
}

}  // namespace thalia
