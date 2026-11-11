// Thalia - Engine implementation
#include "engine.h"
#include "encoder.h"
#include "tokenize.h"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>

namespace thalia {

constexpr double DIRECT_HIT_THRESHOLD = 0.85;  // sim above this = "I know this"

// The thing a sentence is about, when it asks a question about something
// Thalia has facts on. Leading interrogatives are dropped; the first
// content run left is the topic. "猫 会 什么" -> "猫".
static std::string question_topic(const std::string& raw) {
    auto toks = tokenize(raw);
    // Drop a leading question form.
    static const char* kLeading[] = {"什么", "为什么", "怎么", "怎样", "谁",
                                     "哪", "是不是", "会不会", "请", "问",
                                     "你知道", "告诉我"};
    size_t i = 0;
    for (; i < toks.size(); ++i) {
        if (!is_word_token(toks[i])) continue;
        bool dropped = false;
        for (const char* w : kLeading) {
            if (toks[i] == w) { dropped = true; break; }
        }
        if (!dropped) break;
    }
    if (i >= toks.size()) return "";
    std::string topic;
    for (; i < toks.size() && is_word_token(toks[i]); ++i) topic += toks[i];
    return topic;
}

Engine::Engine(const std::string& store_dir, const std::string& identity_seed)
    : self_(identity_seed), memory_(store_dir), sem_(store_dir) {

    // Default lexicon: a few core self-concepts so clean-up can name them.
    add_lexicon_entry("self",        "self:" + identity_seed);
    add_lexicon_entry("unknown",     "lex:unknown");
    add_lexicon_entry("greeting",    "lex:greeting");
    add_lexicon_entry("identity",    "lex:identity");
    add_lexicon_entry("memory_query","lex:memory_query");
    add_lexicon_entry("image_seen",  "lex:image_seen");

    // First boot only: seed the lexical statistics with the innate reflex
    // text at low weight, so the agent is not semantically blind to its
    // own name before it has heard anything. (lang.log missing = virgin.)
    if (sem_.empty()) {
        sem_.observe(Speaker::kInnateText, 0.25);
        sem_.flush();
    }

    // Build the voice from whatever memory was loaded.
    voice_.rebuild(memory_);
}

void Engine::load_lexicon(const std::string& path) {
    std::ifstream f(path);
    if (!f) return;
    std::string line;
    while (std::getline(f, line)) {
        if (line.empty() || line[0] == '#') continue;
        auto sep = line.find('|');
        if (sep == std::string::npos) continue;
        std::string name = line.substr(0, sep);
        std::string seed = line.substr(sep + 1);
        add_lexicon_entry(name, seed);
    }
}

void Engine::add_lexicon_entry(const std::string& name, const std::string& seed) {
    lexicon_.push_back({name, HdcVec::from_seed(seed)});
}

std::string Engine::clean_up(const HdcVec& v, double threshold) const {
    std::string best_name;
    double best_sim = 0;
    for (const auto& e : lexicon_) {
        double s = HdcVec::sim(v, e.vec);
        if (s > best_sim) { best_sim = s; best_name = e.name; }
    }
    return best_sim >= threshold ? best_name : "";
}

Response Engine::perceive(const std::string& kind, const std::string& raw) {
    Response resp;

    // 1. Encode input on both routes:
    //    exact:  surface + identity anchor (stable episodic fingerprint)
    //    assoc:  surface + co-occurrence semantics (words in similar company)
    HdcVec vec_in, vec_in_assoc;
    if (kind == "image") {
        vec_in = Encoder::image_file(raw);
        vec_in_assoc = vec_in;   // images have no language semantics
    } else {
        vec_in = Encoder::text(raw);
        vec_in_assoc = Encoder::text_assoc(raw, sem_);
    }

    // 2. Tint with self
    HdcVec vec_tinted = self_.tint(vec_in);
    HdcVec avec_tinted = self_.tint(vec_in_assoc);

    // 3. Retrieve from memory over both routes
    //    (k modulated by serotonin/adrenaline)
    size_t k = static_cast<size_t>(5 + 5 * (1.0 - self_.chem().serotonin));
    if (self_.chem().adrenaline > 0.6) k = std::max<size_t>(1, k / 2);
    auto hits = memory_.retrieve(vec_tinted, avec_tinted, k, self_.chem());

    // 4. Compose response
    std::ostringstream trace;
    trace << "[k=" << k << " hits=" << hits.size();
    if (!hits.empty()) {
        trace << " exact=" << hits[0].sim_exact << " assoc=" << hits[0].sim_assoc;
    }

    if (!hits.empty() && hits[0].sim_exact >= DIRECT_HIT_THRESHOLD) {
        // Direct hit: we've seen this exact thing before. Quote it verbatim.
        resp.direct_hit = true;
        resp.best_sim = hits[0].sim;
        resp.cited_keys.push_back(hits[0].key);
        const MemoryRecord* r = hits[0].rec;
        if (r->kind == "image") {
            resp.text = "I've seen this image before. (weight=" +
                        std::to_string(r->weight).substr(0,4) +
                        ", hits=" + std::to_string(r->hits) + ")";
        } else if (r->kind == "heard") {
            resp.text = "I remember: " + r->raw_full;
        } else {
            resp.text = "I remember: " + r->raw_full;
        }
        trace << " direct_hit=" << hits[0].sim_exact;
        // Reward slightly for confident retrieval
        self_.reward(0.05);
    } else if (kind == "image") {
        // Images never go through the language voice.
        resp.direct_hit = false;
        resp.best_sim = hits.empty() ? 0 : hits[0].sim;
        if (!hits.empty()) {
            resp.cited_keys.push_back(hits[0].key);
            resp.text = "Image perceived. Closest memory sim=" +
                        std::to_string(hits[0].sim).substr(0,4);
        } else {
            resp.text = "New image. Storing under fresh key.";
        }
        trace << " image";
    } else {
        // Text response: SPEAK from memory (n-gram walk), if the voice exists.
        std::string lower_raw = raw;
        for (auto& c : lower_raw) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));

        // Seeds: retrieved memories (most relevant phrasing first) + the input.
        std::vector<std::string> seeds;
        for (const auto& h : hits) {
            seeds.push_back(h.rec->raw_full);
            resp.cited_keys.push_back(h.key);
        }
        seeds.push_back(raw);

        // Intent seeds: innate reflex text biases the walk's starting point.
        bool identity_q = lower_raw.find("你是谁") != std::string::npos ||
                          lower_raw.find("你叫什么") != std::string::npos ||
                          lower_raw.find("who are you") != std::string::npos ||
                          lower_raw.find("your name") != std::string::npos;
        bool greeting_q = lower_raw.find("你好") != std::string::npos ||
                          lower_raw.find("hello") != std::string::npos ||
                          lower_raw == "hi" || lower_raw.rfind("hi ", 0) == 0 ||
                          lower_raw.find("早上好") != std::string::npos;
        if (identity_q) seeds.push_back("我是 Thalia 。 我是一个小小的非神经智能体 。");
        if (greeting_q) seeds.push_back("hello . ");

        std::string speech = voice_.speak(seeds, self_.chem(), 40);
        resp.best_sim = hits.empty() ? 0 : hits[0].sim;
        resp.direct_hit = false;

        // Deduction first: if the input names something Thalia holds facts
        // about, answer from the fact chain instead of walking n-grams.
        // This is the only path that can produce a sentence it never heard.
        bool deduced = false;
        std::string topic = question_topic(raw);
        if (!topic.empty()) {
            auto answers = infer_.ask(topic);
            if (!answers.empty()) {
                // Prefer a derived answer: that is the one worth showing.
                const Inference::Answer* pick = &answers.front();
                for (const auto& a : answers) {
                    if (a.derived) { pick = &a; break; }
                }
                resp.text = pick->text;
                trace << " infer=" << (pick->derived ? "derived" : "fact")
                      << " facts=" << answers.size();
                if (!pick->path.empty()) {
                    trace << " via=";
                    for (size_t i = 0; i < pick->path.size(); ++i) {
                        if (i) trace << ",";
                        trace << pick->path[i];
                    }
                }
                self_.reward(0.05);
                deduced = true;
            }
        }

        if (deduced) {
            // answered above
        } else if (!speech.empty()) {
            resp.text = speech;
            trace << " voice=yes";
            if (identity_q) self_.focus(0.05);
        } else if (!hits.empty()) {
            // No language learned yet, but we have associations.
            resp.text = "I associate this with: " + hits[0].rec->raw_preview +
                        " (sim=" + std::to_string(hits[0].sim).substr(0,4) + ")";
            trace << " voice=no best=" << hits[0].sim;
        } else {
            resp.text = "I have no memory of this. Recording now.";
            trace << " no_hits voice=no";
            self_.stress(0.05);
        }
    }

    // 5. Record this perception into memory, linking to retrieved hits.
    //    Text input is language too — the voice AND the lexical statistics
    //    learn from conversation.
    std::vector<std::string> links;
    for (const auto& k : resp.cited_keys) links.push_back(k);
    std::string rec_kind = (kind == "image") ? "image" : "percept";
    std::string my_key = memory_.record(rec_kind, raw, vec_tinted, avec_tinted, links);
    resp.cited_keys.insert(resp.cited_keys.begin(), my_key);
    if (rec_kind == "percept") {
        sem_.observe(raw, 1.0);          // conversation feeds the statistics
        voice_.rebuild(memory_);        // cheap at prototype scale
    }

    // 6. Evolve self from this round's salient vector (the input itself).
    double alpha = 0.02 + 0.03 * (self_.chem().dopamine - 0.5);
    if (alpha < 0.005) alpha = 0.005;
    self_.evolve(vec_in, alpha);

    self_.tick();

    trace << "]";
    resp.trace = trace.str();
    return resp;
}

void Engine::tick() {
    self_.tick();
    memory_.decay(0.999);
}

Memory::SleepReport Engine::sleep(double merge_threshold) {
    // 1. Consolidate the store (merge/reinforce/prune).
    auto rep = memory_.consolidate(merge_threshold);

    // 2. Reconsolidation: re-derive every text memory's vector with the
    //    CURRENT semantic model and the CURRENT self. An episode recorded
    //    before the agent knew a word gets re-understood once it does —
    //    the systems-consolidation story, literally: hippocampal traces
    //    are re-written in neocortical (semantic) form during sleep.
    memory_.reencode([this, &rep](MemoryRecord& r) {
        if (r.kind == "image" || r.raw_full.empty()) return;
        r.vec   = self_.tint(Encoder::text(r.raw_full));
        r.avec  = self_.tint(Encoder::text_assoc(r.raw_full, sem_));
        ++rep.re_encoded;
    });

    // 3. Chemical reset to a rested baseline. Sleep is when the body
    //    stops being stressed / urgent; reward state partially carries over.
    auto& c = self_.chem();
    c.dopamine   = 0.5 + (c.dopamine - 0.5) * 0.5;
    c.cortisol   = 0.25;
    c.serotonin  = 0.6;
    c.adrenaline = 0.1;

    // 4. Rebuild the language tables (sleep consolidates language too).
    voice_.rebuild(memory_);

    // 5. Persist the consolidated state.
    sem_.flush();
    memory_.flush();
    return rep;
}

size_t Engine::learn_file(const std::string& path) {
    std::ifstream f(path);
    if (!f) return 0;
    std::string line;
    size_t n = 0;
    while (std::getline(f, line)) {
        // Trim CR and leading/trailing spaces.
        while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
        size_t b = line.find_first_not_of(' ');
        if (b == std::string::npos) continue;
        line = line.substr(b);
        if (line.empty() || line[0] == '#') continue;

        HdcVec v = Encoder::text(line);
        HdcVec av = Encoder::text_assoc(line, sem_);
        memory_.record("heard", line, self_.tint(v), self_.tint(av));
        sem_.observe(line, 1.0);
        infer_.learn(line);
        ++n;
    }
    if (n > 0) {
        voice_.rebuild(memory_);
        sem_.flush();   // corpus goes to disk immediately
    }
    return n;
}

void Engine::flush() {
    sem_.flush();
    memory_.flush();
}

void Engine::feedback_good(const std::string& key, double amount) {
    memory_.boost(key, amount);
    self_.reward(amount);
}

void Engine::feedback_bad(const std::string& key, double amount) {
    memory_.boost(key, -amount);
    self_.stress(amount);
}

}  // namespace thalia
