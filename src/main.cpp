// Thalia - CLI entry point
// Usage:
//   thalia [--store DIR] [--seed SEED] [--lex FILE]
//   Interactive REPL: type text, or "img:<path>" to perceive an image.
//   Special commands:
//     :state        - show self vector sim + chem
//     :mem          - show memory size + recent records
//     :sim <a> <b>  - co-occurrence similarity between two tokens
//     :facts <主题>  - facts stored about a subject
//     :why <主题>    - deductions, with the chain that produced them
//     :good <key>   - mark last response's first cited key as good
//     :bad <key>    - mark as bad
//     :quit
#include "engine.h"
#include "hdc.h"
#include <chrono>
#include <iostream>
#include <sstream>
#include <string>

#ifdef _WIN32
#include <windows.h>
#endif

// Console I/O is UTF-8 throughout (the store format too). On Windows the
// console defaults to the OEM code page (936 here) and SetConsoleCP does
// not help when stdin is redirected or when the shell keeps the old page,
// so lines are also repaired byte-wise: a line that is not valid UTF-8 is
// decoded as the active ANSI code page and re-encoded as UTF-8.
static bool is_valid_utf8(const std::string& s) {
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        size_t n;
        uint32_t cp;
        if (c < 0x80) { ++i; continue; }
        else if ((c >> 5) == 0x6)  { n = 2; cp = c & 0x1F; }
        else if ((c >> 4) == 0xE)  { n = 3; cp = c & 0x0F; }
        else if ((c >> 3) == 0x1E) { n = 4; cp = c & 0x07; }
        else return false;
        if (i + n > s.size()) return false;
        for (size_t k = 1; k < n; ++k) {
            unsigned char cc = static_cast<unsigned char>(s[i + k]);
            if ((cc >> 6) != 0x2) return false;
            cp = (cp << 6) | (cc & 0x3F);
        }
        if (n == 2 && cp < 0x80) return false;
        if (n == 3 && cp < 0x800) return false;
        if (n == 4 && cp < 0x10000) return false;
        if (cp > 0x10FFFF || (cp >= 0xD800 && cp <= 0xDFFF)) return false;
        i += n;
    }
    return true;
}

static void use_utf8_console() {
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
#endif
}

static std::string to_utf8(const std::string& s) {
#ifdef _WIN32
    if (s.empty() || is_valid_utf8(s)) return s;
    int n = MultiByteToWideChar(CP_ACP, 0, s.c_str(), static_cast<int>(s.size()), nullptr, 0);
    if (n <= 0) return s;
    std::wstring w(static_cast<size_t>(n), L'\0');
    MultiByteToWideChar(CP_ACP, 0, s.c_str(), static_cast<int>(s.size()), &w[0], n);
    int m = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), n, nullptr, 0, nullptr, nullptr);
    if (m <= 0) return s;
    std::string out(static_cast<size_t>(m), '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), n, &out[0], m, nullptr, nullptr);
    return out;
#else
    return s;
#endif
}

// Default seed, used when --seed is absent: millisecond wall clock hashed
// through SHA256, digits extracted. Printed at startup so a run stays
// reproducible.
static std::string derive_seed() {
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::system_clock::now().time_since_epoch()).count();
    std::string h = thalia::sha256_hex(std::to_string(ms));
    std::string digits;
    for (char c : h) if (c >= '0' && c <= '9') digits += c;
    return "Thalia:auto:" + digits;
}

int main(int argc, char** argv) {
    use_utf8_console();

    std::string store = "./thalia_data";
    std::string seed;   // empty = derive one from the clock
    std::string lex_path;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--store" && i + 1 < argc) store = argv[++i];
        else if (a == "--seed" && i + 1 < argc) seed = argv[++i];
        else if (a == "--lex" && i + 1 < argc) lex_path = argv[++i];
        else {
            std::cerr << "Usage: thalia [--store DIR] [--seed SEED] [--lex FILE]\n";
            return 1;
        }
    }

    if (seed.empty()) {
        seed = derive_seed();
        std::cout << "auto seed: " << seed << "\n";
    }

    thalia::Engine engine(store, seed);
    if (!lex_path.empty()) engine.load_lexicon(lex_path);

    std::cout << "Thalia ready. store=" << store
              << " mem=" << engine.memory().size()
              << " voice=" << (engine.voice().learned() ? "learned" : "silent")
              << " sem_vocab=" << engine.semantics().vocab() << "\n";
    std::cout << "Type text, or 'img:<path>' for an image.\n"
              << "Commands: :learn <file> :sleep :state :mem :words :sim :facts :why :good :bad :quit\n";

    std::string last_key;
    std::string line;
    while (std::cout << "> " && std::getline(std::cin, line)) {
        line = to_utf8(line);
        if (line.empty()) continue;
        if (line == ":quit") break;
        if (line == ":state") {
            std::cout << "self sim-to-self=1.0 (trivially)\n"
                      << "dopamine="   << engine.self().chem().dopamine   << "\n"
                      << "cortisol="   << engine.self().chem().cortisol   << "\n"
                      << "serotonin="  << engine.self().chem().serotonin  << "\n"
                      << "adrenaline=" << engine.self().chem().adrenaline << "\n";
            continue;
        }
        if (line == ":sleep") {
            std::cout << "Sleeping... (consolidating memory)\n";
            auto rep = engine.sleep();
            std::cout << "Woke. considered=" << rep.considered
                      << " merged="     << rep.merged
                      << " reinforced=" << rep.reinforced
                      << " pruned="     << rep.pruned
                      << " re_encoded=" << rep.re_encoded
                      << " | mem size now " << engine.memory().size() << "\n";
            continue;
        }
        if (line == ":mem") {
            std::cout << "memory size=" << engine.memory().size() << "\n";
            int shown = 0;
            for (auto it = engine.memory().all().rbegin();
                 it != engine.memory().all().rend() && shown < 5; ++it, ++shown) {
                std::cout << "  [" << it->second.kind << "] "
                          << it->second.key.substr(0,12)
                          << " w=" << it->second.weight
                          << " hits=" << it->second.hits
                          << " '" << it->second.raw_preview << "'\n";
            }
            continue;
        }
        if (line.substr(0,6) == ":words") {
            std::string rest = line.size() > 7 ? line.substr(7) : "";
            auto& sem = engine.semantics();
            // split rest into at most 2 whitespace-separated tokens
            std::vector<std::string> args;
            std::string cur;
            for (char c : rest) {
                if (c == ' ' || c == '\t') {
                    if (!cur.empty()) { args.push_back(cur); cur.clear(); }
                } else cur += c;
            }
            if (!cur.empty()) args.push_back(cur);
            if (args.size() >= 2) {
                std::cout << "sim(" << args[0] << "," << args[1]
                          << ") = " << sem.token_sim(args[0], args[1]) << "\n";
            } else {
                std::cout << "semantics vocab=" << sem.vocab()
                          << " pairs=" << sem.pairs() << "\n";
                std::cout << "usage: :words <token-a> <token-b>   (co-occurrence similarity)\n";
            }
            continue;
        }

        if (line.substr(0,4) == ":sim") {
            std::string rest = line.size() > 5 ? line.substr(5) : "";
            auto& sem = engine.semantics();
            std::vector<std::string> args;
            std::string cur;
            for (char c : rest) {
                if (c == ' ' || c == '\t') {
                    if (!cur.empty()) { args.push_back(cur); cur.clear(); }
                } else cur += c;
            }
            if (!cur.empty()) args.push_back(cur);
            if (args.size() >= 2) {
                std::cout << "sim(" << args[0] << "," << args[1]
                          << ")=" << sem.token_sim(args[0], args[1]) << "\n";
            } else {
                std::cout << "usage: :sim <token-a> <token-b>\n";
            }
            continue;
        }

        if (line.substr(0,6) == ":facts") {
            std::string topic = line.size() > 7 ? line.substr(7) : "";
            auto& inf = engine.inference();
            std::cout << "facts=" << inf.facts() << " subjects=" << inf.subjects() << "\n";
            if (!topic.empty()) {
                auto ans = inf.ask(topic);
                if (ans.empty()) std::cout << "nothing known about " << topic << "\n";
                for (const auto& a : ans) {
                    std::cout << "  " << (a.derived ? "[推断] " : "[事实] ") << a.text << "\n";
                }
            }
            continue;
        }
        if (line.substr(0,4) == ":why") {
            std::string topic = line.size() > 5 ? line.substr(5) : "";
            if (topic.empty()) {
                std::cout << "usage: :why <主题>\n";
                continue;
            }
            auto ans = engine.inference().ask(topic);
            int shown = 0;
            for (const auto& a : ans) {
                if (!a.derived) continue;
                std::cout << a.text << "\n";
                for (const auto& step : a.path) std::cout << "  因为 " << step << "\n";
                ++shown;
            }
            if (shown == 0) std::cout << "no deduction available for " << topic << "\n";
            continue;
        }

        if (line.substr(0,6) == ":learn") {
            std::string path = line.size() > 7 ? line.substr(7) : "";
            if (path.empty()) {
                std::cout << "usage: :learn <file>  (one sentence per line)\n";
                continue;
            }
            size_t n = engine.learn_file(path);
            std::cout << "learned " << n << " lines (voice vocab="
                      << engine.voice().vocab()
                      << " sem_vocab=" << engine.semantics().vocab() << ")\n";
            continue;
        }
        if (line.substr(0,5) == ":good") {
            std::string k = line.size() > 6 ? line.substr(6) : last_key;
            if (!k.empty()) engine.feedback_good(k, 0.3);
            std::cout << "boosted " << k.substr(0,12) << "\n";
            continue;
        }
        if (line.substr(0,4) == ":bad") {
            std::string k = line.size() > 5 ? line.substr(5) : last_key;
            if (!k.empty()) engine.feedback_bad(k, 0.3);
            std::cout << "penalized " << k.substr(0,12) << "\n";
            continue;
        }

        std::string kind = "text";
        std::string raw  = line;
        if (line.substr(0,4) == "img:") {
            kind = "image";
            raw  = line.substr(4);
        }

        auto r = engine.perceive(kind, raw);
        last_key = r.cited_keys.empty() ? "" : r.cited_keys.front();
        std::cout << r.text << "\n";
        std::cout << "  " << r.trace
                  << " key=" << last_key.substr(0,12)
                  << " sim=" << r.best_sim << "\n";
    }

    engine.flush();
    std::cout << "Persisted. Bye.\n";
    return 0;
}
