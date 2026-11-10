// Thalia - tokenizer implementation (moved out of speaker.cpp in v0.3
// so the Semantics layer can share it).
#include "tokenize.h"
#include <cstdint>

namespace thalia {

static bool is_ascii_alnum(unsigned char c) {
    return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
           (c >= '0' && c <= '9') || c == '_' || c == '\'';
}

// Decode the length of one UTF-8 sequence from its lead byte.
static size_t utf8_len(unsigned char lead) {
    if (lead < 0x80) return 1;
    if ((lead >> 5) == 0x6) return 2;
    if ((lead >> 4) == 0xE) return 3;
    if ((lead >> 3) == 0x1E) return 4;
    return 1;
}

static uint32_t utf8_codepoint(const std::string& s) {
    unsigned char b0 = static_cast<unsigned char>(s[0]);
    if (b0 < 0x80) return b0;
    size_t n = utf8_len(b0);
    uint32_t cp = b0 & (0xFF >> (n + 1));
    for (size_t i = 1; i < n && i < s.size(); ++i) {
        cp = (cp << 6) | (static_cast<unsigned char>(s[i]) & 0x3F);
    }
    return cp;
}

static bool is_cjk_ideograph(uint32_t cp) {
    // CJK Unified Ideographs + Extension A (good enough for a prototype).
    return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF);
}

std::vector<std::string> tokenize(const std::string& s) {
    std::vector<std::string> out;
    size_t i = 0;
    while (i < s.size()) {
        unsigned char c = static_cast<unsigned char>(s[i]);
        if (c == ' ' || c == '\t' || c == '\r' || c == '\n') { ++i; continue; }
        if (is_ascii_alnum(c)) {
            size_t j = i;
            while (j < s.size() && is_ascii_alnum(static_cast<unsigned char>(s[j]))) ++j;
            out.push_back(s.substr(i, j - i));
            i = j;
        } else {
            size_t n = utf8_len(c);
            std::string tok = s.substr(i, n);
            uint32_t cp = utf8_codepoint(tok);
            if (is_cjk_ideograph(cp)) {
                out.push_back(tok);          // CJK char = word token
            } else {
                out.push_back(tok);          // punctuation = its own token
            }
            i += n;
        }
    }
    return out;
}

bool is_word_token(const std::string& t) {
    if (t.empty()) return false;
    unsigned char c = static_cast<unsigned char>(t[0]);
    if (c < 0x80) return is_ascii_alnum(c);
    return is_cjk_ideograph(utf8_codepoint(t));
}

bool is_sentence_ender(const std::string& t) {
    return t == "。" || t == "！" || t == "？" || t == "!" || t == "?" || t == ".";
}

}  // namespace thalia
