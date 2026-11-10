// Thalia - shared word-level tokenizer.
// ASCII alnum runs are one token; each CJK ideograph is its own token;
// other UTF-8 sequences are single punctuation tokens; whitespace
// separates but is not a token.
// Shared by the Speaker (generation) and the Semantics layer
// (co-occurrence counting) so both see the same word boundaries.
#pragma once
#include <string>
#include <vector>

namespace thalia {

std::vector<std::string> tokenize(const std::string& s);

// True if the token carries lexical content (ASCII word/number or CJK
// ideograph), as opposed to punctuation. Punctuation is kept for the
// Speaker's sentence logic but excluded from co-occurrence statistics.
bool is_word_token(const std::string& t);

// True if the token is a sentence-ender (。！？!?.)
bool is_sentence_ender(const std::string& t);

}  // namespace thalia
