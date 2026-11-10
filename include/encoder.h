// Thalia - Encoder: turn raw input into HDC vectors.
// Dual-route (v0.3), mirroring the episodic/semantic split in memory:
//
//   text(s)        EXACT route: char 3-gram surface + whole-string identity
//                  anchor. Pure function of the string (no semantics), so
//                  identical inputs stay bit-stable no matter how the
//                  language statistics grow. Drives "I remember" direct
//                  hits. Formula is bit-identical to v0.2.
//
//   text_assoc(s, sem)  ASSOCIATION route: surface grams + co-occurrence
//                  semantic vectors, NO identity anchor. Paraphrases and
//                  synonyms land close (words in similar company). Drives
//                  "this reminds me" retrieval.
//
// Why two vectors: a single fused vector cannot serve both masters. The
// identity anchor must outweigh the semantic field for stable exact
// matches (w_id > 1 + w_sem), yet must be outweighed by it for
// associative recall (w_id < 1 + w_sem) — contradictory inequalities.
// So the episodic fingerprint and the semantic field are stored apart.
//
// Image -> SHA256(content) -> seed (both routes identical).
#pragma once
#include "hdc.h"
#include <string>

namespace thalia {

class Semantics;   // fwd: encoder.h must not depend on semantics.h

class Encoder {
public:
    // EXACT route (surface + identity anchor). No semantics involved.
    static HdcVec text(const std::string& s);

    // ASSOCIATION route (surface + co-occurrence semantics, no anchor).
    // Unknown tokens contribute their pure index vectors — an unheard
    // word is just an arbitrary symbol until it keeps company.
    static HdcVec text_assoc(const std::string& s, const Semantics& sem);

    // Image file -> HDC. Reads bytes, SHA256 hashes, seeds vector.
    // "100% non-repeating" guarantee: identical bytes -> identical vec.
    static HdcVec image_file(const std::string& path);

    // Raw bytes -> HDC (used internally and for arbitrary attachments).
    static HdcVec bytes(const uint8_t* data, size_t n);
};

}  // namespace thalia
