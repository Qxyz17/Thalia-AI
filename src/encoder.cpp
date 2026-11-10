// Thalia - Encoder implementation
#include "encoder.h"
#include "semantics.h"
#include "tokenize.h"
#include <fstream>
#include <sstream>
#include <vector>

namespace thalia {

// Weight of the semantic field in the association route, relative to the
// surface gram bundle (1.0). High enough that shared/synonymous words
// dominate; no identity anchor exists on this route by design.
constexpr double kAssocSemWeight = 1.5;

static std::string lowercased(const std::string& s) {
    std::string t = s;
    for (auto& c : t) c = static_cast<char>(::tolower(static_cast<unsigned char>(c)));
    return t;
}

// Char 3-gram vectors of the (lowercased) string.
static std::vector<HdcVec> surface_grams(const std::string& t) {
    std::vector<HdcVec> grams;
    if (t.size() < 3) {
        std::string padded = "__" + t + "__";
        padded = padded.substr(0, 3);
        grams.push_back(HdcVec::from_seed("n3:" + padded));
    } else {
        for (size_t i = 0; i + 3 <= t.size(); ++i) {
            std::string g = t.substr(i, 3);
            grams.push_back(HdcVec::from_seed("n3:" + g));
        }
    }
    return grams;
}

HdcVec Encoder::text(const std::string& s) {
    // EXACT route — bit-identical to v0.2's formula:
    //   bundle( all gram vectors + identity vector, then extra identity
    //   weight ), so exact matches are recognizable forever regardless
    //   of how the semantic model evolves.
    std::string t = lowercased(s);
    std::vector<HdcVec> grams = surface_grams(t);
    HdcVec id_vec = HdcVec::from_seed("id:" + t);
    grams.push_back(id_vec);
    HdcVec b = HdcVec::bundle(grams);
    return HdcVec::bundle(b, id_vec, 1, 2);
}

HdcVec Encoder::text_assoc(const std::string& s, const Semantics& sem) {
    // ASSOCIATION route — surface + semantic field, no identity anchor.
    std::string t = lowercased(s);
    HdcVec surface = HdcVec::bundle(surface_grams(t));

    std::vector<HdcVec> semv;
    for (const auto& tok : tokenize(t)) {
        if (!is_word_token(tok)) continue;   // punctuation carries no meaning
        semv.push_back(sem.vec_of(tok));
    }
    if (semv.empty()) return surface;        // e.g. "!!!" — surface only

    return HdcVec::bundle_weighted({
        {surface, 1.0},
        {HdcVec::bundle(semv), kAssocSemWeight},
    });
}

HdcVec Encoder::image_file(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return HdcVec::from_seed("img:missing:" + path);
    std::stringstream ss;
    ss << f.rdbuf();
    std::string data = ss.str();
    return bytes(reinterpret_cast<const uint8_t*>(data.data()), data.size());
}

HdcVec Encoder::bytes(const uint8_t* data, size_t n) {
    return HdcVec::from_bytes(data, n);
}

}  // namespace thalia
