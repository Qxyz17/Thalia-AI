// Thalia - Memory implementation
#include "memory.h"
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#define thalia_mkdir(p) _mkdir(p)
#else
#define thalia_mkdir(p) ::mkdir((p), 0755)
#endif

namespace thalia {

static int64_t now_ms() {
    using namespace std::chrono;
    return duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count();
}

// Monotonic sequence for key uniqueness: two events in the same millisecond
// must still get different keys (the "100% non-repeating" guarantee).
// Seeded from time so different processes don't collide either.
static uint64_t next_seq() {
    static std::atomic<uint64_t> seq{static_cast<uint64_t>(now_ms())};
    return seq.fetch_add(1);
}

static void ensure_dir(const std::string& path) {
    // Recursively create directories (mkdir -p).
    std::string cur;
    for (size_t i = 0; i < path.size(); ++i) {
        cur += path[i];
        if (path[i] == '/' || i + 1 == path.size()) {
            thalia_mkdir(cur.c_str());
        }
    }
}

// Kind <-> byte encoding (used by log and snapshot).
static uint8_t kind_encode(const std::string& kind) {
    if (kind == "percept")     return 0;
    if (kind == "thought")     return 1;
    if (kind == "action")      return 2;
    if (kind == "self-update") return 3;
    if (kind == "heard")       return 4;
    if (kind == "image")       return 5;
    return 9;
}
static std::string kind_decode(uint8_t k) {
    switch (k) {
        case 0: return "percept";
        case 1: return "thought";
        case 2: return "action";
        case 3: return "self-update";
        case 4: return "heard";
        case 5: return "image";
        default: return "other";
    }
}

Memory::Memory(const std::string& store_dir) : dir_(store_dir) {
    ensure_dir(dir_);
    log_path_ = dir_ + "/memories.log";
    load();
}

std::string Memory::record(const std::string& kind,
                            const std::string& raw,
                            const HdcVec& vec,
                            const HdcVec& avec,
                            const std::vector<std::string>& links) {
    MemoryRecord r;
    r.raw_preview = raw.substr(0, 64);
    r.raw_full = raw;
    r.kind = kind;
    r.vec = vec;
    r.avec = avec;
    r.weight = 1.0;
    r.ts_ms = now_ms();
    r.hits = 0;
    r.links = links;

    // Key: SHA256(raw + timestamp + monotonic seq) -> guaranteed unique
    std::string key_input = raw + "|" + std::to_string(r.ts_ms) + "|" + std::to_string(next_seq());
    r.key = sha256_hex(key_input);

    records_[r.key] = r;

    // Append to log
    std::ofstream f(log_path_, std::ios::app | std::ios::binary);
    if (f) {
        // Format: 32-byte key (binary) | 1 byte kind | uint32 raw_len | raw |
        //         uint32 vec_len | vec | uint32 avec_len | avec |
        //         double weight | int64 ts | int32 hits |
        //         uint32 n_links | each link (string, 32 bytes)
        f.write(reinterpret_cast<const char*>(r.key.data()), 32);
        uint8_t k = kind_encode(kind);
        f.write(reinterpret_cast<const char*>(&k), 1);
        uint32_t rl = static_cast<uint32_t>(raw.size());
        f.write(reinterpret_cast<const char*>(&rl), 4);
        f.write(raw.data(), rl);
        std::string v = vec.serialize();
        uint32_t vl = static_cast<uint32_t>(v.size());
        f.write(reinterpret_cast<const char*>(&vl), 4);
        f.write(v.data(), vl);
        std::string av = avec.serialize();
        uint32_t avl = static_cast<uint32_t>(av.size());
        f.write(reinterpret_cast<const char*>(&avl), 4);
        f.write(av.data(), avl);
        f.write(reinterpret_cast<const char*>(&r.weight), sizeof(double));
        f.write(reinterpret_cast<const char*>(&r.ts_ms), sizeof(int64_t));
        f.write(reinterpret_cast<const char*>(&r.hits), sizeof(int));
        uint32_t nl = static_cast<uint32_t>(links.size());
        f.write(reinterpret_cast<const char*>(&nl), 4);
        for (const auto& lk : links) {
            // Each link padded/truncated to 32 bytes (sha256 hex length).
            char buf[32] = {0};
            size_t c = std::min<size_t>(32, lk.size());
            std::memcpy(buf, lk.data(), c);
            f.write(buf, 32);
        }
    }
    return r.key;
}

std::vector<Memory::Hit> Memory::retrieve(const HdcVec& q, const HdcVec& qa,
                                          size_t k, const ChemState& chem) {
    std::vector<Hit> all;
    all.reserve(records_.size());
    for (const auto& [key, rec] : records_) {
        double se = HdcVec::sim(q, rec.vec);    // exact / episodic route
        double sa = HdcVec::sim(qa, rec.avec);  // association / semantic route
        // Dopamine boosts well-rewarded memories; cortisol lowers threshold.
        double boost = 1.0 + 0.3 * (rec.weight - 1.0) + 0.1 * (chem.dopamine - 0.5);
        double s = std::max(se, sa) * boost;
        all.push_back({key, s, se, sa, &rec});
    }
    // Threshold: cortisol high -> allow lower sims through (more divergent).
    double threshold = 0.55 - 0.15 * (chem.cortisol - 0.3);
    std::vector<Hit> filtered;
    for (auto& h : all) {
        if (h.sim >= threshold) filtered.push_back(h);
    }
    std::sort(filtered.begin(), filtered.end(),
              [](const Hit& a, const Hit& b) { return a.sim > b.sim; });
    if (filtered.size() > k) filtered.resize(k);
    // Increment hit counts
    for (auto& h : filtered) {
        records_[h.key].hits++;
    }
    return filtered;
}

void Memory::decay(double factor) {
    for (auto& [key, rec] : records_) {
        rec.weight *= factor;
    }
    // Prune very-low-weight, never-hit records older than a threshold.
    for (auto it = records_.begin(); it != records_.end(); ) {
        if (it->second.weight < 0.1 && it->second.hits == 0) {
            it = records_.erase(it);
        } else {
            ++it;
        }
    }
}

Memory::SleepReport Memory::consolidate(double merge_threshold) {
    SleepReport rep;
    rep.considered = records_.size();

    // --- Phase 1: find merge candidates (greedy single pass) ---
    // Similar memories get absorbed into a survivor; the survivor's vector
    // drifts toward the bundle (prototype), links union, hits accumulate.
    std::vector<std::string> keys;
    keys.reserve(records_.size());
    for (auto& [k, r] : records_) keys.push_back(k);

    std::set<std::string> absorbed;
    std::vector<std::pair<std::string, std::string>> merges;  // victim -> survivor
    for (size_t i = 0; i < keys.size(); ++i) {
        if (absorbed.count(keys[i])) continue;
        const HdcVec& va = records_[keys[i]].vec;
        for (size_t j = i + 1; j < keys.size(); ++j) {
            if (absorbed.count(keys[j])) continue;
            double s = HdcVec::sim(va, records_[keys[j]].vec);
            if (s >= merge_threshold) {
                absorbed.insert(keys[j]);
                merges.push_back({keys[j], keys[i]});
            }
        }
    }

    // --- Phase 2: apply merges ---
    for (auto& [victim, survivor] : merges) {
        auto& sv = records_[survivor];
        auto& vc = records_[victim];
        sv.vec = HdcVec::bundle(sv.vec, vc.vec);   // drift toward prototype
        for (const auto& l : vc.links) {
            if (std::find(sv.links.begin(), sv.links.end(), l) == sv.links.end()) {
                sv.links.push_back(l);
            }
        }
        sv.hits += vc.hits;
        sv.weight = std::max(sv.weight, vc.weight) + 0.1;
        // Keep the longer raw text (more language material for the Speaker).
        if (vc.raw_full.size() > sv.raw_full.size()) {
            sv.raw_full = vc.raw_full;
            sv.raw_preview = vc.raw_preview;
        }
        rep.merged++;
    }
    for (const auto& v : absorbed) records_.erase(v);

    // --- Phase 3: replay reinforcement ---
    // Frequently-retrieved traces get stronger (the "sudden vivid recall"
    // effect: things that got replayed a lot surface easier next time).
    for (auto& [k, r] : records_) {
        if (r.hits >= 2) {
            r.weight += 0.1 * r.hits;
            rep.reinforced++;
        }
    }

    // --- Phase 4: prune weak, unused traces ---
    for (auto it = records_.begin(); it != records_.end(); ) {
        if (it->second.weight < 0.15 && it->second.hits == 0) {
            it = records_.erase(it);
            rep.pruned++;
        } else {
            ++it;
        }
    }

    return rep;
}

void Memory::boost(const std::string& key, double delta) {
    auto it = records_.find(key);
    if (it != records_.end()) {
        it->second.weight = std::min(5.0, it->second.weight + delta);
    }
}

void Memory::reencode(const std::function<void(MemoryRecord&)>& fn) {
    for (auto& [key, rec] : records_) fn(rec);
}

void Memory::flush() {
    // The log is append-only; we don't rewrite it. Decay is in-memory.
    // For long-term persistence, a periodic compact would snapshot the
    // current records_ map. We do that here too.
    std::string snap = dir_ + "/memories.snapshot";
    std::ofstream f(snap, std::ios::binary | std::ios::trunc);
    if (!f) return;
    // Simple format: count + records
    uint32_t n = static_cast<uint32_t>(records_.size());
    f.write(reinterpret_cast<const char*>(&n), 4);
    for (const auto& [key, rec] : records_) {
        f.write(key.data(), 32);
        uint8_t k = kind_encode(rec.kind);
        f.write(reinterpret_cast<const char*>(&k), 1);
        uint32_t rl = static_cast<uint32_t>(rec.raw_full.size());
        f.write(reinterpret_cast<const char*>(&rl), 4);
        f.write(rec.raw_full.data(), rl);
        std::string v = rec.vec.serialize();
        uint32_t vl = static_cast<uint32_t>(v.size());
        f.write(reinterpret_cast<const char*>(&vl), 4);
        f.write(v.data(), vl);
        std::string av = rec.avec.serialize();
        uint32_t avl = static_cast<uint32_t>(av.size());
        f.write(reinterpret_cast<const char*>(&avl), 4);
        f.write(av.data(), avl);
        f.write(reinterpret_cast<const char*>(&rec.weight), sizeof(double));
        f.write(reinterpret_cast<const char*>(&rec.ts_ms), sizeof(int64_t));
        f.write(reinterpret_cast<const char*>(&rec.hits), sizeof(int));
        uint32_t nl = static_cast<uint32_t>(rec.links.size());
        f.write(reinterpret_cast<const char*>(&nl), 4);
        for (const auto& lk : rec.links) {
            char buf[32] = {0};
            size_t c = std::min<size_t>(32, lk.size());
            std::memcpy(buf, lk.data(), c);
            f.write(buf, 32);
        }
    }
}

void Memory::load() {
    // Prefer snapshot if it exists; otherwise replay log.
    std::string snap = dir_ + "/memories.snapshot";
    std::ifstream sf(snap, std::ios::binary);
    if (sf) {
        uint32_t n = 0;
        sf.read(reinterpret_cast<char*>(&n), 4);
        for (uint32_t i = 0; i < n; ++i) {
            MemoryRecord r;
            char keybuf[32];
            sf.read(keybuf, 32);
            r.key.assign(keybuf, 32);
            uint8_t k = 0;
            sf.read(reinterpret_cast<char*>(&k), 1);
            r.kind = kind_decode(k);
            uint32_t rl = 0;
            sf.read(reinterpret_cast<char*>(&rl), 4);
            r.raw_full.resize(rl);
            sf.read(&r.raw_full[0], rl);
            r.raw_preview = r.raw_full.substr(0, 64);
            uint32_t vl = 0;
            sf.read(reinterpret_cast<char*>(&vl), 4);
            std::string vbuf(vl, '\0');
            sf.read(&vbuf[0], vl);
            r.vec = HdcVec::deserialize(vbuf);
            uint32_t avl = 0;
            sf.read(reinterpret_cast<char*>(&avl), 4);
            std::string avbuf(avl, '\0');
            sf.read(&avbuf[0], avl);
            r.avec = HdcVec::deserialize(avbuf);
            sf.read(reinterpret_cast<char*>(&r.weight), sizeof(double));
            sf.read(reinterpret_cast<char*>(&r.ts_ms), sizeof(int64_t));
            sf.read(reinterpret_cast<char*>(&r.hits), sizeof(int));
            uint32_t nl = 0;
            sf.read(reinterpret_cast<char*>(&nl), 4);
            r.links.resize(nl);
            for (uint32_t j = 0; j < nl; ++j) {
                char buf[32];
                sf.read(buf, 32);
                r.links[j].assign(buf, 32);
            }
            records_[r.key] = r;
        }
        return;
    }
    // No snapshot; replay log
    std::ifstream lf(log_path_, std::ios::binary);
    if (!lf) return;
    while (lf.peek() != EOF) {
        MemoryRecord r;
        char keybuf[32];
        lf.read(keybuf, 32);
        if (lf.gcount() < 32) break;
        r.key.assign(keybuf, 32);
        uint8_t k = 0;
        lf.read(reinterpret_cast<char*>(&k), 1);
        if (lf.gcount() < 1) break;
        r.kind = kind_decode(k);
        uint32_t rl = 0;
        lf.read(reinterpret_cast<char*>(&rl), 4);
        if (lf.gcount() < 4) break;
        r.raw_full.resize(rl);
        lf.read(&r.raw_full[0], rl);
        r.raw_preview = r.raw_full.substr(0, 64);
        uint32_t vl = 0;
        lf.read(reinterpret_cast<char*>(&vl), 4);
        if (lf.gcount() < 4) break;
        std::string vbuf(vl, '\0');
        lf.read(&vbuf[0], vl);
        r.vec = HdcVec::deserialize(vbuf);
        uint32_t avl = 0;
        lf.read(reinterpret_cast<char*>(&avl), 4);
        if (lf.gcount() < 4) break;
        std::string avbuf(avl, '\0');
        lf.read(&avbuf[0], avl);
        r.avec = HdcVec::deserialize(avbuf);
        lf.read(reinterpret_cast<char*>(&r.weight), sizeof(double));
        lf.read(reinterpret_cast<char*>(&r.ts_ms), sizeof(int64_t));
        lf.read(reinterpret_cast<char*>(&r.hits), sizeof(int));
        uint32_t nl = 0;
        lf.read(reinterpret_cast<char*>(&nl), 4);
        if (lf.gcount() < 4) break;
        r.links.resize(nl);
        for (uint32_t j = 0; j < nl; ++j) {
            char buf[32];
            lf.read(buf, 32);
            r.links[j].assign(buf, 32);
        }
        records_[r.key] = r;
    }
}

}  // namespace thalia
