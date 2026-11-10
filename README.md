# Thalia

A small, no-training, non-neural "falseman" agent.

## What it is

Thalia is a deliberate re-implementation of the goals behind Astrax, but with the neural-network machinery stripped out. There is no gradient descent, no backprop, no transformer, no next-token prediction. The agent accumulates experience purely by writing into a content-addressed memory; "thinking" is retrieval + bounded vector arithmetic.

Four layers:

1. **Hyperdimensional Computing (HDC)** — every concept, perception, and the self is a 10000-dim bipolar (+1/-1) bit-packed vector. Binding = XOR, superposition = majority vote (plain or weighted), similarity = normalized agreement. Vectors are seeded from SHA256 of the source string/bytes, so identical inputs always produce identical vectors (the "100% non-repeating" property).

2. **Content-addressed memory, dual-route** — every event (perception / thought / action / self-update) gets a SHA256(raw + timestamp + seq) key and TWO vectors: an **exact route** (`vec`, surface n-grams + a whole-string identity anchor — the episodic "I've seen this exact thing") and an **association route** (`avec`, surface + co-occurrence semantics — the "this reminds me of"). Storage is append-only log + periodic snapshot. No training: write once, retrieve forever.

3. **Distributional semantics** — a `Semantics` layer counts word co-occurrences (window ±5) over everything Thalia hears. Each token owns a deterministic random index vector; its semantic vector is the weighted bundle of its co-occurring words' index vectors. "狗" and "犬" become neighbors because they keep the same company — meaning by counting, not gradients (BEAGLE-style holographic semantics). Statistics live in their own append-only log (`lang.log`), so vocabulary outlives episodic memory pruning.

4. **Self + chemical state** — a slowly-evolving self-vector (`Self::evolve`) that every perception is XOR-bound with, so all memories carry the "I experienced X" tint. Four scalars (dopamine, cortisol, serotonin, adrenaline) modulate retrieval width, threshold, and self-update rate. They are the "body" — reward/stress/focus/urgency APIs nudge them.

## Build

```bash
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j
./thalia --store ./thalia_data --seed "my-identity-seed"
```

Dependency: OpenSSL (libcrypto) for SHA256.

## Use

Interactive REPL. Type text, or `img:<path>` to perceive an image file.

```
> 你是谁
> hello
> img:./photo.jpg
> :state       # show chemical state
> :mem         # show recent memory records
> :words 狗 犬  # co-occurrence similarity between two tokens
> :words        # vocabulary / pair statistics
> :good         # boost the last response (dopamine +)
> :bad          # penalize (cortisol +)
> :sleep        # offline consolidation: merge, re-encode, prune
> :quit
```

## Speech (the Speaker)

Thalia does not contain a language model. It **learns to speak from what it
hears**, the way a baby does — by counting, not by gradient descent:

- `:learn <file>` reads a text file (one sentence per line) into memory as
  "heard" records. `examples/voice_boot.txt` is the bootstrap corpus.
- The Speaker rebuilds word-level n-gram tables over everything it has heard
  (Chinese is tokenized per-character; ASCII per-word). Table weights come
  from memory weights, so `:good`-rewarded memories dominate the phrasing.
- On each text input, it seeds a Markov walk from the retrieved memories +
  your input, and **recombines** an utterance. Everything it says is made of
  fragments it has actually heard — it cannot invent vocabulary.
- Chemicals modulate the voice: high serotonin = greedy, repetitive picks
  (focused); low serotonin = looser association.
- **Self-random** (v0.3): the RNG is seeded from the millisecond wall
  clock hashed through SHA256 (digit-extracted), plus a monotonic
  counter so same-millisecond calls stay distinct. No fixed seed — the
  same context at a different moment yields a different utterance.
  Context steers where the walk starts; the clock rules how it branches.
- **Honest silence**: before it has learned any language, `speak()` returns
  empty and Thalia falls back to plain templates. No faking.
- Conversation itself is language input — every sentence you type during a
  session becomes material for its voice.
- `:sleep` rebuilds the language tables (sleep consolidates language, too).

What comes out is telegraphic, dream-logic speech —
"我有一个小小的非神经智能体。" is a recombination of two corpus sentences.
That's the honest ceiling of n-gram generation, and it's the intended
aesthetic: a creature that can only say what it has heard.

## Semantics (the attack on hollow vectors)

v0.1/v0.2's HDC was semantically hollow: "狗" and "犬" landed at random
distance because their vectors hashed surface bytes. v0.3 attacks this
with corpus co-occurrence statistics — still zero training:

- Every distinct token gets a deterministic random **index vector**
  (`SHA256 -> HDC`). A word never heard in company IS its index: an
  arbitrary symbol, nothing more.
- Every sentence heard (corpus, conversation) updates co-occurrence
  counts within a ±5-token window. Punctuation is excluded — it keeps
  no company.
- A token's **semantic vector** is the weighted bundle of its
  co-occurring words' index vectors, plus a sub-linear self component
  (`kSelfWeight * sqrt(1 + mass)`), so context dominates (meaning is
  the company) while the word never collapses into its sentence-mates.

Measured on a 6-sentence toy corpus: `sim(狗,犬)=0.88` (synonyms, shared
contexts), `sim(球,追)=0.88` (fixed collocates), `sim(草,山)=0.64` (weak
field), `sim(狗,石)=0.56`, `sim(跑,硬)=0.49` (unrelated ≈ random).

### Why two vectors per memory

A single fused vector cannot serve both exact recall and association:
the identity anchor must outweigh the semantic field for stable exact
matches (`w_id > 1 + w_sem`) yet be outweighed by it for associative
recall (`w_id < 1 + w_sem`) — contradictory inequalities, proven on the
majority-vote bit level. So each memory stores both:

- `vec` (exact route): surface grams + identity anchor. Bit-identical
  to v0.2's formula; drives "I remember: ..." direct hits.
- `avec` (association route): surface + semantic field, no anchor.
  Drives "this reminds me of ..." retrieval.

A synonym-variant query retrieves the memory with `assoc≈0.99` while
`exact` stays ~0.50 — strong association, no false episodic certainty.
The dual routes mirror the episodic/semantic division in human memory
(hippocampal exact traces vs neocortical semantic generalization).

## Sleep (offline consolidation)

`:sleep` is the closest thing to "uploading to long-term memory during sleep"
(systems consolidation in neuroscience: hippocampal replay -> neocortical
traces). It's an explicit offline pass over the whole store:

1. **Merge**: near-duplicate memories (HDC sim >= 0.80) are absorbed into a
   survivor. The survivor's vector drifts toward the bundle (a prototype of
   the variants), links are unioned, hit counts accumulate.
2. **Reconsolidation**: every text memory's vectors are re-derived with the
   CURRENT semantic model and the CURRENT self. An episode recorded before
   the agent knew a word gets re-understood once it does — the literal
   systems-consolidation story: hippocampal traces re-written in semantic
   (neocortical) form during sleep.
3. **Reinforce**: frequently-retrieved traces (hits >= 2) gain weight —
   the "things I kept recalling surface faster next time" effect.
4. **Prune**: weak, never-retrieved traces are deleted.
5. Chemicals reset to a rested baseline; dopamine partially carries over.
6. A snapshot is persisted; language statistics are flushed.

The merge threshold (0.80) is deliberately conservative: "你好" and "你好啊"
do NOT merge, because in a 10000-dim space their seeded n-gram vectors are
nearly orthogonal. Only true duplicates (same bytes / same text seen twice,
mod self-drift) merge. Lower it if you want more aggressive generalization.

## Tests

```bash
./scripts/build.sh && ./tests/run_tests.sh
```

20 hermetic tests: first-encounter, direct-hit recall, persistence across
restart, key uniqueness (including same-millisecond writes), image
recognition, dopamine feedback, sleep merging, snapshot persistence,
voice tests (silent before learning, speaks after `:learn`, voice survives
`:sleep`, voice persists across restart), plus semantics tests (synonym
beats unrelated on `:words`, statistics persist across restart, sleep
reconsolidation retrieves a pre-vocabulary memory through the association
route).

## Quick start

```bash
./scripts/build.sh
./build/thalia --store ./data --seed "my seed"
> :learn examples/voice_boot.txt
> 你是谁
> :sleep
> :quit
```

Note: v0.3 changed the on-disk format again (records now carry a second,
association-route vector; language statistics live in `lang.log`). Old
v0.1/v0.2 stores are not readable; delete the store directory.

## Why no training

The original Astrax was a small NN that, despite the architecture being interesting (non-transformer, full-document denoising), couldn't escape the data-volume trap: with limited training data, observed behavior is indistinguishable from a lookup table, and outputs were unstable.

Thalia takes the opposite bet:
- **No training** means no minimum data threshold.
- **One-shot writes** mean the agent starts useful on turn 1.
- **Self-evolution is local** (a nudge per perception), not a global optimization.

The cost is that "skill" doesn't grow with more compute — only with more experience. That's the intended trade.

## What "self" means here

Not consciousness. The self-vector is a slowly-drifting HDC vector that:
- Acts as an XOR mask on every perception (so memories are stored in "I-experienced-X" form, not raw X)
- Is retrievable as a concept (the lexicon entry `self`)
- Evolves from salient events via `nudge(experience, alpha)`

This produces the engineering observable: the agent can answer "who are you" by retrieving its self-vector's neighbors, and its answers shift subtly over time as the self drifts. Whether that constitutes "self-awareness" is a philosophical question; the engineering claim is only that the system has a closed, evolving self-model.

## Files

```
include/hdc.h       # HDC primitives (bind / bundle / bundle_weighted)
include/tokenize.h  # shared word-level tokenizer
include/encoder.h   # text -> HDC, dual-route (exact / association)
include/semantics.h # co-occurrence distributional semantics
include/memory.h    # content-addressed store (dual-route records)
include/self.h      # self-vector + chemical state
include/engine.h    # orchestrator
src/*.cpp           # implementations
src/main.cpp        # CLI
```

## License

MIT.
