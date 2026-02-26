# WWHO — Unified Meta-Vocabulary Tokenizer
### Architecture & Codebase Documentation

> **Project Status:** Active Development  
> **Stack:** Python, tiktoken, DFA-based syllabic tokenization, BPE  
> **Target:** Multilingual LLM tokenization for Latin + Indic scripts (Sinhala, Devanagari)

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [High-Level Architecture](#2-high-level-architecture)
3. [Module Breakdown](#3-module-breakdown)
   - [router.py](#31-routerpy--code-switch-segmenter)
   - [linguis_trie.py](#32-linguis_triepy--table-driven-dfa-tokenizer)
   - [gpe_trainer.py](#33-gpe_trainerpy--gpe-bpe-trainer)
   - [encoder.py](#34-encoderpy--sgpe--meta-encoders)
   - [export.py](#35-exportpy--hf-tokenizer-export)
4. [Data Flow](#4-data-flow)
5. [The Meta-Vocabulary ID Space](#5-the-meta-vocabulary-id-space)
6. [Core Architectural Principles](#6-core-architectural-principles)
7. [Glossary](#7-glossary)

---

## 1. Project Overview

WWHO is a **hybrid tokenization system** that unifies two separate token ID spaces:

- **tiktoken** — handles Latin script, ASCII, code, digits, emoji via OpenAI's `o200k_base` BPE vocabulary (~200K tokens)
- **SGPE (Syllabic Grapheme-aware Pair Encoding)** — handles Indic scripts (currently Sinhala and Devanagari) via a linguistically-informed BPE trained on top of DFA-based syllable segmentation

The central claim is that **syllabic pre-tokenization for Indic scripts before BPE merging** produces more linguistically coherent tokens than raw byte/character-level BPE — and that both spaces can be unified into a single integer ID space without collision for use in a single LLM vocabulary embedding table.

---

## 2. High-Level Architecture

```
Raw Text
   │
   ▼
┌─────────────────────────┐
│   CodeSwitchSegmenter   │  (router.py)
│  Splits text into       │
│  Latin / Sinhala /      │
│  Devanagari segments    │
└────────────┬────────────┘
             │
      ┌──────┴──────┐
      │             │
  LATIN          INDIC SEGMENT
      │             │
      ▼             ▼
  tiktoken     LinguisTrie DFA
  (o200k_base) (linguis_trie.py)
  BPE bytes    Syllabic units
      │             │
      │             ▼
      │         GPE/BPE Merges
      │         (SGPE vocab)
      │             │
      └──────┬──────┘
             │
             ▼
    ┌─────────────────┐
    │   MetaVocab     │  (encoder.py)
    │ tiktoken IDs:   │
    │  [0, 200K)      │
    │ SGPE IDs:       │
    │  [200K, 200K+N) │
    └─────────────────┘
             │
             ▼
      List[int]  (unified token IDs)
```

---

## 3. Module Breakdown

### 3.1 `router.py` — Code-Switch Segmenter

**Responsibility:** Detect script boundaries in raw text and split into typed segments.

#### Key Classes

**`Script` (Enum)**
```
LATIN      — ASCII, digits, punctuation, emoji, code
SINHALA    — U+0D80..U+0DFF
DEVANAGARI — U+0900..U+097F
```

**`TextSegment` (dataclass)**
```python
@dataclass
class TextSegment:
    text: str
    script: Script
    has_leading_space: bool  
```
The `has_leading_space` flag passes inter-segment spacing information down to the DFA tokenizer so it can prepend a space marker to the first syllable — a common convention in BPE (à la GPT-2/SentencePiece).

**`CodeSwitchSegmenter.segment(text)`**

The segmenter walks character-by-character, classifying each character using `_get_char_script()`. It accumulates runs of the same script. The tricky part is the **trailing-space absorption** logic: when a Latin segment ends with a space character followed immediately by Indic text, that space is stripped from the Latin segment and the following Indic segment gets `has_leading_space=True`. This mirrors the leading-space convention used in GPT-style BPE.

**`_INDIC_PUNCT_CHARS = "\u0964\u0965"`** — Devanagari dandas (।।). These are classified as Sinhala by the router to allow both scripts to terminate sentences without breaking the Indic segment.

---

### 3.2 `linguis_trie.py` — Table-Driven DFA Tokenizer

**Responsibility:** Given an Indic text segment, tokenize it into linguistically valid syllables using a finite automaton defined in a JSON schema.

Despite the name `LinguisTrie`, this is **not a trie data structure**. It is a **DFA (Deterministic Finite Automaton)** tokenizer. The name is a branded/project-specific term. This should be clarified in the codebase.

#### Key Classes

**`LanguageSchema`** — loaded from a JSON file (e.g., `schemas/sinhala.json`). Contains:
- `char_classes`: maps label strings to sets of Unicode codepoints (e.g., `"CONSONANT" → {0x0D9A, 0x0D9B, ...}`)
- `transitions`: DFA transition table: `state → (char_class → next_state)`
- `start_state`, `accept_states`, `emit_states`

**`CharClassifier`** — maps a codepoint to its character class label via a flat lookup table built from the schema. Unknown codepoints → `"O"` (Other).

**`LinguisTrie.tokenize(text, leading_space)`** — the main tokenization loop:

1. Walk text left-to-right.
2. If `leading_space` mode and current char is whitespace: buffer whitespace, the last space before non-whitespace becomes a `pending_space` prefix.
3. Classify the current character. If the DFA has no transition from `START` on this class: emit it as a single-character token (pass-through for unknown/punctuation).
4. If the start transition leads to an `emit_state`: emit immediately (single-char tokens like isolated vowels or marks).
5. Otherwise: run the DFA greedily, tracking the last accept position. Emit the span up to the last accept position.

**`pending_space`** is prepended to the first character of the next syllable, not emitted as a standalone token — this is the leading-space convention.

#### Factory Functions

```python
build_sinhala_linguis_trie()    # loads schemas/sinhala.json
build_devanagari_linguis_trie() # loads schemas/devanagari.json
```

Both cache the loaded `LinguisTrie` in `_dfa_cache` (module-level dict keyed by schema path).

---

### 3.3 `gpe_trainer.py` — GPE/BPE Trainer

**Responsibility:** Train the SGPE vocabulary from a corpus. Outputs `vocab.json` containing the token-to-ID mapping and the ordered list of BPE merge rules.

This is the most complex module. It implements a full BPE training loop with:
- Multiprocessing pre-tokenization (parallel `_pretokenize_line`)
- Chunked streaming corpus processing with partial counter pickling (to stay within RAM)
- A heap-based priority queue for O(log n) best-pair extraction
- An inverted index (`token_index`) for O(1) candidate word lookup per merge

#### Training Pipeline

**Step 1 — `stream_and_count()`**

Streams the training JSONL file in chunks of 5M sentences. Each chunk is processed by a multiprocessing Pool calling `_pretokenize_line()` per line. Results are word-frequency counters saved as pickled partial counters. After all chunks are processed, partial counters are merged sequentially with progressive pruning.

The **pruning strategy during merge** is adaptive:
- `safe_prune = max(1, min_freq - remaining_chunks)` — a word that hasn't reached `min_freq` yet but could still reach it from remaining chunks is not pruned
- Every 5 chunks, a hard prune at `min_freq // 2` is triggered to keep RAM usage bounded

**Step 2 — `build_word_types()`**

Converts the word-frequency counter (mapping `tuple[str]` → `int`) into integer ID arrays using a `SymbolTable`. Rare syllables (below `prune_freq`) are replaced with a sentinel `-1` (UNK).

**Step 3 — `count_all_pairs()` + heap build**

Counts all adjacent symbol-pair frequencies across all word types (weighted by word frequency). Builds a max-heap (negated for Python's min-heap).

**Step 4 — Merge Loop**

```
while budget remaining:
    best_pair = pop_best(heap, pair_counts)
    if freq < min_freq: stop
    merged_id = symbols.add_merged(a_id, b_id)
    merge_and_update(word_types, pair_counts, token_index, merged_id, heap)
    save checkpoint every N steps
```

`merge_and_update()` is the performance-critical function. It:
1. Finds all word types containing the pair via `token_index[a] ∩ token_index[b]`
2. For each such word type, scans and replaces the pair with `merged_id`
3. Updates counts for all affected neighboring pairs (left neighbor, right neighbor)
4. Pushes dirty pairs back to the heap
5. Updates `token_index` to reflect removed and added token occurrences

**`SymbolTable`** — bidirectional string↔int mapping. `add_merged(a_id, b_id)` concatenates the string representations and calls `get_or_add`.

#### Multiprocessing Worker

`_init_worker(script_mode)` initializes module-global worker state (segmenter, DFA tries). `_pretokenize_line(text)` uses these globals. This is the standard Python multiprocessing pattern for avoiding pickling of complex objects.

#### Output — `vocab.json`

```json
{
  "version": "wwho_sgpe",
  "script_mode": "mixed",
  "vocab_size": <int>,
  "special_tokens": ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
  "num_merges": <int>,
  "prune_freq": <int>,
  "leading_space": true,
  "merges": [["a", "b"], ...],
  "vocab": {"token": id, ...}
}
```

---

### 3.4 `encoder.py` — SGPE & Meta Encoders

Contains two encoder classes:

#### `SGPEEncoder`

Pure SGPE encoder (no tiktoken). Loads `vocab.json`, uses `LinguisTrie` for syllabic tokenization, then applies BPE merges.

**`tokenize(text)`:**
1. `layer1_tokenize(text)` → syllables via DFA
2. `segment_into_words(syllables)` → split syllables into word groups at boundaries
3. For each word group: replace OOV syllables with `[UNK]`, apply `_apply_merges_to_word()`

**`_apply_merges_to_word(tokens)`:** Standard BPE merge application — repeatedly find the highest-priority adjacent pair (lowest merge index), merge it, repeat until no mergeable pair remains.

**`encode(text)`:** `tokenize()` → map each token string to its vocab ID (or `unk_id`).

#### `MetaVocab`

Manages the unified ID space:

```
[0,              tiktoken_size)  →  tiktoken tokens
[tiktoken_size,  tiktoken_size + sgpe_size)  →  SGPE tokens
```

All SGPE token IDs are offset by `tiktoken_size` when exposed in the unified space.

Key methods:
- `encode_sgpe_token(token, unk_id_raw)` → offset ID
- `decode_id(uid)` → token string (for SGPE IDs only; returns `None` for tiktoken range)
- `is_tiktoken_id(uid)` → bool

#### `WWHOMetaEncoder`

The production encoder. Uses `CodeSwitchSegmenter` to segment text, then routes each segment to either tiktoken or the SGPE pipeline, returning IDs in the unified `MetaVocab` space.

**`encode(text)` → `list[int]`:**
- LATIN segments → `tiktoken.encode(seg.text)` → raw tiktoken IDs (no offset)
- INDIC segments → DFA syllabify → `segment_into_words` → `_apply_merges()` → `meta.encode_sgpe_token()`

**`decode(ids)`:**
- Accumulates consecutive tiktoken IDs into a buffer, flushes via `tiktoken.decode()`
- Non-tiktoken IDs → direct string lookup in `_sgpe_reverse`
- This correctly handles the mixed-ID-stream case

**`tokenize(text)` → `list[str]`:** Same routing logic but returns string tokens instead of IDs. For tiktoken, each ID is individually decoded (single-token decode) to get the string representation.

---

### 3.5 `export.py` — HF Tokenizer Export

**Responsibility:** Export the trained SGPE vocabulary in Hugging Face `tokenizers` JSON format, with optional meta-vocab export for the unified ID space.

#### `export_hf_tokenizer()`

Exports raw SGPE vocab as a standard HF BPE tokenizer JSON. Includes:
- `added_tokens` for special tokens
- `post_processor` with `[CLS]`/`[SEP]` template (BERT-style)
- `model.type = "BPE"` with vocab and merge rules

#### `export_meta_tokenizer()`

Exports an extended HF-style JSON where:
- All SGPE token IDs are offset by `tiktoken_size`
- A `meta_config.json` sidecar records the offset and tiktoken model used

The resulting `tokenizer_meta.json` is **not a standard HF tokenizer** — no existing HF tokenizer class can load and use it directly. It serves as a serialization format for the unified vocab mapping, not for direct HF inference use. This is a deliberate design choice; WWHO relies on its own custom routing layer, and the meta-export serves as the serialization format for that wrapper.

---

## 4. Data Flow

### Training Flow

```
corpus.jsonl
     │
     ▼  [multiprocessing Pool]
_pretokenize_line()  ──→  CodeSwitchSegmenter + LinguisTrie DFA
     │
     ▼
word_counter (Counter[tuple[str], int])
     │  [partial pickles + merge]
     ▼
word_types (list[list[int]])  +  word_freqs (list[int])
     │  [pair counting + heap]
     ▼
BPE merge loop
     │
     ▼
vocab.json  +  tokenizer.json  +  tokenizer_meta.json
```

### Inference Flow

```
text (str)
     │
     ▼
CodeSwitchSegmenter.segment()
     │
  ┌──┴──────────────────┐
LATIN                 INDIC
  │                     │
tiktoken.encode()    LinguisTrie.tokenize()
  │                     │
  │               segment_into_words()
  │                     │
  │               _apply_merges()
  │                     │
  │               MetaVocab.encode_sgpe_token()
  └──────────┬──────────┘
             │
         list[int]  (unified IDs)
```

---

## 5. The Meta-Vocabulary ID Space

This is the core architectural claim of WWHO.

```
ID 0          →  tiktoken token 0
...
ID 199,999    →  tiktoken token 199,999
ID 200,000    →  SGPE token with raw_id=0
ID 200,001    →  SGPE token with raw_id=1
...
ID 200,000+N  →  SGPE token with raw_id=N
```

Where `N = len(sgpe_vocab)`.

**Total vocabulary size** = `tiktoken.n_vocab + len(sgpe_vocab)`.

**Why this works for an LLM:** The embedding table of an LLM is indexed by token ID. If the model is trained with this unified vocab, it sees tiktoken tokens at their original IDs and SGPE tokens at offset IDs. The model learns separate embeddings for each. Cross-script attention is handled by the transformer — the tokenizer does not need to enforce any interaction.

**Current limitation:** There is no pre-trained LLM using this vocab. The encoder is a standalone tokenization tool. Integrating it with a training framework (e.g., as a custom HF tokenizer or a SentencePiece replacement) requires a wrapper that HF Trainer can call — this does not currently exist.

---

---|------|-------|----------|
| 1 | `gpe_trainer.py` | `boundary_tokens` always returned as `set()` — dead code | Low |
| 2 | `gpe_trainer.py` | `_is_boundary_token` and `segment_into_words` defined here but imported by `encoder.py` — wrong layering | Medium |
| 3 | `encoder.py` | `SGPEEncoder` hardcodes Sinhala DFA, cannot handle Devanagari | Medium |
| 4 | `encoder.py` | Space and punct injection into SGPE vocab at encoder init — should be guaranteed by trainer | Medium |
| 5 | `encoder.py` | `_apply_merges` duplicated between `SGPEEncoder` and `WWHOMetaEncoder` | Low |
| 6 | `router.py` | `CodeSwitchRouter.tokenize_to_ids()` raises `NotImplementedError` — partially dead class | Low |
| 7 | `linguis_trie.py` | Named `LinguisTrie` but is a DFA, not a trie — misleading name | Low |
| 8 | `linguis_trie.py` | Silent fallback when no DFA accept state reached — swallows tokenization errors | High |
| 9 | `export.py` | `tokenizer_meta.json` is not loadable by any HF tokenizer class — not documented | Medium |
| 10 | All | No unit tests exist in the provided codebase | High |

---

## 6. Core Architectural Principles

### The Fundamental Decoupling Principle

The core insight of the WWHO architecture is: **embed linguistic structure as a prior constraint, not a learned latent variable.**
Instead of asking a statistical algorithm (BPE) to discover syllable boundaries from bytes, WWHO defines them formally via DFA (What), segments the scripts (Where), and strictly bounds the statistics to those linguistic units (How Often). Statistical merging operates *inside* linguistic guarantees, never over them.

### Why syllabic pre-tokenization instead of raw character BPE?

Indic scripts are abugidas — consonants carry an inherent vowel, modified by dependent vowel signs, halant (virama) for consonant clusters, and ZWJ/ZWNJ for ligature control. A raw byte-level BPE (like GPT-4's tiktoken) will split these at arbitrary byte boundaries, producing tokens that cut across syllable boundaries. For example, a conjunct consonant cluster like `ක්‍ෂ` (three Unicode codepoints: consonant + virama + ZWJ + consonant) may be split across two BPE tokens, making it impossible for the model to attend to the full conjunct as a unit.

The DFA pre-tokenizes at syllable boundaries, guaranteeing that BPE merges only happen within or across complete syllables — the minimum phonological unit.

### Why keep tiktoken for Latin instead of training a unified BPE?

Practical reasons: tiktoken's `o200k_base` vocabulary is heavily optimized for English, code, and common multilingual text. Retraining a Latin vocabulary from scratch on a corpus that is primarily Indic would produce a suboptimal Latin vocabulary. The hybrid approach reuses a high-quality existing tokenizer for its domain and adds a specialized tokenizer for the underserved domain.

The tradeoff is the complexity of the unified ID space and the need for the router.

### Should `MetaVocab` know about tiktoken internals?

Currently `MetaVocab` only needs `tiktoken_size` (an integer) — it treats the tiktoken space as a black box offset. This is a clean design. The coupling to tiktoken itself lives in `WWHOMetaEncoder`, not in `MetaVocab`.

### Framework Compatibility (HF `transformers`)

Not directly. HF Trainer expects a tokenizer implementing the `PreTrainedTokenizer` or `PreTrainedTokenizerFast` interface. `WWHOMetaEncoder` is a standalone class. A `PreTrainedTokenizer` wrapper needs to be written that delegates to `WWHOMetaEncoder`. This is non-trivial because HF assumes a single underlying tokenizer library (tokenizers, sentencepiece, or tiktoken), not a hybrid.

---

## 7. Glossary

| Term | Definition |
|------|-----------|
| **SGPE** | Syllabic Grapheme-aware Pair Encoding. BPE trained on top of syllabic pre-tokenization for Indic scripts. |
| **WWHO** | Project name for the unified meta-vocabulary tokenizer. |
| **GPE** | Grapheme Pair Encoding — used interchangeably with SGPE in the codebase. |
| **LinguisTrie** | Project-specific name for the DFA-based syllable tokenizer. Not a trie. |
| **MetaVocab** | The unified ID space: tiktoken IDs followed by offset SGPE IDs. |
| **leading_space** | Convention where the space before a word is prepended to its first token rather than emitted as a standalone token. Used in GPT-2/SentencePiece style tokenizers. |
| **boundary token** | A token containing no Indic script characters — typically punctuation, spaces, digits, or ASCII. Not included in BPE merge training. |
| **DFA** | Deterministic Finite Automaton. The underlying machine for the syllable tokenizer, defined by JSON schema. |
| **has_leading_space** | Flag on `TextSegment` indicating that the space before this Indic segment was absorbed from the preceding Latin segment. |
| **prune_freq** | During training, syllables occurring fewer than this many times in the corpus are replaced with `[UNK]` before BPE training begins. |
| **min_freq** | Minimum frequency for a pair to be merged in BPE. Merges below this threshold stop the training loop. |
| **token_index** | Inverted index: `token_id → set of word_type indices` that contain this token. Used for O(1) candidate lookup in `merge_and_update()`. |
| **emit_state** | A DFA state that forces immediate token emission on entry (single-char tokens). Distinct from accept states which allow further extension. |
