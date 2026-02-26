# WWHO — Unified Meta-Vocabulary Tokenizer

> **Project Status:** Active Development
> **Documentation:** [docs/](docs/)

WWHO is a **hybrid tokenization system** designed to solve the challenges of processing multilingual text that includes both high-resource Latin scripts and complex Indic scripts (like Sinhala and Devanagari) within a single Large Language Model (LLM).

## Key Features

*   **Hybrid Architecture:** Combines OpenAI's `tiktoken` (o200k_base) for Latin scripts with a custom **SGPE (Syllabic Grapheme-aware Pair Encoding)** for Indic scripts.
*   **Unified ID Space:** Maps tokens to a single contiguous integer space without collision, allowing a single model to learn mixed-script representations.
*   **Linguistic Integrity:** Uses DFA-based pre-tokenization to guarantee that Indic tokens always respect syllable boundaries, preventing "shattered" tokens.
*   **Code-Switching Ready:** A smart router dynamically splits text into script blocks and dispatches them to the correct tokenizer.

## Quick Start

The easiest way to explore WWHO is via the interactive orchestrator:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the CLI
python orchestrator.py
```

This will launch a menu to train, evaluate, test, and export your tokenizer.

## Documentation

Comprehensive developer documentation is available in the `docs/` directory:

*   [**Introduction**](docs/introduction.mdx) - Overview of the problem and solution.
*   [**Architecture**](docs/architecture.mdx) - High-level design and data flow.
*   [**Project Structure**](docs/project-structure.mdx) - File-by-file breakdown of the codebase.
*   [**Usage Guide**](docs/usage.mdx) - Detailed instructions for training, testing, and exporting.
*   [**Schemas**](docs/schemas.mdx) - Documentation for the JSON-based DFA rules.

## Project Structure

```
.
├── docs/               # Developer documentation
├── schemas/            # DFA definitions for Indic scripts (Sinhala, Devanagari)
├── tests/              # Test suite (battle_v2.py, etc.)
├── gpe_trainer.py      # Main BPE training script
├── encoder.py          # Inference logic (SGPEEncoder, WWHOMetaEncoder)
├── router.py           # Code-switching segmentation logic
├── linguis_trie.py     # DFA-based syllable tokenizer
├── export.py           # Export to Hugging Face format
└── orchestrator.py     # CLI tool for managing the pipeline
```

## Contributing

We welcome contributions! Please see `docs/project-structure.mdx` to understand the codebase layout before making changes.
