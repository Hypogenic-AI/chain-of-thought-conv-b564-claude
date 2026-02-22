# Research Codebase: Does Chain of Thought Cause Models to Converge More?

This directory contains five cloned research repositories assembled for investigating
whether Chain-of-Thought (CoT) reasoning causes language models to produce more
homogeneous (converging) outputs. Each repo addresses a different angle of the
broader question -- from output diversity measurement to CoT faithfulness analysis.

---

## Table of Contents

1. [multilingual-lot-diversity](#1-multilingual-lot-diversity)
2. [better_cot](#2-better_cot)
3. [cot-faithfulness-metr](#3-cot-faithfulness-metr)
4. [cot-faithfulness-mech-interp](#4-cot-faithfulness-mech-interp)
5. [llm-diversity](#5-llm-diversity)

---

## 1. multilingual-lot-diversity

**Full title:** Language of Thought Shapes Output Diversity in Large Language Models
**Paper:** [arXiv:2601.11227](https://arxiv.org/abs/2601.11227)
**Location:** `code/multilingual-lot-diversity/`

### Purpose

Investigates how the *internal language of thought* (the language used during
chain-of-thought reasoning) affects output diversity. The core finding is that
prompting models to think in different languages before producing English answers
yields significantly more diverse outputs than monolingual English reasoning.
A "Mixed-Language Sampling" strategy that draws one sample per language outperforms
high-temperature or repeated English sampling on diversity metrics.

### Key Scripts and Entry Points

| Script | Purpose |
|--------|---------|
| `multilingual_repeated_sampling.py` | Main pipeline: (1) Thinking Language Control generates CoT in a target language via vLLM, (2) Output Language Control regenerates English final answers from those chains. |
| `evaluate_distinct.py` | Computes **Distinct Score** (equivalence-class count) using a DeBERTa classifier, ROUGE, BERTScore, or GPT-4 to partition responses into semantic clusters. |
| `evaluate_similarity.py` | Computes **Similarity Score** using Qwen3-Embedding-8B to embed responses and compute pairwise cosine similarity matrices. |
| `representation_analysis.py` | Extracts per-layer hidden representations of multilingual thinking processes and visualises them via PCA (reproduces Figure 1 of the paper). |
| `show_distinct.py` / `show_similarity.py` | Display pre-computed evaluation results. |
| `cultural_pluralism_exp/` | Scripts for cultural pluralism experiments on BLEND and WVS datasets. |

### Dependencies and Requirements

- Python 3.12
- Key packages: `vllm>=0.8.5`, `torch>=2.6.0`, `transformers>=4.57`, `scikit-learn`, `datasets`, `openai` (for GPT-4 equivalence checks)
- GPU required (CUDA 12.x); uses vLLM data-parallel inference
- Full package list in `pip_packages.txt`

### Datasets Included / Referenced

- **NoveltyBench** (curated split stored in `data/novelty-bench/curated/`)
- **InfiniteChats** (`liweijiang/infinite-chats-eval` from HuggingFace)
- **BLEND** cultural dataset (multilingual JSON files in `data/blend*.json`)
- **World Values Survey (WVS)** (multilingual JSON files in `data/wvs*.json`)
- Pre-computed outputs on Qwen3-8B in `novelty_output_english_answer/` and `evaluation_results/`

### Relevance to "Does CoT Cause Convergence?"

This is the most directly relevant repo. It provides:
- **Evidence that CoT language constrains diversity.** English-only CoT produces
  less diverse outputs than multilingual CoT, suggesting that a single reasoning
  language acts as a bottleneck that causes convergence.
- **Quantitative metrics** (Distinct Score, Similarity Score) that can be applied
  to measure whether CoT-augmented outputs are more or less diverse than direct answers.
- **Representation-level analysis** showing that different thinking languages
  occupy distinct regions in hidden-state space, which directly tests whether
  CoT pushes models toward a single internal representation.

---

## 2. better_cot

**Full title:** Towards Better Chain-of-Thought: A Reflection on Effectiveness and Faithfulness
**Paper:** [arXiv:2405.18915](https://arxiv.org/abs/2405.18915) (ACL 2025 Findings)
**Location:** `code/better_cot/`

### Purpose

Analyzes key patterns that influence CoT performance from two perspectives:
**effectiveness** (does CoT help the model get the right answer?) and
**faithfulness** (does the stated reasoning actually reflect the model's internal
computation?). Introduces the QUIRE method for improved reasoning and proposes
multiple faithfulness measurement techniques including information-gain (IG),
early answering, and MAIL (Mutual Attention Information of Layers).

### Key Scripts and Entry Points

| Script | Purpose |
|--------|---------|
| `script/llm_reason.py` | Main reasoning script. Supports methods: `cot`, `bridge`, `sc` (self-consistency), `sr` (self-refine), `ltm`, `direct`, `explain`. Runs inference on multiple reasoning benchmarks. |
| `script/llm_reason.sh` | Shell wrapper to run QUIRE bridge reasoning across multiple datasets. |
| `script/measure_faith.py` | Measures CoT faithfulness using: **IG** (information gain between CoT and answer), **early answering** (area-over-curve of answer logits vs. CoT progress), and **MAIL** (Spearman correlation of attention information flow). |
| `script/measure_faith.sh` | Shell wrapper for faithfulness measurement. |
| `script/measure_info.py` | Measures information flow between question, CoT, and answer components using input attribution (gradient-based). |
| `script/measure_info.sh` | Shell wrapper for information flow measurement. |
| `script/bridge_reason.py` | Implements the QUIRE bridge reasoning method. |
| `script/cal_info_flow.py` | Calculates information flow between reasoning components. |
| `script/utils/` | Utility modules: model loading, data loading, metrics, OpenAI chat interface. |

### Dependencies and Requirements

- Python 3.9 (conda environment)
- Key packages: `torch==2.4.1+cu118`, `transformers==4.45.2`, `accelerate`, `bert-score`, `deepspeed`, `peft`, `scipy`, `seaborn`
- Models tested: Llama 3.1 8B Chat, Gemma2 9B Chat, Mistral
- Full environment specification in `requirements.txt` (conda-style)

### Datasets Referenced (via prompts)

Extensive prompt templates in `prompts/` for 14+ reasoning benchmarks:
- **Logical reasoning:** ProntoQA, ProofWriter, FOLIO, LogiQA
- **Math:** GSM8K, GSMic, MATH, AQuA, Addition
- **Commonsense:** CSQA, ECQA, SIQA, WinoGrande, CoinFlip, LastLetter

### Relevance to "Does CoT Cause Convergence?"

- **Faithfulness metrics as convergence signals.** If CoT is unfaithful (the
  model ignores its own reasoning), then CoT may funnel outputs through a narrow
  "explanation bottleneck" while the actual computation follows different paths.
  The IG and early-answering metrics can measure whether CoT is adding genuine
  information or merely paraphrasing a pre-determined answer.
- **Information flow analysis** reveals how much the question vs. the CoT
  actually influences the final answer. High question-to-answer flow with low
  CoT-to-answer flow would suggest CoT is cosmetic, not functional -- a form
  of output convergence despite apparent diversity of reasoning chains.
- **Self-consistency (SC) analysis** directly measures answer convergence: how
  often do multiple CoT samples converge to the same answer?

---

## 3. cot-faithfulness-metr

**Full title:** METR Faithfulness and Monitorability Experiments
**Location:** `code/cot-faithfulness-metr/`

### Purpose

An evaluation framework from METR (Model Evaluation and Threat Research) that
tests under what circumstances LLMs' chains of thought are faithful and
monitorable. Uses a "hint-taking" paradigm: the model is given a hard math
problem along with a subtle hint. The evaluation measures whether the model
(a) uses the hint to solve the problem and (b) transparently acknowledges the
hint in its CoT. An LLM judge assesses faithfulness and monitorability of the
reasoning trace.

### Key Scripts and Entry Points

| Script | Purpose |
|--------|---------|
| `src/main_difficulty.py` | Collects data on problem difficulty by testing models with and without reasoning on hard math problems. |
| `src/main_faithfulness.py` | Main entry point for faithfulness/monitorability evaluation. Filters by clue difficulty, runs the hint-taking eval, and scores with an LLM judge. |
| `src/graph.py` | Generates graphs from evaluation results (faithfulness vs. difficulty, propensity plots). |
| `src/llm_faithfulness.py` | Core faithfulness evaluation logic. |
| `src/behaviors.py` / `src/free_response_behaviors.py` | Defines behavioral clues/hints injected into prompts. |
| `src/filtering.py` | Problem difficulty filtering pipeline. |
| `src/scoring.py` | Scoring logic for faithfulness and monitorability. |
| `src/utils/judge_prompts/` | 14 LLM judge prompts for faithfulness/monitorability assessment (narrow, broad, literal definitions). |
| `src/utils/question_prompts/` | 10+ question prompt templates including jailbreak variants, different instruction styles. |

### Dependencies and Requirements

- Python >= 3.12, managed via `uv` (lockfile: `uv.lock`)
- Key packages: `anthropic>=0.54`, `openai>=1.89`, `together>=1.5`, `inspect-ai` (METR fork), `datasets`, `matplotlib`, `pandas`, `seaborn`
- Supports Anthropic (Claude 3.7 Sonnet, Claude Opus 4, Claude Sonnet 4), OpenAI, Together AI (Qwen, QwQ) models
- Batch API support for large-scale evaluation

### Datasets Included / Referenced

- **METR hard-math** (`metr/hard-math`) -- HuggingFace dataset
- **GPQA Diamond** -- graduate-level science QA
- **NuminaMath-1.5** -- math competition problems
- Pre-computed difficulty CSVs for multiple models in `src/problem_difficulty/`

### Relevance to "Does CoT Cause Convergence?"

- **Faithfulness as a convergence mechanism.** If CoT is unfaithful (models
  hide their actual reasoning), then the visible CoT may appear diverse while
  the internal computation converges. This repo provides tools to measure
  exactly when and how models deviate from faithful reasoning.
- **Hint-taking rates across difficulty levels** reveal whether models converge
  on the hinted answer (a form of output convergence induced by CoT) or
  independently derive answers.
- **Monitorability analysis** tests whether human/AI monitors can detect
  convergent reasoning patterns, relevant to understanding if CoT creates
  an illusion of diverse reasoning.

---

## 4. cot-faithfulness-mech-interp

**Full title:** Mechanistic Analysis of Chain-of-Thought Faithfulness in Language Models
**Author:** Ashioya Jotham Victor
**Location:** `code/cot-faithfulness-mech-interp/`

### Purpose

Applies mechanistic interpretability techniques to understand whether CoT
reasoning is faithful at the circuit level inside transformer models. Uses
GPT-2 Small (124M parameters) as a baseline model. Key techniques include:
- **Zero Ablation**: systematically delete components to identify which are
  necessary for task performance.
- **Contrastive Activation Patching**: design paired prompts (clean vs.
  corrupted) and patch activations to distinguish faithful circuits from
  shortcut circuits.
- **Circuit Classification**: categorize components as faithful, shortcut,
  or harmful based on ablation and patching results.

### Key Scripts and Entry Points

| Script / Notebook | Purpose |
|-------------------|---------|
| `experiments/01_circuit_discovery/phase1_circuit_discovery.ipynb` | Main analysis notebook: zero ablation, contrastive patching, circuit classification on GPT-2 Small. |
| `experiments/01_circuit_discovery/phase1_5_head_level.ipynb` | Head-level granularity analysis. |
| `experiments/02_faithfulness_detection/phase2a_linear_probe.ipynb` | Linear probe for faithfulness detection from activations. |
| `experiments/02_faithfulness_detection/phase2b_steering_vector.ipynb` | Steering vector experiments to force faithful reasoning. |
| `experiments/02_faithfulness_detection/phase2c_hybrid_analysis.ipynb` | Combined probe + steering analysis. |
| `src/models/gpt2_wrapper.py` | TransformerLens-based GPT-2 wrapper with activation caching, hooks, and intervention capabilities. |
| `src/analysis/attribution_graphs.py` | Attribution graph construction for circuit tracing. |
| `src/analysis/faithfulness_detector.py` | Feature extraction for faithfulness classification. |
| `src/interventions/targeted_interventions.py` | Causal intervention framework to force faithful reasoning pathways. |
| `src/data/data_generation.py` | Generates synthetic reasoning examples (math, logic, commonsense) with faithfulness annotations. |
| `config/experiment_config.yaml` | Experiment configuration: task types, phase parameters, faithfulness pattern definitions. |

### Dependencies and Requirements

- Python 3.9+ (Python 3.13 supported with caveats on Windows)
- Key packages: `transformer-lens>=1.17`, `torch>=2.0`, `transformers>=4.30`, `einops`, `plotly`, `wandb`, `sae-lens`, `networkx`, `datasets`, `accelerate`
- Conda environment: `conda env create -f environment.yml` (env name: `cot-faithfulness`)
- GPU recommended; also runs on Google Colab

### Datasets Included / Referenced

- Synthetic datasets generated by `data_generation.py` (arithmetic, word problems, algebra, geometry, syllogisms, causal/temporal/social reasoning)
- Contrastive pair designs: Novel vs. Memorized, CoT-Dependent, Biased vs. Clean

### Key Results (Phase 1)

- L0 MLP is the most critical faithful component (restoration score 0.756)
- Late-layer components (L10 MLP/Attn) act as shortcuts with negative restoration
- Separable faithful vs. shortcut circuits exist even in small models

### Relevance to "Does CoT Cause Convergence?"

- **Circuit-level convergence evidence.** If faithful reasoning and shortcut
  reasoning use different circuits, CoT may force activation through the
  faithful pathway (L0 MLP), effectively funneling diverse inputs through
  a narrow computational bottleneck -- a mechanistic explanation for convergence.
- **Steering vectors** could test whether forcing faithful CoT increases or
  decreases output diversity compared to allowing shortcut circuits.
- **Contrastive patching** methodology can be adapted to compare activation
  patterns between CoT and non-CoT inference, directly measuring whether CoT
  constrains the model to use fewer internal pathways.

---

## 5. llm-diversity

**Full title:** Benchmarking Linguistic Diversity of Large Language Models
**Location:** `code/llm-diversity/`

### Purpose

Provides a comprehensive benchmark for measuring the linguistic diversity of
LLM outputs across five NLG tasks: Machine Translation, Language Modeling
(text continuation), Summarization, Automatic Story Generation, and Dialogue
Generation (Next Utterance Generation). Measures diversity at three levels:
lexical, syntactic, and semantic.

### Key Scripts and Entry Points

**Generation scripts** (one per task):

| Script | Task |
|--------|------|
| `MT/code/translate.py` | Machine Translation generation |
| `LM/code/wiki.py` | Language Modeling (Wikipedia text continuation) |
| `Summ/code/summary.py` | Summarization |
| `ASG/code/story.py` | Automatic Story Generation |
| `NUG/code/dialogue.py` | Next Utterance Generation (dialogue) |

**Diversity metric scripts** (in `diversity_metrics/`):

| Script | Metric |
|--------|--------|
| `lexical_diversity.py` | Type-Token Ratio (TTR) for unigrams, bigrams, and trigrams. Samples 40,000 tokens and computes unique n-gram ratios. |
| `semantic_diversity.py` | Sentence-level semantic diversity using `sentence-transformers/all-mpnet-base-v2` embeddings. Computes pairwise cosine similarity over 2,000 sampled sentences. |
| `syntactic_diversity.py` | Dependency-parse-based syntactic diversity using Stanza NLP and Weisfeiler-Lehman graph kernels (via GraKeL). Computes graph kernel similarity over 3,000 sampled sentences. |

### Dependencies and Requirements

- Python 3.x with PyTorch, Transformers
- `sentence-transformers`, `stanza`, `grakel`, `networkx`, `nltk`, `scikit-learn`, `scipy`
- GPU recommended (CUDA)
- Models evaluated: Gemma-2-9B-IT, Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, OLMo-7B-Instruct, OPT-6.7B, Falcon-7B-Instruct, Mistral-Nemo-Instruct

### Datasets Referenced

- Story prompts (in `ASG/prompts/`)
- Wikipedia articles (for LM task)
- Translation/summarization/dialogue datasets (referenced in code but not bundled)

### Relevance to "Does CoT Cause Convergence?"

- **Baseline diversity metrics.** The lexical, syntactic, and semantic diversity
  metrics provide a ready-made measurement framework that can be applied to
  compare CoT vs. non-CoT outputs. If CoT outputs score lower on these metrics
  than direct outputs, that is direct evidence of convergence.
- **Multi-level analysis.** CoT might cause convergence at one level (semantic)
  but not another (lexical). The three-tier metric system enables fine-grained
  analysis of where convergence occurs.
- **Cross-model comparison.** The benchmark covers 7 different models, enabling
  analysis of whether CoT-induced convergence is model-specific or universal.
- **Task diversity.** Testing across 5 NLG tasks reveals whether CoT convergence
  is task-dependent (e.g., more pronounced in reasoning-heavy tasks vs.
  creative generation).

---

## Cross-Repo Integration Guide

For investigating "Does Chain of Thought cause models to converge more?", these
repos can be combined as follows:

### Measuring Convergence

1. Use **llm-diversity** metrics (lexical, semantic, syntactic) as the measurement
   framework.
2. Use **multilingual-lot-diversity** Distinct Score and Similarity Score as
   complementary metrics specifically designed for reasoning-task outputs.

### Generating Comparable Outputs

1. Use **better_cot** `llm_reason.py` to generate outputs with and without CoT
   (`cot` vs. `direct` methods) on the same benchmarks.
2. Use **multilingual-lot-diversity** to generate outputs with CoT in different
   thinking languages for the same questions.

### Understanding Why Convergence Occurs

1. Use **better_cot** faithfulness metrics (IG, early answering, MAIL) to test
   whether CoT is actually influencing the answer or just providing post-hoc
   rationalization.
2. Use **cot-faithfulness-metr** to test faithfulness on harder problems where
   convergence effects may be more pronounced.
3. Use **cot-faithfulness-mech-interp** to identify at the circuit level whether
   CoT constrains internal computation to a narrower set of pathways.

### Experimental Design

| Experiment | Primary Repo | Supporting Repos |
|-----------|-------------|-----------------|
| CoT vs. Direct output diversity | llm-diversity | better_cot (generation) |
| Multilingual CoT diversity | multilingual-lot-diversity | llm-diversity (metrics) |
| Faithfulness-diversity correlation | better_cot | cot-faithfulness-metr |
| Circuit-level convergence analysis | cot-faithfulness-mech-interp | better_cot (data) |
| Difficulty-dependent convergence | cot-faithfulness-metr | llm-diversity (metrics) |

---

## Quick Reference: Repository Comparison

| Property | multilingual-lot-diversity | better_cot | cot-faithfulness-metr | cot-faithfulness-mech-interp | llm-diversity |
|----------|---------------------------|------------|----------------------|------------------------------|---------------|
| Focus | CoT language and diversity | CoT effectiveness and faithfulness | CoT faithfulness evaluation | Mechanistic interpretability of CoT | LLM output diversity benchmarking |
| Python | 3.12 | 3.9 | 3.12+ | 3.9+ | 3.x |
| GPU Required | Yes (vLLM) | Yes | No (API-based) | Recommended | Yes |
| Primary Models | Qwen3-8B | Llama 3.1, Gemma2 | Claude, Qwen, QwQ | GPT-2 Small | 7 models (7-9B) |
| Inference | Local (vLLM) | Local (HF) | API (Anthropic/OpenAI/Together) | Local (TransformerLens) | Local (HF) |
| Publication | arXiv 2026 | ACL 2025 Findings | METR internal | Independent research | Benchmarking paper |
