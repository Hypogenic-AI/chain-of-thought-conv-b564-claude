# Resources Catalog

## Summary

This document catalogs all resources gathered for the research project: **"Does Chain of Thought cause models to converge more?"** Resources include 21 academic papers, 5 datasets, and 5 code repositories covering chain-of-thought reasoning, output diversity/homogenization, faithfulness, and internal representations.

---

## Papers

Total papers downloaded: **21**

| # | Title | Authors | Year | File | Key Info |
|---|-------|---------|------|------|----------|
| 1 | Chain-of-Thought Prompting Elicits Reasoning in LLMs | Wei et al. | 2022 | papers/2201.11903_*.pdf | Foundational CoT paper |
| 2 | Language of Thought Shapes Output Diversity | Xu & Zhang | 2026 | papers/2601.11227_*.pdf | **Most relevant** — English CoT creates convergence basins |
| 3 | LLM Output Homogenization is Task Dependent | Jain et al. | 2025 | papers/2509.21267_*.pdf | 8-category task taxonomy for homogenization |
| 4 | We're Different, We're the Same | Wenger & Kenett | 2025 | papers/2501.19361_*.pdf | 22 LLMs show cross-model creative homogeneity |
| 5 | Language Models Don't Always Say What They Think | Turpin et al. | 2023 | papers/2305.04388_*.pdf | CoT unfaithfulness — hides biases |
| 6 | Reasoning Models Don't Always Say What They Think | Chen et al. | 2025 | papers/2505.05410_*.pdf | Reasoning models 61-75% unfaithful |
| 7 | Knowing Before Saying | Afzal et al. | 2025 | papers/2505.24362_*.pdf | Models encode CoT success pre-generation |
| 8 | Homogenization Effects on Creative Ideation | Anderson et al. | 2024 | papers/2402.01536_*.pdf | LLM homogenization in creativity |
| 9 | Homogenizing Effect of LLMs on Expression | Mirka et al. | 2025 | papers/2508.01491_*.pdf | Comprehensive homogenization review |
| 10 | CoT in the Wild Not Always Faithful | Bao et al. | 2025 | papers/2503.08679_*.pdf | Realistic unfaithfulness rates |
| 11 | Dissociation of Faithful and Unfaithful Reasoning | Chen et al. | 2024 | papers/2405.15092_*.pdf | Faithful vs unfaithful mechanisms |
| 12 | Measuring CoT Faithfulness and Verbosity | Liu et al. | 2025 | papers/2510.27378_*.pdf | Unlearning-based faithfulness metric |
| 13 | Is CoT Reasoning Faithful? (better_cot) | Su et al. | 2025 | papers/2508.01191_*.pdf | Three faithfulness measurement approaches |
| 14 | Mechanistic Interpretability of CoT | Chen et al. | 2025 | papers/2507.22928_*.pdf | SAE probing of CoT internals |
| 15 | Towards Faithful CoT: Bridging Reasoners | Xiao et al. | 2024 | papers/2405.18915_*.pdf | Approaches to faithful CoT |
| 16 | Latent CoT Reasoning Survey | Xu et al. | 2025 | papers/2505.16782_*.pdf | Latent reasoning survey |
| 17 | Towards Reasoning Era (Long CoT Survey) | Sui et al. | 2025 | papers/2503.09567_*.pdf | Long CoT survey |
| 18 | Representation Engineering Survey | Chen et al. | 2025 | papers/2502.17601_*.pdf | RepE techniques survey |
| 19 | Active Prompting with CoT | Diao et al. | 2023 | papers/2302.12246_*.pdf | Adaptive CoT exemplar selection |
| 20 | Flow of Reasoning: Divergent Reasoning | Yu et al. | 2024 | papers/2406.05673_*.pdf | Divergent reasoning in LLMs |
| 21 | Measuring Faithfulness in CoT | Lanham et al. | 2023 | papers/2307.13702_*.pdf | Systematic faithfulness measurement |

See papers/README.md for detailed descriptions.

---

## Datasets

Total datasets downloaded: **5** (+ 1 requiring authentication)

| Name | Source | Size | Task | Location | Notes |
|------|--------|------|------|----------|-------|
| BIG-Bench Hard | HuggingFace (lukaemon/bbh) | 3,187 examples (13 tasks) | Multi-task reasoning | datasets/bigbenchhard/ | Core CoT benchmark |
| MMLU-Pro | HuggingFace (TIGER-Lab/MMLU-Pro) | 12,032 test | Multi-domain knowledge | datasets/mmlu_pro/ | Requires CoT for good performance |
| NoveltyBench | Multilingual-LoT-Diversity repo | 100 questions | Diversity evaluation | datasets/noveltybench/ | Open-ended, no ground truth |
| BLEND | Multilingual-LoT-Diversity repo | 402 questions | Cultural knowledge | datasets/blend/ | Country-mapped options |
| WVS | Multilingual-LoT-Diversity repo | 283 questions | Cultural values | datasets/wvs/ | Value/opinion questions |
| GPQA* | HuggingFace (Idavidrein/gpqa) | ~450 questions | Graduate-level QA | *Not downloaded* | Requires HF auth token |

See datasets/README.md for detailed descriptions and download instructions.

---

## Code Repositories

Total repositories cloned: **5**

| Name | URL | Purpose | Location | Notes |
|------|-----|---------|----------|-------|
| Multilingual-LoT-Diversity | github.com/iNLP-Lab/Multilingual-LoT-Diversity | Language of Thought output diversity | code/multilingual-lot-diversity/ | **Most relevant** — CoT language effects on diversity |
| better_cot | github.com/BugMakerzzz/better_cot | CoT faithfulness probing (3 methods) | code/better_cot/ | ACL 2025 Findings; 14+ benchmarks |
| CoT-faithfulness-and-monitorability | github.com/METR/CoT-faithfulness-and-monitorability | METR CoT faithfulness evaluation | code/cot-faithfulness-metr/ | Hint-taking paradigm |
| cot-faithfulness-mech-interp | github.com/ashioyajotham/cot-faithfulness-mech-interp | Mechanistic interpretability of CoT | code/cot-faithfulness-mech-interp/ | TransformerLens on GPT-2 Small |
| llm-diversity | github.com/YanzhuGuo/llm-diversity | LLM linguistic diversity benchmarking | code/llm-diversity/ | 3 metric levels (lexical, semantic, syntactic) |

See code/README.md for detailed descriptions.

---

## Resource Gathering Notes

### Search Strategy
1. **Paper-finder service** attempted first but timed out; fell back to manual web search
2. **Web searches** across arXiv, Semantic Scholar, Google Scholar using multiple query formulations:
   - "chain of thought reasoning convergence language models"
   - "chain of thought prompting effects model behavior"
   - "LLM persona convergence representation similarity"
   - "chain of thought faithfulness reasoning traces"
   - "LLM model collapse homogenization monoculture"
   - "language of thought output diversity"
   - "measuring similarity LLM outputs"
   - "chain of thought internal representations probing"
3. **GitHub search** for implementations of papers found
4. **HuggingFace** for datasets referenced in papers

### Selection Criteria
- **Papers**: Prioritized (1) direct relevance to CoT convergence, (2) recency (2023-2026), (3) methodological rigor and reproducibility, (4) citation count for foundational works
- **Datasets**: Selected benchmarks used across multiple papers in the literature, plus diversity-specific evaluation sets
- **Code**: Prioritized repos with (1) reusable measurement frameworks, (2) ready-made datasets, (3) diversity/faithfulness evaluation tools

### Challenges Encountered
- Paper-finder service timeout — required manual web search
- GPQA dataset is gated on HuggingFace — requires authentication token
- Some BBH dataset loaders deprecated on HuggingFace — used alternative sources
- No single paper directly measures CoT's effect on cross-model convergence — this is the research gap

### Gaps and Workarounds
- **No direct CoT convergence study exists**: This is the core research gap. All evidence is indirect (from diversity, faithfulness, and representation studies).
- **GPQA not downloaded**: Requires HuggingFace authentication. Can be obtained by setting HF_TOKEN.
- **No pre-computed CoT traces dataset**: The experiment will need to generate CoT traces from scratch.

---

## Recommendations for Experiment Design

Based on gathered resources, the following experimental approach is recommended:

### 1. Primary Dataset(s)
- **BIG-Bench Hard**: Core benchmark for comparing CoT vs. standard prompting. 13 diverse reasoning tasks with established baselines.
- **NoveltyBench**: For measuring output diversity on open-ended questions.
- **MMLU-Pro**: For measuring CoT-dependent performance and convergence on knowledge questions.

### 2. Baseline Methods
- **Standard prompting** (no CoT) — primary baseline
- **Zero-shot CoT** ("Let's think step by step")
- **Few-shot CoT** (with exemplars)
- **Multiple models** (at least 3-5 from different families) to measure cross-model convergence

### 3. Evaluation Metrics
- **Pairwise cosine similarity** of output embeddings (within-model and cross-model)
- **Distinct Score** (functional equivalence classes via DeBERTa) following Xu & Zhang (2026)
- **LLM-judge functional diversity** following Jain et al. (2025)
- **Strategy classification** for problem-solving tasks

### 4. Code to Adapt/Reuse
- **multilingual-lot-diversity**: Diversity metrics (Distinct Score, Similarity Score), NoveltyBench evaluation pipeline, vLLM-based generation
- **better_cot**: Faithfulness measurement (IG, early answering, MAIL) for understanding CoT fidelity
- **llm-diversity**: Multi-level linguistic diversity metrics (lexical, semantic, syntactic)
- **cot-faithfulness-mech-interp**: Mechanistic interpretability techniques if probing internal representations

### 5. Experimental Design Recommendations
1. **Core experiment**: Generate responses from 3-5 LLMs on BBH and NoveltyBench with and without CoT. Measure pairwise similarity within-model (convergence across runs) and cross-model (convergence across architectures).
2. **Temperature control**: Test at T=0, 0.6, 1.0 to separate CoT's convergence effect from sampling randomness.
3. **Task-dependent analysis**: Use Jain et al.'s taxonomy to analyze convergence separately for objective (Category A/D) vs. creative (Category G/H) tasks.
4. **Reasoning strategy analysis**: For problem-solving tasks, go beyond final-answer similarity to measure strategy-level diversity using LLM-judge functional diversity.
