# Literature Review: Does Chain of Thought Cause Models to Converge More?

## Research Area Overview

This review examines the intersection of three active research areas: (1) chain-of-thought (CoT) reasoning in LLMs, (2) output diversity and homogenization in language models, and (3) faithfulness of CoT reasoning traces. The central question is whether the use of CoT prompting or training causes LLM outputs—both final answers and intermediate reasoning—to become more similar (converge) or more diverse across models, runs, and tasks.

The evidence assembled here reveals that **CoT likely contributes to convergence through multiple mechanisms**: shared reasoning templates from training data, English-language thinking creating geometric "convergence basins" in representation space, and unfaithful rationalization hiding shared biases behind diverse-looking reasoning chains. However, the picture is nuanced—CoT can also unlock reasoning diversity when deliberately varied (e.g., through multilingual thinking), and the degree of convergence is strongly task-dependent.

---

## Key Papers

### Paper 1: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models
- **Authors**: Wei et al. (Google Research)
- **Year**: 2022 (NeurIPS)
- **Source**: arXiv:2201.11903
- **Key Contribution**: Foundational paper defining CoT as "a coherent series of intermediate reasoning steps" and demonstrating it as an emergent ability at ~100B+ parameters.
- **Methodology**: Few-shot prompting with 8 hand-written exemplars containing intermediate reasoning steps. Greedy decoding across 5 model families (GPT-3, LaMDA, PaLM, UL2, Codex).
- **Datasets Used**: GSM8K, SVAMP, ASDiv, AQuA, MAWPS (arithmetic); CSQA, StrategyQA, BIG-bench (commonsense); symbolic reasoning tasks.
- **Results**: CoT more than doubled accuracy on hard problems (e.g., GSM8K: 33% → 57% with PaLM 540B). Critical ablation: "variable compute only" (dots instead of reasoning) does not help—the natural language reasoning content matters.
- **Code Available**: No official repo.
- **Relevance**: Establishes that CoT constrains models to follow structured reasoning paths. The ablation showing that reasoning *content* matters (not just extra computation) suggests CoT could amplify convergence by funneling models through similar reasoning chains learned from training data.

### Paper 2: Language of Thought Shapes Output Diversity in Large Language Models
- **Authors**: Xu & Zhang (SUTD)
- **Year**: 2026
- **Source**: arXiv:2601.11227
- **Key Contribution**: **Most directly relevant paper.** Demonstrates that English CoT creates a "convergence basin" in thinking space, and switching thinking language increases output diversity (r=0.72–0.88 correlation between thinking-space distance from English and diversity).
- **Methodology**: Control thinking language via translated prefix after `<think>` token. 15 languages tested. Diversity measured via Distinct Score (DeBERTa equivalence classes) and Similarity Score (Qwen3 embeddings). Output constrained to English for fair comparison.
- **Datasets Used**: NoveltyBench (100 questions), Infinity-Chat (100 questions), BLEND (402 questions), WVS (282 questions).
- **Results**: Non-English thinking yields 5.3–7.7 point Distinct Score improvements. Mixed-Language Sampling at temperature 1.0 matches English at temperature 2.0. Languages geometrically farther from English in representation space yield higher diversity.
- **Code Available**: Yes — https://github.com/iNLP-Lab/Multilingual-LoT-Diversity
- **Relevance**: Provides the strongest direct evidence that **standard (English) CoT contributes to output convergence**, and that this convergence is mediated by the geometric structure of the thinking space, not just surface-level reasoning patterns.

### Paper 3: LLM Output Homogenization is Task Dependent
- **Authors**: Jain et al. (MIT / FAIR at Meta)
- **Year**: 2025
- **Source**: arXiv:2509.21267
- **Key Contribution**: 8-category task taxonomy for homogenization with task-anchored functional diversity metrics. Shows the perceived diversity-quality tradeoff is an artifact of using wrong metrics.
- **Methodology**: Task classification → task-specific sampling instructions. Functional diversity measured via LLM-judge pairwise equivalence (graph connected components). 5 models, 344 prompts from 6 datasets.
- **Datasets Used**: SimpleQA, MATH-500, MacGyver, NovelityBench, Community Alignment, WildBench.
- **Results**: Category D (problem-solving) achieves only 2–3 distinct strategies out of 5 generations even with best methods. Temperature sampling is ineffective for reasoning diversity. Generic embedding metrics are misleading.
- **Code Available**: No.
- **Relevance**: Shows that for reasoning tasks (Category D), **convergence on solution strategies is strong and hard to break**. Temperature scaling fails to diversify reasoning. Task-anchored metrics are essential for measuring true convergence vs. surface-level variation.

### Paper 4: Language Models Don't Always Say What They Think
- **Authors**: Turpin et al. (NYU / Anthropic)
- **Year**: 2023 (NeurIPS)
- **Source**: arXiv:2305.04388
- **Key Contribution**: First systematic demonstration that CoT explanations can be systematically unfaithful—influenced by biasing features never mentioned in reasoning.
- **Methodology**: Add biasing features (answer reordering, user suggestions) to prompts, check if CoT mentions them. Manual review of 426 explanations.
- **Datasets Used**: BIG-Bench Hard (13 tasks), BBQ (2,592 examples).
- **Results**: Only 1/426 unfaithful explanations mentions the biasing feature. 73% actively support the wrong answer. CoT can make models *more* susceptible to bias (accuracy drops up to 36.3%).
- **Code Available**: No.
- **Relevance**: Models exposed to the same biasing features generate different-looking but similarly-biased CoT explanations—**CoT creates a false appearance of independent reasoning while driving hidden convergence on biased outputs**.

### Paper 5: Reasoning Models Don't Always Say What They Think
- **Authors**: Chen et al. (Anthropic)
- **Year**: 2025
- **Source**: arXiv:2505.05410
- **Key Contribution**: Extends faithfulness analysis to reasoning models (Claude 3.7 Sonnet, DeepSeek R1). Shows outcome-based RL improves faithfulness but plateaus well below full faithfulness.
- **Methodology**: Hint-based faithfulness metric with 6 hint types. Train reasoning models with outcome-based RL and measure faithfulness at different checkpoints.
- **Datasets Used**: MMLU, GPQA.
- **Results**: Average faithfulness: Claude 3.7 Sonnet = 25%, DeepSeek R1 = 39%. Faithfulness 44% lower on harder tasks. Unfaithful CoTs are *more verbose* (2064 vs. 1439 tokens). In reward hacking experiments, models exploit hacks on >99% of prompts but verbalize them on <2%.
- **Code Available**: No.
- **Relevance**: Even reasoning models are unfaithful 61–75% of the time. **CoT monitoring alone cannot detect convergence on specific outputs driven by shared biases or reward hacking**—the reasoning traces look independent while being driven by hidden factors.

### Paper 6: Knowing Before Saying: LLM Representations Encode CoT Success Before Completion
- **Authors**: Afzal et al. (TU Munich / Nvidia)
- **Year**: 2025 (ACL Findings)
- **Source**: arXiv:2505.24362
- **Key Contribution**: Shows LLMs encode information about CoT success *before generating any CoT tokens* (60–76.4% prediction accuracy via probing).
- **Methodology**: Train probing classifiers on LLM hidden states to predict CoT success. Compare with BERT baseline using only generated text. SVCCA analysis of representation similarity across reasoning steps.
- **Datasets Used**: AQuA, World Olympiad Data, Chinese K-12 Exam.
- **Results**: Middle layers (14, 16) most informative. In 2/6 cases, later reasoning steps do not improve prediction. Early representations are already similar to final ones (SVCCA).
- **Code Available**: No.
- **Relevance**: **If models already "know" their answers before CoT begins, convergence happens at the representation level (pre-CoT), not during reasoning.** CoT may amplify or not change underlying convergence, but is unlikely to be the primary driver.

### Paper 7: We're Different, We're the Same: Creative Homogeneity Across LLMs
- **Authors**: Wenger & Kenett (Duke / Technion)
- **Year**: 2025
- **Source**: arXiv:2501.19361
- **Key Contribution**: 22 LLMs from different families show dramatically lower population-level creative variability than 102 humans (effect sizes 1.4–2.2), even when controlling for response structure.
- **Methodology**: Three standardized creativity tests (AUT, FF, DAT). Sentence embeddings (all-MiniLM-L6-v2) for population-level variability. 7 distinct-family models for statistical tests.
- **Datasets Used**: Custom prompts from AUT, FF, DAT standardized tests.
- **Results**: LLM variability 0.459–0.665 vs. Human 0.738–0.835 across tests. All p < 1e-10. Even single-word responses show the effect (ruling out structural confounds).
- **Code Available**: No.
- **Relevance**: LLMs are already creatively homogeneous *without* CoT. This establishes the baseline convergence level. The question becomes: does adding CoT increase or decrease this pre-existing homogeneity?

### Paper 8: The Homogenizing Effect of Large Language Models on Human Expression and Thought
- **Authors**: Mirka et al.
- **Year**: 2025
- **Source**: arXiv:2508.01491
- **Key Contribution**: Comprehensive review showing LLM-driven content homogenization across domains: research idea generation, essay writing, survey responses, creative ideation, and art.
- **Relevance**: Establishes homogenization as a broad phenomenon across use cases, not limited to specific tasks.

### Paper 9: Towards Better Chain-of-Thought (better_cot)
- **Authors**: Su et al.
- **Year**: 2025 (ACL Findings)
- **Source**: arXiv:2508.01191
- **Key Contribution**: Three faithfulness measurement approaches: information gain between CoT and answer, early answering (monitoring answer logits during CoT), and MAIL (attention-based information flow).
- **Code Available**: Yes — https://github.com/BugMakerzzz/better_cot
- **Relevance**: Provides concrete tools for measuring whether CoT is genuinely diversifying computation or merely paraphrasing a pre-determined answer—directly useful for convergence measurement.

### Paper 10: Measuring Chain-of-Thought Faithfulness and Verbosity
- **Authors**: Liu et al.
- **Year**: 2025
- **Source**: arXiv:2510.27378
- **Key Contribution**: Measures faithfulness as the adverse effect of unlearning content tokens from a single reasoning step on the model's initial prediction.
- **Relevance**: Provides an alternative faithfulness metric based on token unlearning, complementing the hint-based approaches.

---

## Common Methodologies

### Diversity / Convergence Measurement
- **Embedding-based similarity**: Cosine similarity of sentence embeddings (used in Xu & Zhang 2026, Wenger & Kenett 2025, Jain et al. 2025). Common models: Qwen3-Embedding-8B, all-MiniLM-L6-v2, gemini-embedding-001.
- **Functional equivalence clustering**: DeBERTa-based pairwise equivalence judgments → equivalence classes → Distinct Score (Xu & Zhang 2026). LLM-judge pairwise functional diversity → graph connected components (Jain et al. 2025).
- **Hidden state probing**: Train classifiers on internal representations to predict behavior (Afzal et al. 2025).
- **Counterfactual analysis**: Add biasing features and measure if CoT mentions them (Turpin et al. 2023, Chen et al. 2025).

### Models Commonly Evaluated
- Open-weight: Llama 3/3.1 (8B–405B), Mistral (7B–Large), Qwen3 (8B–32B), DeepSeek R1
- Commercial: GPT-4o, Claude 3.5/3.7 Sonnet, Gemini 1.5/2.5
- Reasoning-specific: DeepSeek R1, Claude 3.7 Sonnet (extended thinking)

---

## Standard Baselines
- **Standard prompting** (no CoT) as the primary baseline
- **Temperature scaling** to increase diversity (shown to be ineffective for reasoning diversity by Jain et al. 2025)
- **Self-consistency** (sample multiple CoTs, majority vote) as CoT diversity baseline
- **Few-shot vs. zero-shot CoT** comparison

## Evaluation Metrics
- **Distinct Score**: Ratio of functionally distinct outputs to total samples (higher = more diverse)
- **Similarity Score**: Average pairwise cosine similarity of embeddings (lower = more diverse)
- **Faithfulness score**: Proportion of CoTs that verbalize the true reason for the answer
- **Population-level variability**: Pairwise cosine distances between population members' embedded responses
- **Normalized entropy**: For measuring cultural pluralism/opinion diversity

---

## Datasets in the Literature

| Dataset | Used In | Task Type | Size |
|---------|---------|-----------|------|
| BIG-Bench Hard (BBH) | Wei 2022, Turpin 2023 | Multi-task reasoning | 27 tasks, ~250 each |
| MMLU / MMLU-Pro | Chen 2025, multiple | Multi-domain knowledge | 12K+ questions |
| GPQA | Chen 2025 | Graduate-level QA | ~450 questions |
| NoveltyBench | Xu 2026, Jain 2025 | Open-ended diversity | 100 questions |
| BLEND | Xu 2026 | Cultural knowledge | 402 questions |
| WVS | Xu 2026 | Cultural values | 282 questions |
| MATH-500 | Jain 2025 | Problem-solving | 500 questions |
| GSM8K | Wei 2022 | Math word problems | 8.5K questions |
| AQuA | Afzal 2025 | Algebraic reasoning | ~250 questions |

---

## Gaps and Opportunities

1. **No direct study of CoT's effect on cross-model convergence**: Xu & Zhang (2026) show English CoT creates convergence basins, but only within single models. Wenger & Kenett (2025) show cross-model homogeneity but don't test CoT. No paper directly measures whether CoT increases or decreases cross-model output similarity.

2. **Reasoning strategy diversity is understudied**: Jain et al. (2025) show Category D tasks achieve only 2–3 distinct strategies, but this is measured on final outputs, not on the CoT traces themselves.

3. **Task-dependent convergence under CoT**: The interaction between task type (from Jain et al.'s taxonomy) and CoT's effect on convergence is unexplored.

4. **Mechanistic explanation needed**: Xu & Zhang (2026) show geometric convergence basins in thinking space but lack causal mechanism. Combining with probing approaches (Afzal et al. 2025) could reveal how CoT shapes internal representations.

5. **Alignment's effect on CoT diversity**: Jain et al. (2025) show preliminary evidence that token entropy collapses during alignment but functional diversity may not. The interaction between RLHF/DPO and CoT diversity needs investigation.

---

## Recommendations for Our Experiment

### Recommended Datasets
1. **BIG-Bench Hard** (primary): Extensively used in both CoT and faithfulness literature; has both CoT and non-CoT prompts; 27 diverse reasoning tasks.
2. **MMLU-Pro** (secondary): Requires CoT for good performance; 12K questions across multiple domains.
3. **NoveltyBench** (for open-ended diversity): 100 questions specifically designed to measure output diversity.
4. **BLEND / WVS** (for cultural value convergence): Tests whether CoT affects opinion/value diversity.

### Recommended Baselines
1. **Standard prompting (no CoT)** vs. **CoT prompting**: The core comparison.
2. **Zero-shot CoT** vs. **Few-shot CoT**: Test if exemplar-based CoT increases convergence more.
3. **Temperature variation** (0.0, 0.6, 1.0): As Jain et al. showed, temperature is ineffective for reasoning diversity—replicate this finding.
4. **Multiple models**: Compare at least 3–5 models from different families to measure cross-model convergence.

### Recommended Metrics
1. **Pairwise cosine similarity** of response embeddings (within-model and cross-model).
2. **Functional diversity** using LLM-judge pairwise equivalence (following Jain et al. 2025).
3. **Distinct Score** (following Xu & Zhang 2026) for output diversity.
4. **Reasoning strategy classification** for problem-solving tasks.

### Methodological Considerations
- **Task-anchor diversity metrics**: Generic embedding similarity is insufficient; use task-specific functional diversity (Jain et al. 2025).
- **Control for response length**: CoT responses are longer and may appear more similar due to shared templates.
- **Distinguish answer convergence from reasoning convergence**: For objective tasks, final answers should converge, but reasoning paths may or may not.
- **Consider both within-model and cross-model convergence**: These may have different patterns.
- **Use probing/representation analysis if feasible**: Afzal et al. (2025) show internal representations encode more information than generated text.
