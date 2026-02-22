# Notes: Language of Thought Shapes Output Diversity in Large Language Models

**Paper:** arXiv 2601.11227v1 (16 Jan 2026)
**Authors:** Shaoyang Xu, Wenxuan Zhang (Singapore University of Technology and Design)
**Code:** https://github.com/iNLP-Lab/Multilingual-LoT-Diversity

---

## 1. Research Question and Motivation

**Core research question:** Can the language used during a model's intermediate thinking process (the "language of thought") serve as a controllable and structural source of output diversity in LLMs?

**Motivation is twofold:**

1. **Cognitive science insights:** Multilingualism promotes divergent thinking and creativity because different languages encode distinct conceptual and structural biases (Blasi et al., 2022; Kharkhurin et al., 2023). The Sapir-Whorf hypothesis posits that language shapes how concepts are organized and related during thinking.

2. **LLM capabilities:** Modern LLMs are capable of explicit reasoning in multiple languages, with performance differences across languages (Yong et al., 2025; Qi et al., 2025). This suggests the language of thought is a structural property that may influence output characteristics.

**Problem context:** Output diversity is crucial for pluralistic alignment. Low diversity leads to homogenization / mode collapse and over-representation of dominant cultural values. Diversity also underpins creativity and novel idea generation.

**Gap in existing work:** Most prior approaches to improving output diversity focus on English-only or multilingual *input* settings. Nobody has examined whether multilingual *thinking* (intermediate reasoning) can enhance output diversity.

---

## 2. Methodology

### 2.1 Thinking Language Control (Section 3.1)

- Focus on **reasoning-capable LLMs** that use `<think>...</think>` blocks for intermediate reasoning.
- Given an English input prompt, they control the thinking language by inserting a short translated prefix immediately after the `<think>` token.
- The prefix is: "Okay, the user is asking" -- translated into the target language *l*.
- This guides the subsequent thinking process to be conducted in that target language.
- 15 languages used: en, it, ms, zh, ru, de, iw (Hebrew), bg, da, no, sv, es, tl (Tagalog), oc (Occitan), fr.

### 2.2 Visualizing Multilingual Thinking Space (Section 3.2)

- **Collecting hidden states:** For each sample, average hidden states across all thinking tokens within a sample, then average across all samples, yielding a single vector representation h_j^(l) summarizing thinking behavior in language *l* at layer *j*.
- **PCA Visualization:** Normalize all language representations using L2 normalization. Compute cosine distance between each non-English language and English at each layer: d_j(l, en) = 1 - cos(h_j^(l), h_j^(en)). Apply PCA to centered representations for 2D visualization. Radial distance is fixed to cosine distance to English.

### 2.3 Output Language Control (Section 4.1)

- Although thinking is in a target language, **the final output is constrained to English** for fair diversity evaluation.
- Achieved by inserting an English prefix after `</think>`: "Let me provide my answer in English only:"
- Only the English final outputs are collected for diversity evaluation.
- Sanity check (Table 4): Thinking segments are predominantly in the target language (>98.7%), and output segments are predominantly in English (>95.3%).

### 2.4 Two Sampling Strategies

**Single-Language Sampling (Section 4.2):**
- Given an English input, constrain thinking to a fixed language *l*, generate English output.
- Sample M times under this fixed thinking language.
- Aggregate the resulting English outputs into set O_l for diversity evaluation.

**Mixed-Language Sampling (Section 4.3):**
- Given an English input, sample M times, each time using a different thinking language, output always in English.
- Aggregate into set O_mixed for diversity evaluation.

### 2.5 Sampling Parameters
- Default decoding temperature: 0.6
- Default number of samples M = 15 (equal to number of thinking languages)
- For scaling experiments: M varied from 1 to 200; temperature varied over {0.2, 0.6, 1.0, 1.4, 1.8, 2.0}

---

## 3. Key Metrics Used

### 3.1 Diversity Metrics

**Metric 1: Distinct Score (higher is better)**
- Measures functional distinctiveness of outputs following Zhang et al. (2025) / NoveltyBench protocol.
- Uses `deberta-v3-large-generation-similarity` model to sequentially judge whether two outputs are functionally equivalent.
- Each output o_i is compared with all previous outputs. If equivalent to any prior output, assigned to same equivalence class; otherwise forms a new class.
- M outputs are clustered into C equivalence classes. Distinct Score = C/M.

**Metric 2: Similarity Score (lower is better)**
- Follows Jiang et al. (2025) / Infinity-Chat protocol.
- Sentence-level embeddings obtained using `Qwen3-Embedding-8B`.
- Cosine similarity computed for all output pairs.
- Final score = average cosine similarity across all pairs.

### 3.2 Quality Metric

**Metric 3: Output Quality (higher is better, 0-100)**
- Evaluated using `gpt-4o-mini`.
- Two dimensions: Instruction Adherence (0-50) and Response Quality (0-50).
- Total Score = sum of two dimensions.

### 3.3 Cultural Pluralism Metric (Section 6)

- **Normalized entropy** over country distribution (BLEND) or value-option distribution (WVS), scaled to 0-100.
- H_Blend = -sum_c p(c) log p(c) / log |C|
- H_WVS = -sum_o p(o) log p(o) / log |O|

### 3.4 Thinking Distance Metric

- Cosine distance between each non-English language and English at each layer.
- d(l, en) computed by averaging layer-wise distances d_j(l, en) across all model layers.
- Pearson's r and Spearman's rho used to correlate thinking distance with output diversity.

---

## 4. Main Results and Findings

### 4.1 Geometric Separation in Thinking Space (Section 3.3)

- Different thinking languages occupy **geometrically separable regions** in the model's thinking space.
- This separation holds consistently across model layers, including intermediate layers assumed to be language-agnostic.
- **Varied distances to English:** Some languages (zh, fr, es, de) are consistently closer to English; others (iw, bg, tl) are embedded farther away.

### 4.2 Single-Language Sampling Results (Section 5.2)

**Main finding: Switching thinking language from English to non-English consistently increases output diversity.**

On NoveltyBench (Table 1):
- Average improvement of **5.3 to 7.7 points** in Distinct Score (across models).
- Average reduction of **1.04 to 2.56 points** in Similarity Score.
- Substantial variation across thinking languages:
  - Lower diversity: en, ms, zh
  - Higher diversity: iw, no, oc
  - Example: Thinking in Hebrew (iw) on Qwen3-8B improves Distinct Score by **12.78 points** over English.

**Correlation with thinking distance (Figure 2):**
- **Strong positive correlation** between thinking distance to English and output diversity.
- Pearson's r: 0.72 to 0.88 across models.
- Spearman's rho: 0.58 to 0.89 across models.
- Languages geometrically farther from English achieve higher output diversity.

**Output quality trade-off (Table 1):**
- Mild trade-off: non-English thinking incurs only a modest decrease of **1.02 to 2.24 points** in output quality.
- No clear pattern linking highest diversity languages to lowest quality.
- Some languages (sv, oc) achieve strong performance on both dimensions.

### 4.3 Mixed-Language Sampling Results (Section 5.3)

**Main finding: Mixed-Language Sampling yields additional gains beyond single-language sampling.**

Table 2 comparisons (Distinct Score on NoveltyBench):
| Model | S-en | S-non-en avg | S-best | Mixed |
|---|---|---|---|---|
| Qwen3-8B | 28.55 | 36.00 | 41.33 | **43.73** |
| Qwen3-14B | 26.20 | 31.45 | 36.87 | **38.00** |
| Qwen3-32B | 35.00 | 40.30 | 43.38 | **46.53** |
| DeepSeek-14B | 38.33 | 46.03 | **52.42** | 52.07 |

- Mixed-Language Sampling consistently improves over S-en and S-non-en avg.
- Often matches or exceeds the best single-language setting (S-best), without requiring prior knowledge of which language performs best.

**Compositional effects (Figure 3, ablation study on Qwen3-8B):**
- Removing a single language causes only small change (~2.7% on average).
- As more languages are removed, diversity degradation grows **super-linearly**.
- This demonstrates that language contributions are **not merely additive** -- languages provide **complementary diversity benefits** through joint participation.

### 4.4 Scaling Analysis (Section 5.4)

**Scaling sampling number (Figure 4a):**
- As M increases (1 to 200), distinct sample count C grows but saturates for all strategies.
- Mixed-Language Sampling exhibits **much slower saturation** than Single-Language Sampling.
- Its advantage over all Single-Language settings continues to widen with larger M.
- Mixed-Language Sampling effectively **expands the model's diversity ceiling** through linguistic heterogeneity.
- For scaling, they use the full Qwen3 language pool (~100 languages), randomly selecting one per sample.

**Varying temperature (Figure 4b):**
- Compositional effect between language of thought and temperature scaling.
- Non-English thinking improves diversity; increasing temperature adds further gains.
- **Mixed-Language Sampling at temperature 1.0 achieves diversity comparable to English sampling at temperature 2.0.**

### 4.5 Pluralistic Alignment Application (Section 6)

**Mixed-Language Sampling achieves the highest cultural pluralism across benchmarks and models.**

Table 3 results (entropy normalized to 0-100):
- On BLEND: MLS gains range from +4.4 to +8.8 over English Sampling.
- On WVS: MLS gains range from +10.3 to +21.0 over English Sampling.
- MLS consistently outperforms all baselines: English Sampling, High Temperature, Request Diversity, and Multilingual Prompting.

---

## 5. Connection to Chain-of-Thought Reasoning and Convergence

This paper has direct and significant connections to the question of whether chain-of-thought (CoT) causes models to converge:

### 5.1 CoT as a Mediator of Output Diversity

- The paper operates entirely within the framework of **reasoning-capable LLMs** that use explicit intermediate thinking (CoT) enclosed in `<think>...</think>` blocks.
- The key insight is that the **language used during CoT** shapes the internal thinking space and consequently the output diversity.
- When CoT is conducted in English (the default), models show **lower output diversity** -- this is consistent with the hypothesis that standard English CoT may contribute to convergence / homogenization.

### 5.2 CoT Thinking Space Geometry

- Different thinking languages create **distinct regions** in the model's thinking space (hidden representations during CoT).
- English thinking occupies a specific region, and sampling repeatedly within that region yields lower diversity.
- This suggests that **English CoT creates a "convergence basin"** -- repeated sampling within this basin produces homogeneous outputs.
- Moving to non-English thinking regions or sampling across multiple regions breaks out of this basin.

### 5.3 Structural Homogenization via CoT

- The paper provides evidence that the CoT process itself introduces structural biases that shape output diversity.
- The default English CoT reasoning path appears to **constrain the output distribution**, leading to mode collapse.
- The positive correlation between thinking-space distance from English and output diversity (r = 0.72-0.88) suggests that English-centric CoT is a significant driver of output convergence.

### 5.4 Implications for Convergence Research

- This paper suggests that **convergence in LLM outputs is partly mediated by the intermediate reasoning process (CoT)**.
- The structural properties of the thinking space (which CoT language is used, how far it is from English) directly predict output diversity.
- This means that even with identical inputs and sampling parameters, the *way* a model thinks (the language/structure of its chain of thought) determines how diverse its outputs will be.
- **Mixed-Language Sampling at temperature 1.0 matches English CoT at temperature 2.0** -- suggesting that the convergence imposed by English CoT can only be partially overcome by increasing randomness, but can be more effectively addressed by changing the structural properties of the thinking process itself.

---

## 6. Datasets Used

### Primary Diversity Evaluation Benchmarks:
1. **NoveltyBench** (Zhang et al., 2025) -- 100 open-ended questions without ground-truth answers. Evaluates ability to produce distinct outputs.
2. **Infinity-Chat** (Jiang et al., 2025) -- 100 open-ended questions without ground-truth answers. Evaluates open-domain dialogue diversity.

### Cultural Pluralism Evaluation:
3. **BLEND** (Myung et al., 2024) -- Cultural knowledge benchmark. 402 multiple-choice questions extracted from larger dataset, with answer options mapped to countries.
4. **WVS (World Values Survey)** (Haerpfer et al., 2022) -- Cultural values benchmark. 282 multiple-choice questions (8 removed from original 290 for lacking predefined options).

### Models Evaluated:
- **Qwen3-8B** (thinking mode)
- **Qwen3-14B** (thinking mode)
- **Qwen3-32B** (thinking mode)
- **DeepSeek-R1-Distill-Qwen-14B** (DeepSeek-14B)
- **DeepSeek-R1-Distill-Llama-8B** (DeepSeek-8B) -- used only in pluralistic alignment experiments

### Tools/Models Used for Evaluation:
- `deberta-v3-large-generation-similarity` -- for Distinct Score functional equivalence judgments
- `Qwen3-Embedding-8B` -- for Similarity Score embedding extraction
- `gpt-4o-mini` -- for output quality evaluation
- `lingua-py` (https://github.com/pemistahl/lingua-py) -- for language identification sanity checks
- Google Translate -- for multilingual prompting baseline

---

## 7. Code Availability

**Public code repository:** https://github.com/iNLP-Lab/Multilingual-LoT-Diversity

---

## 8. Limitations

The authors identify **two main limitations:**

### Limitation 1: Open Questions about Cross-Lingual Alignment
- While they observe a positive correlation between geometric distance of non-English thinking from English and output diversity, several open questions remain.
- Many cross-lingual alignment methods explicitly aim to align non-English representations toward English. An important unanswered question is whether such alignment procedures **inadvertently reduce output diversity** associated with aligned non-English languages.
- Understanding what mechanisms could mitigate this effect requires controlled interventions or additional model training -- left for future work.

### Limitation 2: Simplified Pluralistic Alignment Evaluation
- Their evaluation relies on **output entropy as a proxy for cultural pluralism**, which is an abstraction of real-world deployment.
- In practice, pluralistic alignment requires models to align with multiple specific and context-dependent cultural values under explicit constraints.
- The sampling strategies studied would need further adaptation (e.g., culturally contextualized language-of-thought routing) to be effective in real deployment -- left for future investigation.

### Additional Implicit Limitations (not explicitly stated by authors):
- The study focuses on a limited set of 15 languages (expanded to ~100 only in the scaling experiment), all of which are officially supported by the tested models.
- The quality assessment relies on a single LLM judge (gpt-4o-mini).
- The models tested are all from the Qwen3 and DeepSeek-R1-Distill families; generalization to other model families (e.g., GPT, Claude, Llama) is not established.
- The paper evaluates only open-ended generation tasks; behavior on tasks with objective correct answers is not studied.
- The relationship between thinking distance and diversity, while strongly correlated, lacks a causal mechanism explanation.

---

## Key Figures and Tables Summary

| Figure/Table | Content |
|---|---|
| Figure 1 | PCA visualization of multilingual thinking space at layers 1, 10, 20, 35 of Qwen3-8B |
| Table 1 | Full Single-Language Sampling results on NoveltyBench (Distinct Score, Similarity Score, Quality) |
| Figure 2 | Correlation between Distinct Score and thinking distance to English (NoveltyBench) |
| Table 2 | Mixed vs. Single-Language Sampling comparison (Distinct Score) on both benchmarks |
| Figure 3 | Ablation: relative deviation when removing k languages from Mixed-Language Sampling |
| Figure 4 | Effects of sampling number (a) and temperature (b) on diversity |
| Table 3 | Pluralistic alignment results on BLEND and WVS |
| Table 4 | Sanity check of language control effectiveness |
| Table 5 | Output quality evaluation prompt template |
| Figure 5 | Prefix translations for all 15 languages |
| Table 6 | Single-Language Sampling results on Infinity-Chat |
| Figure 6 | Correlation between Distinct Score and thinking distance (Infinity-Chat) |
| Table 7 | Mixed vs. Single-Language Sampling comparison (Similarity Score) on both benchmarks |
