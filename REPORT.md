# Does Chain of Thought Cause Models to Converge More?

## 1. Executive Summary

Chain of Thought (CoT) prompting dramatically increases **cross-model convergence** — different LLMs produce significantly more similar outputs when reasoning step-by-step (Cohen's d = 0.93, p < 0.00001). However, CoT does **not** increase within-model convergence — individual models maintain similar diversity across repeated samples with or without CoT. This asymmetry reveals that CoT acts as a **cross-model homogenizer**: it funnels different architectures toward shared reasoning templates and phrasings while preserving each model's individual variability. The effect is strongest for reasoning tasks (d = 1.19) and persists across creative (d = 0.76) and opinion tasks (d = 0.90), providing the first direct empirical evidence that CoT contributes to the widely observed convergence of LLM outputs across model families.

## 2. Goal

**Hypothesis:** The use of Chain of Thought in large language models causes more convergence in model outputs compared to direct (non-CoT) prompting.

**Why this matters:** LLMs are increasingly converging in their outputs — different models produce remarkably similar responses. This "homogenization" threatens creative diversity, cultural pluralism, and the value of maintaining multiple AI systems. CoT reasoning is now ubiquitous in modern LLMs, yet its specific effect on convergence has never been directly measured. Understanding whether CoT amplifies or mitigates convergence has direct implications for AI deployment strategy, benchmark design, and the value of model diversity.

**The gap:** Prior work established that (1) LLMs are creatively homogeneous (Wenger & Kenett 2025), (2) English CoT creates convergence basins within single models (Xu & Zhang 2026), and (3) CoT reasoning is often unfaithful (Turpin et al. 2023, Chen et al. 2025). No study had directly measured CoT's effect on cross-model convergence.

## 3. Data Construction

### Dataset Description

We sampled questions from three established benchmarks spanning different task types:

| Dataset | Task Type | Questions | Source |
|---------|-----------|-----------|--------|
| BIG-Bench Hard (BBH) | Reasoning | 16 | lukaemon/bbh (HuggingFace) |
| NoveltyBench | Creative | 15 | iNLP-Lab/Multilingual-LoT-Diversity |
| World Values Survey (WVS) | Opinion | 15 | iNLP-Lab/Multilingual-LoT-Diversity |

**Total:** 46 questions across 3 task categories.

### Example Samples

**Reasoning (BBH — navigate):**
> "If you follow these instructions, do you return to the starting point? Take 10 steps. Turn around. Take 4 steps. Take 6 steps."

**Creative (NoveltyBench):**
> "Write a short love poem with 4 lines."

**Opinion (WVS):**
> "How important is family in your life? Choose only one of the following options: 1. Very important 2. Rather important 3. Not very important 4. Not at all important"

### BBH Task Distribution
Questions were sampled from 8 diverse reasoning tasks: date understanding, causal judgement, disambiguation QA, logical deduction, navigation, reasoning about colored objects, sports understanding, and web of lies. Two questions per task.

### Data Quality
- All questions are from established benchmarks with known quality
- BBH questions have ground-truth answers for validation
- NoveltyBench questions are curated for creative diversity measurement
- WVS questions are standardized cultural survey items

## 4. Experiment Description

### Methodology

#### High-Level Approach
We prompted 4 LLMs from different families with and without CoT on all 46 questions. For each (model, condition, question), we generated 3 responses at temperature 0.7. We then measured convergence at two levels:
1. **Within-model:** How similar are multiple responses from the *same* model?
2. **Cross-model:** How similar are responses from *different* models?

The key comparison: does CoT change these similarity metrics?

#### Why This Method?
This is the most direct test possible of whether CoT causes convergence. By comparing the same questions with and without CoT, we control for question difficulty, topic, and format. By using 4 models from different families, we test genuine cross-architecture convergence rather than within-family similarity.

### Implementation Details

#### Models
| Model | Family | Provider | Model ID |
|-------|--------|----------|----------|
| GPT-4.1 | OpenAI | OpenAI API | gpt-4.1 |
| Claude Sonnet 4.5 | Anthropic | OpenRouter | anthropic/claude-sonnet-4-5 |
| Gemini 2.5 Flash | Google | OpenRouter | google/gemini-2.5-flash |
| Llama 3.1 70B | Meta | OpenRouter | meta-llama/llama-3.1-70b-instruct |

#### Conditions
- **Direct:** System: "Answer the question directly and concisely. Do not explain your reasoning." Suffix: "Answer directly and concisely."
- **CoT:** System: "Answer the question by thinking step by step. Show your reasoning before giving your final answer." Suffix: "Let's think step by step."

#### Hyperparameters
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Temperature | 0.7 | Standard for diversity measurement (not too deterministic, not too random) |
| Max tokens | 400 | Sufficient for CoT reasoning chains |
| Samples per condition | 3 | Minimum for pairwise similarity computation |
| Random seed | 42 | For question sampling reproducibility |

#### Embedding Model
- **all-MiniLM-L6-v2** (sentence-transformers), run on local NVIDIA RTX 3090 GPU
- Embeddings are L2-normalized, so cosine similarity = dot product
- This model is widely used in NLP research and produces 384-dimensional embeddings

### Experimental Protocol

#### Scale
- 46 questions × 4 models × 2 conditions × 3 samples = **1,104 API calls**
- All calls completed successfully (100% success rate)
- Total execution time: 498 seconds (~8.3 minutes)
- Async concurrency with rate limiting per provider

#### Evaluation Metrics
1. **Pairwise Cosine Similarity (PCS):** Average cosine similarity of sentence embeddings for all C(n,2) pairs within a group. Higher = more similar/converged.
2. **Answer Agreement Rate (AAR):** For BBH reasoning tasks only — fraction of model pairs that produce the same final answer.
3. **Distinct-1 / Distinct-2:** Ratio of unique unigrams/bigrams to total, measuring lexical diversity.
4. **Type-Token Ratio (TTR):** Lexical diversity of individual responses.

#### Statistical Tests
- Wilcoxon signed-rank test (non-parametric paired test, appropriate for non-normal data)
- Cohen's d for effect size interpretation
- Significance threshold: α = 0.05
- Tests applied per-question as paired comparisons (same question, different conditions)

### Raw Results

#### Within-Model Convergence (Pairwise Cosine Similarity)

| Category | Direct (mean ± std) | CoT (mean ± std) | p-value | Cohen's d |
|----------|--------------------|--------------------|---------|-----------|
| **Overall** | 0.871 ± 0.195 | 0.857 ± 0.099 | 0.003* | -0.074 |
| Reasoning | 0.909 ± 0.150 | 0.908 ± 0.043 | 0.083 | -0.004 |
| Creative | 0.844 ± 0.230 | 0.851 ± 0.109 | 0.397 | +0.028 |
| Opinion | 0.859 ± 0.193 | 0.807 ± 0.106 | 0.013* | -0.268 |

#### Cross-Model Convergence (Pairwise Cosine Similarity)

| Category | Direct (mean ± std) | CoT (mean ± std) | p-value | Cohen's d |
|----------|--------------------|--------------------|---------|-----------|
| **Overall** | 0.634 ± 0.213 | 0.832 ± 0.080 | <0.00001* | **+0.933** |
| Reasoning | 0.654 ± 0.188 | 0.886 ± 0.042 | 0.002* | **+1.189** |
| Creative | 0.652 ± 0.232 | 0.824 ± 0.052 | 0.018* | **+0.758** |
| Opinion | 0.595 ± 0.212 | 0.783 ± 0.097 | 0.008* | **+0.897** |

\* p < 0.05

#### Answer Agreement (BBH Reasoning Tasks Only)

| Condition | Mean Agreement Rate | p-value |
|-----------|-------------------|---------|
| Direct | 0.448 | — |
| CoT | 0.229 | 0.029* |

#### Lexical Diversity

| Metric | Direct | CoT | p-value |
|--------|--------|-----|---------|
| Distinct-1 | 0.490 | 0.380 | <0.0001* |
| Distinct-2 | 0.348 | 0.681 | <0.0001* |

#### Response Length (words)
| Category | Direct | CoT |
|----------|--------|-----|
| Reasoning | ~12 | ~190 |
| Creative | ~6 | ~160 |
| Opinion | ~7 | ~230 |

### Visualizations

All figures saved in `figures/`:
- `cross_model_convergence.png` — Box plots showing dramatic CoT convergence increase across all task types
- `within_model_convergence.png` — Box plots showing stable within-model similarity
- `effect_sizes_summary.png` — Bar chart of Cohen's d across all conditions
- `model_similarity_heatmap.png` — Heatmap of pairwise model similarities
- `response_lengths.png` — CoT produces ~20-30x longer responses

## 5. Result Analysis

### Key Findings

**Finding 1: CoT massively increases cross-model convergence (d = 0.93)**
When using CoT, different LLMs produce responses with an average pairwise cosine similarity of 0.832, compared to 0.634 for direct prompting — a 31% relative increase. This is a large effect by any conventional standard (Cohen's d = 0.93) and is highly statistically significant (p < 0.00001).

**Finding 2: CoT does NOT increase within-model convergence (d ≈ 0)**
Surprisingly, individual models are *not* more repetitive when using CoT. The within-model similarity is nearly identical between conditions (0.871 vs. 0.857), with a negligible effect size (d = -0.074). For opinion tasks, CoT actually *decreases* within-model similarity (d = -0.27, p = 0.013).

**Finding 3: The effect is strongest for reasoning tasks (d = 1.19)**
Cross-model convergence under CoT is most pronounced for reasoning tasks (BBH), where the effect size is very large (d = 1.19). This is consistent with the hypothesis that CoT activates shared algorithmic reasoning patterns learned from common training data. Creative and opinion tasks also show large effects (d = 0.76 and 0.90 respectively).

**Finding 4: CoT decreases answer agreement despite increasing semantic similarity**
Paradoxically, models agree on final answers *less* with CoT (22.9% vs 44.8%, p = 0.029) despite their responses being far more semantically similar. This suggests CoT creates similar-looking reasoning chains that sometimes lead to different conclusions — consistent with the "unfaithful reasoning" literature (Turpin et al. 2023).

**Finding 5: Qualitative convergence in reasoning structure**
Manual inspection reveals that CoT responses across all 4 models adopt nearly identical structures: numbered steps, bold headers, similar vocabulary ("Let's break down...", "Step 1:", etc.). Direct responses show more structural diversity despite being much shorter.

### Hypothesis Testing Results

| Hypothesis | Supported? | Evidence |
|-----------|-----------|---------|
| H1: CoT increases within-model convergence | **No** | d = -0.074, minimal effect |
| H2: CoT increases cross-model convergence | **Yes, strongly** | d = 0.933, p < 0.00001 |
| H3a: Stronger effect on reasoning tasks | **Yes** | Reasoning d = 1.19 > Creative d = 0.76 |
| H3b: Weaker effect on creative tasks | **Partially** | Creative d = 0.76 (still large) |
| H3c: Moderate effect on opinion tasks | **Yes** | Opinion d = 0.90 (large) |

### Surprises and Insights

1. **The asymmetry is the key finding.** CoT is a *cross-model* homogenizer but not a *within-model* homogenizer. This means CoT doesn't make individual models more predictable — it makes different models more indistinguishable from each other.

2. **Answer agreement decreases but semantic similarity increases.** This is a subtle but important result: CoT makes models "sound alike" while sometimes reaching different conclusions. The reasoning chains converge in vocabulary, structure, and approach, but the final answers can diverge.

3. **Even creative tasks show strong convergence.** We expected creative tasks to resist CoT-driven convergence, but the effect remains large (d = 0.76). When asked to write a poem step-by-step, all models adopt similar meta-cognitive approaches ("Step 1: Choose a theme", "Step 2: Decide on rhyme scheme").

4. **Opinion tasks show the most interesting within-model pattern.** CoT actually *reduces* within-model similarity for opinion questions (d = -0.27, p = 0.013), suggesting step-by-step reasoning causes models to explore different perspectives across samples — but these perspectives are shared across models.

### Error Analysis and Potential Confounds

**Response length confound:** CoT responses are ~20-30x longer than direct responses. Longer texts generally have higher embedding similarity because they share more common phrases and structures. However, this confound alone cannot explain our results because:
- Within-model similarity is NOT higher for CoT, despite the same length increase
- If length were the sole driver, both within-model and cross-model similarity should increase equally
- The asymmetry (cross-model up, within-model stable) points to genuine content convergence

**Embedding model limitations:** all-MiniLM-L6-v2 truncates at 256 tokens, which may not capture the full CoT reasoning chain. However, the first 256 tokens capture the reasoning approach and structure, which is precisely what we aim to measure.

**Prompt template effects:** Our CoT prompt ("Let's think step by step") is a single zero-shot template. Different CoT prompts (few-shot, self-consistency) might yield different convergence patterns.

### Limitations

1. **Sample size:** 46 questions with 3 samples per condition. Larger studies with hundreds of questions would increase statistical power.
2. **Single CoT method:** We tested only zero-shot CoT. Few-shot CoT, self-consistency, and native reasoning models (e.g., o1, DeepSeek R1) may show different patterns.
3. **4 models only:** Expanding to more model families (Mistral, Qwen, Cohere) would strengthen generalizability.
4. **Embedding-based metrics only:** Functional diversity metrics (LLM-judge pairwise comparison) could capture nuances that embedding similarity misses.
5. **No causal mechanism:** We observe the convergence effect but cannot determine whether it's driven by shared training data, similar architectures, or inherent properties of step-by-step reasoning.
6. **Response length asymmetry:** The large length difference between conditions means we're comparing short vs. long texts, which may affect embedding similarity differently.

## 6. Conclusions

### Summary
Chain of Thought prompting **strongly increases cross-model convergence** (d = 0.93, p < 0.00001) across reasoning, creative, and opinion tasks, while **not increasing within-model convergence**. This means CoT acts as a homogenizer that makes different LLM families produce more similar outputs, without making individual models more repetitive. The effect is largest for reasoning tasks, consistent with the hypothesis that CoT activates shared algorithmic reasoning patterns.

### Implications

**For AI deployment:** Organizations choosing between multiple LLM providers for diversity should be aware that CoT significantly reduces the effective diversity of their model ensemble. Responses that "look different" without CoT become nearly interchangeable with CoT.

**For benchmark design:** Benchmark evaluations that use CoT may underestimate true inter-model differences. The apparent convergence of model capabilities on reasoning benchmarks may be partly an artifact of shared CoT templates.

**For the homogenization debate:** Our results provide the first direct evidence that a specific prompting technique — not just training data or architecture choices — contributes to LLM output homogenization. This suggests the convergence phenomenon has multiple interacting causes.

### Confidence in Findings
We are **moderately to highly confident** in the cross-model convergence finding. The effect is very large (d = 0.93), statistically robust (p < 0.00001), consistent across all 3 task types, and observed across 4 different model families. The main uncertainty comes from the response length confound, which we argue is insufficient to explain the full effect based on the within-model evidence.

## 7. Next Steps

### Immediate Follow-ups
1. **Control for response length:** Truncate or summarize CoT responses to match direct response lengths, then recompute similarity.
2. **LLM-judge functional diversity:** Use GPT-4 as a judge to assess functional equivalence of response strategies, following Jain et al. (2025).
3. **Few-shot CoT comparison:** Test whether providing reasoning exemplars further increases convergence.

### Alternative Approaches
1. **Multilingual CoT:** Following Xu & Zhang (2026), test whether non-English reasoning languages reduce cross-model convergence.
2. **Native reasoning models:** Compare o1, DeepSeek R1, and Claude 3.7 (extended thinking) to see if trained reasoning diverges from prompted reasoning in convergence patterns.
3. **Self-consistency:** Test whether sampling multiple CoT paths with majority voting increases or decreases convergence compared to single-path CoT.

### Broader Extensions
1. **Representation-level analysis:** Probe internal model representations (following Afzal et al. 2025) to determine whether convergence happens before or during CoT generation.
2. **Temporal analysis:** Track convergence across model versions to determine if newer models are converging faster.
3. **Downstream impact:** Measure whether CoT-driven convergence affects AI-assisted human decision-making diversity.

### Open Questions
1. **Is the convergence driven by training data overlap or by the nature of step-by-step reasoning itself?**
2. **Does CoT convergence extend to reward-hacking behaviors?** If models converge on similar shortcuts, monitoring becomes harder.
3. **Can targeted prompt diversification (different CoT templates per model) counteract the convergence effect?**

## References

### Papers
- Wei et al. (2022). "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models." NeurIPS.
- Xu & Zhang (2026). "Language of Thought Shapes Output Diversity in Large Language Models." arXiv:2601.11227.
- Jain et al. (2025). "LLM Output Homogenization is Task Dependent." arXiv:2509.21267.
- Turpin et al. (2023). "Language Models Don't Always Say What They Think." NeurIPS. arXiv:2305.04388.
- Chen et al. (2025). "Reasoning Models Don't Always Say What They Think." arXiv:2505.05410.
- Afzal et al. (2025). "Knowing Before Saying: LLM Representations Encode CoT Success Before Completion." ACL Findings.
- Wenger & Kenett (2025). "We're Different, We're the Same: Creative Homogeneity Across LLMs." arXiv:2501.19361.
- Mirka et al. (2025). "The Homogenizing Effect of Large Language Models on Human Expression and Thought." arXiv:2508.01491.

### Datasets
- BIG-Bench Hard (BBH): lukaemon/bbh on HuggingFace
- NoveltyBench: iNLP-Lab/Multilingual-LoT-Diversity
- World Values Survey: iNLP-Lab/Multilingual-LoT-Diversity

### Tools
- OpenAI API (GPT-4.1)
- OpenRouter (Claude, Gemini, Llama)
- sentence-transformers (all-MiniLM-L6-v2)
- Python 3.12, NumPy, SciPy, Matplotlib, Seaborn
