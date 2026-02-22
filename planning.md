# Research Plan: Does Chain of Thought Cause Models to Converge More?

## Motivation & Novelty Assessment

### Why This Research Matters
LLMs are increasingly converging in their outputs—different models produce remarkably similar responses to the same prompts. This "homogenization" threatens creative diversity, cultural pluralism, and the value of having multiple AI systems. Chain of Thought (CoT) reasoning is now ubiquitous in modern LLMs, yet its effect on this convergence phenomenon is unknown. Understanding whether CoT amplifies or mitigates convergence has direct implications for how we deploy reasoning-enhanced AI systems.

### Gap in Existing Work
The literature establishes three facts independently but never connects them:
1. LLMs are creatively homogeneous across model families (Wenger & Kenett 2025: effect sizes 1.4–2.2 vs humans)
2. English CoT creates "convergence basins" in thinking space within single models (Xu & Zhang 2026)
3. CoT is unfaithful 61–75% of the time, potentially hiding shared biases behind diverse-looking reasoning (Turpin 2023, Chen 2025)

**No paper directly measures whether CoT increases or decreases cross-model output similarity.** This is the core gap.

### Our Novel Contribution
We conduct the first direct empirical study of CoT's effect on both within-model and cross-model convergence, using 4 LLMs from different families across 3 task types (reasoning, creative, opinion). We measure convergence at both the semantic embedding level and the functional output level.

### Experiment Justification
- **Experiment 1 (BBH reasoning tasks)**: Tests whether CoT funnels different models toward identical reasoning strategies and answers on objective tasks where there is a correct answer.
- **Experiment 2 (NoveltyBench creative tasks)**: Tests whether CoT reduces creative diversity, where there is no "correct" answer and diversity is inherently valuable.
- **Experiment 3 (WVS opinion tasks)**: Tests whether CoT homogenizes value/opinion expressions, where cultural diversity should be preserved.

---

## Research Question
Does Chain of Thought prompting cause LLM outputs to converge more (become more similar) across different models and across repeated samples from the same model, compared to direct (non-CoT) prompting?

## Hypothesis Decomposition

### H1: Within-model convergence
CoT reduces the diversity of multiple responses from the same model on the same question (i.e., within-model pairwise similarity increases with CoT).

### H2: Cross-model convergence
CoT increases the similarity of responses across different model families on the same question (i.e., cross-model pairwise similarity increases with CoT).

### H3: Task dependence
The effect of CoT on convergence is task-dependent:
- H3a: Stronger convergence effect on reasoning tasks (where CoT activates shared algorithmic patterns)
- H3b: Weaker or reversed effect on creative tasks (where CoT might actually enable more elaborate divergent responses)
- H3c: Moderate effect on opinion tasks

---

## Proposed Methodology

### Approach
We prompt 4 LLMs from different families with and without CoT on questions from 3 task categories. For each (model, condition, question), we generate 5 responses at temperature 0.7. We measure convergence using:
1. **Semantic similarity**: Pairwise cosine similarity of sentence embeddings
2. **Answer agreement**: For objective tasks, whether models agree on the final answer
3. **Lexical diversity**: Type-token ratio and distinct n-gram ratios

### Models (4 from different families)
1. **GPT-4.1** (OpenAI) — flagship commercial model
2. **Claude Sonnet 4.5** (Anthropic, via OpenRouter) — strong reasoning model
3. **Gemini 2.5 Flash** (Google, via OpenRouter) — multimodal model
4. **Llama 3.1 70B** (Meta, via OpenRouter) — open-weight model

### Datasets (3 task types)
1. **BIG-Bench Hard** (reasoning): 15 questions sampled from diverse tasks
2. **NoveltyBench** (creative): 15 questions from the curated set
3. **WVS** (opinion/values): 15 questions about cultural values

Total: 45 questions × 4 models × 2 conditions × 5 samples = 1,800 API calls

### Experimental Steps
1. Set up environment with required packages
2. Sample and prepare question sets from each dataset
3. Define prompt templates for direct and CoT conditions
4. Run API calls for all (model, condition, question, sample) combinations
5. Generate sentence embeddings for all responses (local GPU with sentence-transformers)
6. Compute within-model and cross-model similarity metrics
7. Statistical testing (paired t-tests, Wilcoxon signed-rank)
8. Generate visualizations and write report

### Prompt Templates
**Direct (No CoT):**
```
{question}
Answer directly and concisely.
```

**Zero-shot CoT:**
```
{question}
Let's think step by step.
```

### Baselines
- Direct prompting (no CoT) serves as the primary baseline
- Temperature 0.7 is used for all conditions to allow meaningful diversity measurement

### Evaluation Metrics
1. **Pairwise Cosine Similarity (PCS)**: Average cosine similarity of all C(N,2) pairs of response embeddings
2. **Answer Agreement Rate (AAR)**: For BBH only — fraction of responses with the same final answer
3. **Type-Token Ratio (TTR)**: Lexical diversity of responses
4. **Distinct-1/Distinct-2**: Ratio of unique unigrams/bigrams to total unigrams/bigrams

### Statistical Analysis Plan
- Paired t-test or Wilcoxon signed-rank test comparing CoT vs. direct for each convergence metric
- Cohen's d for effect size
- Significance level α = 0.05
- Separate analysis per task type to test H3

## Expected Outcomes
- **If H1 supported**: Within-model PCS is significantly higher with CoT → CoT reduces within-model diversity
- **If H2 supported**: Cross-model PCS is significantly higher with CoT → CoT makes different models more similar
- **If H3 supported**: Effect is larger for reasoning tasks than creative tasks

**Alternative outcomes**: CoT could *decrease* convergence by enabling more elaborate, model-specific reasoning paths. Or the effect could be negligible, suggesting convergence is driven by other factors (training data, RLHF).

## Timeline and Milestones
1. Planning & setup: 15 min ✓
2. Implementation: 30 min
3. API data collection: 30–45 min
4. Analysis & visualization: 20 min
5. Documentation: 20 min

## Potential Challenges
- **API rate limits**: Mitigate with exponential backoff and sequential processing
- **Cost**: ~$20-50 total, well within budget
- **Response length variation**: CoT responses are longer; normalize embeddings properly
- **Model availability**: If a model is unavailable on OpenRouter, substitute with another from the same family

## Success Criteria
1. Successfully collect responses from ≥3 models under both conditions
2. Compute convergence metrics with statistical tests
3. Determine whether CoT significantly changes convergence (either direction)
4. Identify task-dependent patterns
