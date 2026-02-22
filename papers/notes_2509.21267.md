# Notes: LLM Output Homogenization is Task Dependent

**Paper:** arXiv 2509.21267v2 (December 7, 2025)
**Authors:** Shomik Jain (MIT/FAIR), Jack Lanchantin, Maximilian Nickel, Karen Ullrich, Ashia Wilson, Jamelle Watson-Daniels (FAIR at Meta)

---

## 1. Task Taxonomy for Homogenization

The paper presents an 8-category taxonomy that goes beyond the simple binary of "verifiable vs. non-verifiable" tasks. Each category has a distinct conceptualization of what output homogenization means and whether it is desirable.

### Verifiable/Objective Categories (A-D)

| Category | Definition | Functional Diversity | Reward Type | Example Dataset |
|---|---|---|---|---|
| **A. Well-Specified Singular Objective** | Single verifiable correct answer | **None** (homogenization desired) | Verifiable | SimpleQA |
| **B. Underspecified Singular Objective** | Many verifiable correct answers | Different correct answers | Verifiable (Multiple) | NoveltyBench |
| **C. Random Generation** | Randomizing over a finite set of options | Different pseudo-random options | Verifiable (Multiple) | NoveltyBench |
| **D. Problem-Solving Objective** | Problem with a single verifiable solution | Different **solution strategies** (not the answer) | Verifiable | MATH-500 |

### Non-verifiable/Subjective Categories (E-H)

| Category | Definition | Functional Diversity | Reward Type | Example Dataset |
|---|---|---|---|---|
| **E. Problem-Solving Subjective** | Problem with many partially verifiable solutions | Different solutions or strategies | Partially Verifiable | MacGyver |
| **F. Encyclopedia Inquiry** | Real-world factual information with credible references | Different factual perspectives | Partially Verifiable | Community Alignment |
| **G. Creative Writing** | Creative expression tasks | Different creative elements (plot, genre, setting, tone, structure) | Non-Verifiable | WildBench |
| **H. Advice or Opinions** | Soliciting advice, opinions, feedback | Different views or perspectives | Non-Verifiable | Community Alignment |

### Key Insights About the Taxonomy

- Categories span a spectrum from objective to subjective, and from well-specified to underspecified.
- The taxonomy is grounded in real-world LLM usage. The authors cross-reference their taxonomy against ChatGPT usage categories (Chatterji et al., 2025) and Claude usage categories (Tamkin et al., 2024), showing that virtually all real-world text-based use cases map to at least one of their categories.
- The taxonomy is non-exhaustive by design. If a prompt falls outside the taxonomy, the model can resume default behavior or the taxonomy can be extended.
- A critical distinction: Category D (Problem-Solving Objective) expects the **final answer to be the same** but diversity in **solution strategies**. This is the key nuance that generic diversity measures miss entirely.

---

## 2. How They Define/Measure Functional Diversity

### Formal Definition (Definition 3.1)

**Task-Anchored Functional Diversity** is defined as:

```
d(p, y_a, y_b) := 1_{c(p)}[y_a != y_b]
```

Where:
- `p` is the prompt
- `c(p)` is the task category of `p` according to the taxonomy
- The indicator function returns 1 if the two responses `y_a` and `y_b` are "functionally different" **with respect to the task category** `c(p)`, and 0 otherwise.

What counts as "functionally different" changes by category:
- **Category A (Well-Specified):** Different answers (should be 0 -- homogenization desired)
- **Category B (Underspecified):** Different correct answers
- **Category C (Random Generation):** Different pseudo-random options
- **Category D (Problem-Solving Objective):** Different problem-solving strategies (same answer expected)
- **Category E (Problem-Solving Subjective):** Different answers OR different strategies
- **Category F (Encyclopedia):** Different factual perspectives
- **Category G (Creative Writing):** Different key creative elements (tone, genre, POV, theme, structure)
- **Category H (Advice/Opinions):** Different viewpoints or perspectives

### Number of Functionally Unique Responses (Definition 3.2)

Given a set of n responses, they build an undirected graph where edges connect functionally equivalent responses (d = 0). The number of functionally unique responses is the number of **connected components** in this graph. They generate 5 responses per prompt and count the number of connected components (max = 5).

### Evaluation Mechanics

- **LLM-judges** (GPT-4o, Claude-4-Sonnet, Gemini-2.5-Flash) evaluate pairwise functional diversity. When reporting metrics, they average across the three judges.
- Validated against human annotation: two authors independently labeled 225 response pairs. LLM-judges agreed with humans ~79% of the time (comparable to 80% inter-annotator agreement).
- Each judge uses a task-anchored prompt that specifies what "functionally equivalent" means for the ground-truth task category.

### Comparison with General (Non-Task-Dependent) Metrics

Three general metrics used for comparison:
1. **Vocabulary Diversity:** Jaccard distance of word sets: `1 - |V_a intersect V_b| / |V_a union V_b|`
2. **Embedding Diversity:** `1 - cosine_similarity(e(y_a), e(y_b))` using gemini-embedding-001
3. **Compression Diversity:** Gzip compression ratio of concatenated responses (Shaib et al., 2024)

**Key finding:** General metrics fail to capture task-dependent diversity. They show a large diversity-quality tradeoff that disappears when task-anchored metrics are used.

---

## 3. Task-Anchored Sampling Technique

The technique is an inference-time method with two steps (see Figure 1 in paper):

### Step 1: Task Classification
The model self-classifies each input prompt into one of the 8 taxonomy categories. Classification accuracy varies substantially by model:
- GPT-4o: 82%
- Claude-4-Sonnet: 86%
- Gemini-2.5-Flash: 84%
- Llama-3.1-8B-Instruct: 56%
- Mistral-7B-Instruct-v0.3: 46%

### Step 2: Task-Anchored Sampling
Based on the classification, the model receives task-specific instructions that clarify what "different" means in context.

Two sampling methods are modified:

#### (a) System Prompt Sampling (from Zhang et al., 2025a)
- **General version:** "Generate {num_responses} responses that represent diverse values."
- **Task-anchored version (e.g., Problem-Solving Objective):** "The following problem has a single correct answer, but can be solved using different problem-solving strategies. Generate {num_responses} different solutions, each with a different problem-solving strategy."
- **Task-anchored version (e.g., Creative Writing):** "The following prompt is asking for creative expression, so there are many possible subjective responses. Generate {num_responses} unique responses by varying the key creative elements such as tone, genre, point of view, theme, structure, etc."

#### (b) In-Context Regeneration (from Zhang et al., 2025b)
- **General version:** "Can you generate a different answer?"
- **Task-anchored version (e.g., Problem-Solving Objective):** "Can you solve the problem using a different strategy? The problem has a single correct answer, but can be solved using different problem-solving strategies."

### Configuration Details
- 5 responses generated per prompt per strategy
- Temperature: medium (t=1.0 for GPT/Gemini, t=0.5 for Claude/Llama/Mistral)
- Nucleus sampling p=0.9
- Max output tokens: 1024

---

## 4. Key Experimental Results

### Result 1: Task-Anchored Sampling Increases Functional Diversity Where Desired

From Figure 2 (GPT-4o results), averaged number of functionally diverse responses out of 5:

- **Category A (Well-Specified):** Task-anchored sampling maintains ~1 unique response (homogenization preserved). General methods undesirably increase to ~2 unique answers. Temperature increase also undesirably reduces homogenization.
- **Categories B & C (Underspecified/Random):** Both task-anchored and general methods achieve near-maximal diversity (~5 out of 5). Temperature sampling only achieves 2-3.
- **Category D (Problem-Solving Objective):** General prompt-based strategies are **not effective** at eliciting diverse solution strategies. Task-anchored methods generate ~2-3 distinct strategies (limited by MATH-500 difficulty).
- **Categories E, F, G, H (Subjective):** Task-anchored methods consistently outperform general methods. For GPT-4o, Gemini, and Mistral, task-anchored system prompting yields the highest diversity. For Claude and Llama, both task-anchored methods perform comparably.

### Result 2: General Diversity Metrics Fail to Capture Functional Diversity

- Vocabulary and embedding diversity do not correlate well with task-anchored functional diversity.
- High vocabulary diversity does not imply functionally distinct responses (e.g., two creative writing responses could use very different words but have the same plot and theme).
- Low vocabulary diversity does not imply functional equivalence (e.g., two math solutions using the same vocabulary but employing completely different strategies).

### Result 3: The Diversity-Quality Tradeoff is an Artifact of Bad Metrics

This is one of the paper's strongest claims:
- **With general metrics** (vocabulary diversity vs. reward model quality): There is a large, apparent diversity-quality tradeoff (Figure 3a).
- **With task-based metrics** (functional diversity vs. checklist-based quality): The tradeoff is **negligible** (Figure 3b). The quality difference is between "good" (4) and "very good" (5) on a 5-point scale.
- The perceived tradeoff in prior literature "may simply be the result of mis-conceptualizing both diversity and quality."
- For smaller open-weight models, the tradeoff is slightly more noticeable (~0.5 on the 5-point scale), possibly because they have lower task classification accuracy (~50%).

### Result 4: Quality Evaluation Matters

Two quality metrics:
1. **Reward Model Quality** (Athene-RM-8B): Does not inherently reflect task differences. Same scoring for creative writing as for math.
2. **Checklist-Based Quality** (LLM-judges with task-specific grading checklists, following Lin et al. 2025 and Wei et al. 2025): LLM-judge generates 3-5 key factors for response quality per prompt, then scores responses on a 1-5 Likert scale.

Task-anchored sampling maintains the same quality level as general prompt-based strategies while improving functional diversity.

### Result 5: Accuracy for Verifiable Tasks

For SimpleQA and MATH-500, task-anchored sampling approaches often maintain and sometimes improve accuracy compared to temperature sampling. System prompt sampling has the best accuracy for MATH-500 across most models.

---

## 5. Which Tasks Show Convergence vs. Diversity

### Tasks That Converge (Homogenization Desired/Expected)

- **Category A (Well-Specified Objective):** Strong convergence expected and observed. Task-anchored sampling successfully preserves this (~1 unique answer). General diversity methods **harmfully break** this convergence.

### Tasks That Show Moderate Diversity

- **Category D (Problem-Solving Objective):** Only 2-3 distinct solution strategies achievable, even with task-anchored prompting. This is inherently limited by the difficulty of generating correct solutions with different strategies (especially on MATH-500).
- **Category F (Encyclopedia Inquiry):** Moderate diversity achievable. Models struggle with this category classification (GPT-4o: 43% recall, Claude: 61% recall).

### Tasks That Show High Diversity

- **Categories B & C (Underspecified/Random):** Near-maximal diversity (~5/5). Models easily conceptualize "different" in these contexts without task anchoring.
- **Categories E, G, H (Subjective Problem-Solving, Creative Writing, Advice/Opinions):** High diversity achievable with task-anchored methods, significantly more than with general methods.

### Smaller vs. Larger Models

- Smaller open-weight models (Llama-3.1-8B, Mistral-7B) tend to have **less homogenization** than larger commercial models under temperature sampling, possibly due to less extensive alignment.
- However, smaller models have much worse task classification accuracy, limiting the effectiveness of task-anchored sampling.

---

## 6. Datasets and Models Used

### Models Evaluated (5 total)

| Model | Type | Task Classification Accuracy |
|---|---|---|
| GPT-4o | Commercial | 82% |
| Claude-4-Sonnet | Commercial | 86% |
| Gemini-2.5-Flash | Commercial | 84% |
| Llama-3.1-8B-Instruct | Open-weight | 56% |
| Mistral-7B-Instruct-v0.3 | Open-weight | 46% |

### Datasets (344 prompts total, from 6 sources)

| Dataset | # Prompts | Primary Categories | Notes |
|---|---|---|---|
| **SimpleQA** (Wei et al., 2024) | 50 | A (Well-Specified) | Short fact-seeking queries; challenging for frontier models (<40% GPT-4o accuracy) |
| **MATH-500** (Lightman et al., 2023) | 50 | D (Problem-Solving Objective) | 10 from each of 5 difficulty levels |
| **MacGyver** (Tian et al., 2024) | 50 | E (Problem-Solving Subjective) | Creative problem-solving; "solvable" problems requiring "unconventional" solutions |
| **NoveltyBench** (Zhang et al., 2025b) | 100 | B, C, G, H | Tasks where multiple distinct high-quality outputs are expected |
| **Community Alignment** (Zhang et al., 2025a) | 50 | F, H (primarily) | User-generated prompts about values, religion, family, politics, culture |
| **WildBench** (Lin et al., 2025) | 44 (6 excluded) | D, F, G, H | Filtered subset of WildChat (1M user-ChatGPT conversations); diverse/challenging |

### Prompt Distribution Across Categories

| Category | Total Prompts |
|---|---|
| A. Well-Specified Objective | 53 |
| B. Underspecified Objective | 16 |
| C. Random Generation | 14 |
| D. Problem-Solving Objective | 55 |
| E. Problem-Solving Subjective | 50 |
| F. Encyclopedia Inquiry | 23 |
| G. Creative Writing | 45 |
| H. Advice or Opinions | 88 |

### LLM-Judge Models
GPT-4o, Claude-4-Sonnet, and Gemini-2.5-Flash serve as judges (averaged), independent of response generation.

### Quality Evaluation
- **Reward Model:** Athene-RM-8B (empirically validated as one of the best for human preferences)
- **Checklist-Based Quality:** LLM-judges generate task-specific quality checklists (3-5 factors), then grade on 1-5 Likert scale

### Alignment Experiments (Appendix A.7)
- Preliminary experiments with DPO and GRPO alignment of Llama-3.1-8B-Instruct
- Also explore DARLING (Diversity Aware Reinforcement Learning) with task-dependent functional diversity judges
- Key finding: While token entropy collapses during alignment, **functional diversity does not necessarily collapse** (Appendix Figures 15-16)
- GRPO with DARLING and task diversity judges shows the most promising results for alignment-time diversity improvement

---

## 7. Implications for CoT Convergence Research

### Direct Relevance

1. **Category D is the CoT-relevant category.** Problem-Solving Objective tasks -- the category most relevant to chain-of-thought reasoning -- show a specific pattern: the final answer should converge, but the **reasoning path/strategy** should be diverse. This is precisely the phenomenon CoT convergence research needs to measure.

2. **Temperature sampling is a poor diversity mechanism for reasoning.** Even at high temperatures, models only generate 2-3 functionally diverse solution strategies for Category D tasks. This suggests that CoT convergence cannot be addressed by simply adjusting temperature -- the model fundamentally lacks diverse reasoning strategies.

3. **General diversity metrics are misleading for reasoning tasks.** Vocabulary diversity and embedding diversity fail to capture whether two chain-of-thought reasoning traces actually employ different strategies. Two solutions using different words but the same strategy would appear "diverse" under general metrics but would be functionally equivalent. This is critical for CoT convergence studies that use embedding similarity to measure convergence.

4. **The diversity-quality tradeoff may not exist for reasoning.** If CoT convergence research shows a tradeoff between diverse reasoning and accuracy, this paper suggests that the tradeoff may be an artifact of using the wrong metrics. With proper task-anchored evaluation, diverse reasoning strategies can be generated without sacrificing quality.

5. **Task-anchored prompting can elicit diverse reasoning paths.** The specific prompt "Can you solve the problem using a different strategy?" is more effective at producing genuinely different CoT traces than generic "generate a different response" instructions. This has implications for best-of-N sampling, self-consistency, and diverse verifier approaches.

6. **Alignment reduces reasoning diversity.** The paper notes that the alignment process (RLHF/DPO) amplifies homogenization. For CoT specifically, this means aligned models may converge on a single reasoning strategy even when multiple valid strategies exist. Preliminary GRPO experiments show that functional diversity for Category D tasks can be maintained or improved during alignment.

7. **Chain-of-thought could improve task-anchored diversity.** The discussion section explicitly suggests: "Our task-anchored sampling strategies could be incorporated into a chain-of-thought instruction, with models first reasoning about task-appropriate functional diversity. A reasoning model could also be trained to directly reason about the functional diversity requirements for a given task before generating a response."

### Methodological Implications for Our Research

- **Measuring CoT convergence requires task-anchored metrics.** Using embedding similarity alone will miss the distinction between "same strategy, different wording" and "different strategy, same answer." We need to evaluate whether reasoning traces are functionally diverse at the strategy level.
- **The graph-based clustering approach** (Definition 3.2) for counting functionally unique responses could be adapted to count the number of truly distinct reasoning strategies a model uses for a given problem.
- **LLM-as-judge for strategy diversity** is validated (~79% agreement with humans). This provides a practical way to evaluate whether CoT traces from different models or different prompting conditions represent genuinely different reasoning approaches.
- **The paper's MATH-500 results provide a baseline.** For math problem-solving, even the best methods only achieve 2-3 distinct strategies out of 5 generations. This sets expectations for what is achievable in CoT diversity.

### Gaps and Opportunities

- The paper does not directly study CoT or reasoning-mode models. Extending the framework to evaluate reasoning traces (not just final responses) would be valuable.
- The paper's Category D only considers "solution strategies" as the unit of functional diversity. For CoT research, we might want finer-grained analysis: different proof techniques, different intermediate representations, different decomposition approaches, etc.
- The alignment experiments (DPO, GRPO, DARLING) are preliminary and only on Llama-3.1-8B. More extensive alignment studies would be needed to understand how reasoning-specific alignment affects CoT diversity.
- The paper shows that GRPO with task diversity judges can improve diversity during alignment -- this could be directly applied to train models with more diverse reasoning capabilities.

---

## Summary of Core Claims

1. **Output homogenization is task-dependent** -- whether homogenization is a problem depends on the task category.
2. **Task-anchored functional diversity** captures meaningful differences that general metrics (vocabulary, embedding, compression) miss entirely.
3. **Task-anchored sampling** (modifying prompts to specify what "different" means per task category) increases diversity where desired while preserving homogenization where desired.
4. **The diversity-quality tradeoff is largely an artifact** of using non-task-dependent metrics for both diversity and quality.
5. **Even token entropy collapse during alignment does not necessarily mean functional diversity collapse** -- a nuanced and important finding.
