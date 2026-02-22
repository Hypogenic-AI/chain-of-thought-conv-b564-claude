# Paper Notes: Wei et al. 2022 & Wenger & Kenett 2025

## Paper 1: Chain-of-Thought Prompting Elicits Reasoning in Large Language Models

- **Authors:** Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, Denny Zhou (Google Research, Brain Team)
- **Published:** NeurIPS 2022; arXiv:2201.11903
- **Read chunks:** 1-4 (intro, methodology, results, ablation, symbolic reasoning, discussion/conclusion) and 15 (appendix -- annotator C exemplars)

### Definition of Chain of Thought (CoT)

- A **chain of thought** is defined as "a coherent series of intermediate reasoning steps that lead to the final answer for a problem."
- CoT prompting = few-shot prompting where each exemplar is a triple: (input, chain of thought, output).
- The chain of thought resembles a step-by-step solution written in natural language, inserted *before* the final answer.
- No finetuning is involved -- CoT is elicited purely via in-context exemplars in an off-the-shelf language model.

### Methodology

- **Approach:** Augment standard few-shot prompts with manually written chains of thought in the exemplars. For arithmetic tasks, 8 hand-written exemplars were used (no prompt engineering); for AQuA, 4 exemplars from the training set.
- **Models evaluated:** Five model families:
  1. GPT-3 (350M, 1.3B, 6.7B, 175B via InstructGPT variants)
  2. LaMDA (422M, 2B, 8B, 68B, 137B)
  3. PaLM (8B, 62B, 540B)
  4. UL2 20B
  5. Codex (code-davinci-002)
- **Decoding:** Greedy decoding (no sampling). LaMDA results averaged over 5 random seeds (different exemplar orderings).
- **Baseline:** Standard few-shot prompting (input-output pairs only, no intermediate reasoning).

### Benchmarks

**Arithmetic Reasoning (Section 3):**
1. GSM8K -- math word problems (Cobbe et al., 2021)
2. SVAMP -- math word problems with varying structures
3. ASDiv -- diverse math word problems
4. AQuA -- algebraic word problems (multiple choice)
5. MAWPS -- math word problem benchmark

**Commonsense Reasoning (Section 4):**
1. CSQA -- commonsense question answering
2. StrategyQA -- multi-hop strategy inference
3. Date Understanding (BIG-bench)
4. Sports Understanding (BIG-bench)
5. SayCan -- natural language to robot action sequences

**Symbolic Reasoning (Section 5):**
1. Last letter concatenation (e.g., "Amy Brown" -> "yn")
2. Coin flip / state tracking

### Key Results

1. **CoT is an emergent ability of model scale.** CoT prompting does NOT help small models; it only yields gains with models of ~100B+ parameters. Smaller models produce fluent but illogical chains of thought.

2. **Larger gains on harder problems.** For GSM8K (lowest baseline performance), CoT more than doubled accuracy for the largest GPT and PaLM models. For easy single-step problems (SingleOp subset of MAWPS), improvements were negligible or negative.

3. **State-of-the-art results.** PaLM 540B with CoT prompting achieved new SOTA on GSM8K (56.9% solve rate, up from 33% standard prompting; surpassing finetuned GPT-3 at 55%), as well as on SVAMP and MAWPS. On AQuA and ASDiv, within 2% of SOTA.

4. **Commonsense reasoning improvements.** PaLM 540B + CoT set new SOTA on StrategyQA (75.6% vs. 69.4% prior best) and outperformed a sports enthusiast on Sports Understanding (95.4% vs. 84%). Gain was minimal on CSQA.

5. **Symbolic reasoning & OOD generalization.** CoT enabled near-100% in-domain solve rates on last-letter concatenation and coin flip. It also facilitated length generalization to out-of-domain sequences (longer than seen in exemplars), though with reduced performance.

### Ablation Studies (Section 3.3)

Three ablations tested on LaMDA 137B and PaLM 540B:

1. **Equation only:** Prompt model to output just a mathematical equation before the answer. Result: Does NOT help much on GSM8K (semantics too complex to translate directly to equations), but helps on simpler 1-2 step problems.

2. **Variable compute only:** Replace chain of thought with a sequence of dots (...) equal to the number of characters in the needed equation. Result: Performs about the same as baseline. This shows that just having more intermediate tokens (variable computation) is NOT sufficient -- the natural language reasoning content matters.

3. **Chain of thought after answer:** Provide the CoT *after* the final answer instead of before. Result: Performs about the same as baseline. This shows the model actually depends on the sequential reasoning in the CoT to arrive at the correct answer; it is not merely activating relevant knowledge.

### Robustness (Section 3.4)

- Tested with chains of thought written by three different annotators (A, B, C) with different styles.
- All annotator variants substantially outperformed standard prompting, though there was variance.
- Concise chains of thought also worked.
- Three sets of 8 exemplars randomly sampled from GSM8K training set also outperformed standard prompting.
- Robust to different exemplar orderings and varying numbers of exemplars.

### Error Analysis

- Of 50 correct GSM8K answers (LaMDA 137B): 48/50 chains of thought were logically and mathematically correct; 2 coincidentally arrived at correct answer.
- Of 50 incorrect answers: 46% had almost-correct CoT (minor errors: calculator mistake, symbol mapping error, one step missing); 54% had major errors (semantic understanding, coherence failures).
- Scaling from PaLM 62B to 540B fixed many one-step-missing and semantic understanding errors.

### Limitations Noted by Authors

1. CoT emulates human thought processes but does not prove the model is "reasoning" -- left as open question.
2. Manual annotation cost is minimal for few-shot but would be prohibitive for finetuning.
3. No guarantee of correct reasoning paths -- can lead to both correct and incorrect final answers.
4. Emergence only at large scale makes it costly to deploy in practice.

### Conclusions

- CoT reasoning is an emergent property of model scale.
- Standard prompting provides only a lower bound on LLM capabilities.
- CoT is a simple, broadly applicable method requiring no finetuning and no task-specific model checkpoints.

---

## Paper 2: We're Different, We're the Same: Creative Homogeneity Across LLMs

- **Authors:** Emily Wenger (Duke University), Yoed Kenett (Technion)
- **Published:** arXiv:2501.19361, January 2025
- **Read chunks:** 1-4 (full main content through additional analysis)

### Core Research Question

Prior work showed that using a *single* LLM (typically ChatGPT) as a creative partner leads to homogeneous creative outputs. But does this homogeneity stem from using one particular model, or is it a property of LLMs *in general*? Would using *different* LLMs restore creative diversity?

### Methodology

**Approach:** Elicit creative responses from both humans and a broad set of LLMs using three standardized creativity tests, then compare population-level diversity of responses.

**Why standardized creativity tests?** They produce structured output formats, allowing the authors to disambiguate similarity in response *structure* from similarity in response *content*. This is critical because LLMs may share output quirks (e.g., tense, verbosity) that could inflate apparent similarity.

### Creativity Tests Used

1. **Guilford's Alternative Uses Test (AUT):** Present an object (book, fork, table, hammer, pants) and ask for creative uses. Scored on fluency, originality, flexibility, elaboration. Five objects tested.

2. **Forward Flow (FF):** Provide a starting word (candle, table, bear, snow, toaster) and ask for a chain of 20 successive free-association words. Measures how much thought diverges from the starting point. Five start words tested.

3. **Divergent Association Task (DAT):** Ask subjects to list 10 words that are as unrelated as possible. Single prompt (no variation).

### Models Evaluated

- **22 LLMs total** from public APIs, including: AI21-Jamba-Instruct, Cohere Command R/R+, Meta Llama 3 (8B, 70B), Meta Llama 3.1 (8B, 70B, 405B), Mistral (large, large 2407, Nemo, small), Google Gemini 1.5, GPT-4o, GPT-4o-mini, Phi 3 (multiple variants), Phi 3.5 mini.
- **For statistical tests, restricted to 7 distinct-family models** to control for within-family similarity: AI21 Jamba 1.5 Large, Google Gemini 1.5, Cohere Command R Plus, Meta Llama 3 70B Instruct, Mistral Large, GPT-4o, Phi 3 medium 128k Instruct.
- Default system prompt: "You are a helpful assistant."

### Human Subjects

- 102 participants (after filtering from 114) from Prolific platform, IRB-approved.
- Demographics: diverse in age (18-55+), gender (51% female, 46% male, 3% non-binary), race.
- Safeguards against bots: attention checks, response time filters (<5 min excluded), post-hoc inspection.
- Secondary validation: public pre-LLM-era datasets for AUT, FF, and DAT.

### Evaluation Metrics

**Individual originality (O_t):**
- Measures how novel each response is relative to the prompt.
- Automated scoring using GloVe 840B word embeddings and cosine distance, following established methods that correlate with human scorer rankings.

**Population-level variability (V_t):**
- Key metric. Measures semantic distances *between* responses from different individuals in a population.
- Uses sentence embeddings (all-MiniLM-L6-v2 from sentence_transformers) to embed each individual's full response set as a single vector.
- Computes cosine distance between all pairs of population members.
- V_t skewing toward 0 = homogeneous (similar responses); toward 1 = diverse (variable responses).
- Statistical comparison: Welch's t-test, significance threshold p = 0.01, reports effect size (Cohen's d) and test power.

### Key Results

#### Result 1: Individual Originality -- LLMs roughly match humans
- AUT: LLMs slightly outperform humans (mean 0.711 vs. 0.696, small effect size 0.1)
- FF: Humans slightly outperform LLMs (mean 0.637 vs. 0.603, medium effect size 0.52)
- DAT: LLMs slightly outperform humans (mean 0.801 vs. 0.753, large effect size 0.77)
- Overall: Roughly comparable individual creativity, consistent with prior work.

#### Result 2 (KEY FINDING): Population-level variability -- LLMs are far more homogeneous
- **AUT:** LLM mean variability 0.459 vs. Human 0.738 (effect size 2.2, p = 3.9e-80)
- **FF:** LLM mean variability 0.534 vs. Human 0.835 (effect size 2.0, p = 2.8e-66)
- **DAT:** LLM mean variability 0.665 vs. Human 0.819 (effect size 1.4, p = 6.2e-11)
- All differences are statistically significant with very large effect sizes (all > 1.0) and test power of 1.0.
- LLM responses cluster together tightly in t-SNE visualizations of sentence embeddings; human responses are much more spread out.

#### Result 3: Word overlap partially explains homogeneity
- After removing stopwords, LLM response pairs share far more words in common than human response pairs across all tests.

### Additional Analysis (Section 5)

#### Controlling for AUT response structure
- Prompt engineering to match LLM response lengths to human response lengths (3 prompt versions).
- Even when restricting to single-word AUT responses only (eliminating all structural confounds): LLM variability = 0.708 vs. Human = 0.850 (effect size 1.1, p = 2.3e-19).
- Conclusion: It is the *substance*, not the *structure*, of LLM responses that is homogeneous.

#### Within-family similarity (Llama models)
- Models in the same family (5 Llama variants) exhibit slightly lower response diversity than models from different families, though the mean difference was not statistically significant.
- Visual inspection of distributions shows a leftward shift (more homogeneous) for same-family models.

#### Effect of system prompt on creativity
- Tested four system prompts: baseline, "more creative", "very creative" (with $200 incentive framing), and "not creative".
- Creative system prompts slightly increase individual LLM creativity and inter-LLM variability, but human responses are still more variable.

### Implications

- If all popular LLMs produce similar creative outputs, then using *any* LLM as a creative partner -- regardless of which one -- may drive users toward a limited, converging set of "creative" outputs.
- This is a "generative monoculture" problem: homogeneity among LLMs could lead to collective narrowing of societal creativity, bias propagation, and other downstream harms.
- Feature space alignment / "feature universality" observed in LLM internals may be the underlying cause.

---

## Cross-Paper Connections and Relevance to Research Question

### Does Chain of Thought cause models to converge more?

These two papers address complementary aspects of the convergence question:

1. **Wei et al. 2022** establishes that CoT prompting is a powerful technique that dramatically changes LLM behavior at scale -- it unlocks reasoning abilities that standard prompting cannot. However, the paper does not examine whether CoT causes outputs to become more or less *diverse* across models or across runs. The ablation studies show that the specific natural language content of the chain matters (not just extra tokens or post-hoc reasoning), but diversity/homogeneity is not measured.

2. **Wenger & Kenett 2025** provides strong evidence that LLMs are already creatively homogeneous *without* explicit CoT -- different LLMs from different families, given the same creative prompts, produce remarkably similar outputs. The paper uses standardized creativity tests (AUT, FF, DAT) that do not involve CoT-style reasoning.

### Open Questions for Our Research

- **Does adding CoT prompting to creative or open-ended tasks increase or decrease cross-model homogeneity?** Wei et al. show CoT constrains the reasoning pathway to intermediate steps, which could either (a) further converge outputs by funneling models through similar reasoning paths, or (b) diversify outputs by enabling more varied reasoning strategies.

- **Does CoT-induced reasoning structure impose a convergent template?** The ablation showing that "variable compute only" (dots) does not help suggests the *content* of CoT matters. If all models learn similar reasoning strategies from pretraining, CoT could amplify convergence by making models explicitly traverse similar reasoning chains.

- **Scale dependence:** Wei et al. show CoT only works at ~100B+ parameters. Wenger & Kenett tested models of varying sizes. Is homogeneity worse among the largest models where CoT is most effective?

- **Structural vs. substantive homogeneity under CoT:** Wenger & Kenett carefully control for structural similarity. A similar analysis is needed for CoT outputs -- do models produce similar reasoning *steps* and similar final answers, or just similar surface forms?

- **Connection to feature universality:** Wenger & Kenett cite work on feature space alignment across LLMs as a possible mechanism for creative homogeneity. If CoT operates through similar internal representations across models (as suggested by mechanistic interpretability work), this could be a mechanistic link between CoT and convergence.
