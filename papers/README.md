# Downloaded Papers

## Core Chain-of-Thought Papers

1. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** ([PDF](2201.11903_wei2022_chain_of_thought_prompting.pdf))
   - Authors: Wei et al. (Google Research)
   - Year: 2022 (NeurIPS)
   - arXiv: 2201.11903
   - Why relevant: Foundational CoT paper. Defines CoT, establishes it as emergent ability at 100B+ params. Ablation studies show reasoning content matters (not just extra computation).

## CoT and Output Diversity / Convergence

2. **Language of Thought Shapes Output Diversity in Large Language Models** ([PDF](2601.11227_kim2026_language_thought_output_diversity.pdf))
   - Authors: Xu & Zhang (SUTD)
   - Year: 2026
   - arXiv: 2601.11227
   - Why relevant: **Most directly relevant.** Shows English CoT creates "convergence basins" in thinking space. Non-English thinking increases diversity (r=0.72-0.88 correlation).

3. **LLM Output Homogenization is Task Dependent** ([PDF](2509.21267_padmakumar2025_homogenization_task_dependent.pdf))
   - Authors: Jain et al. (MIT / FAIR at Meta)
   - Year: 2025
   - arXiv: 2509.21267
   - Why relevant: 8-category task taxonomy for homogenization. Problem-solving tasks achieve only 2-3 distinct strategies. Temperature scaling ineffective for reasoning diversity.

4. **We're Different, We're the Same: Creative Homogeneity Across LLMs** ([PDF](2501.19361_chakrabarty2025_creative_homogeneity_across_llms.pdf))
   - Authors: Wenger & Kenett (Duke / Technion)
   - Year: 2025
   - arXiv: 2501.19361
   - Why relevant: 22 LLMs show dramatically lower creative variability than humans (effect sizes 1.4-2.2). Establishes baseline cross-model homogeneity.

5. **Homogenization Effects of Large Language Models on Human Creative Ideation** ([PDF](2402.01536_anderson2024_homogenization_creative_ideation.pdf))
   - Authors: Anderson et al.
   - Year: 2024
   - arXiv: 2402.01536
   - Why relevant: Early evidence of LLM homogenization in creative tasks.

6. **The Homogenizing Effect of Large Language Models on Human Expression and Thought** ([PDF](2508.01491_mirka2025_homogenizing_effect_llms.pdf))
   - Authors: Mirka et al.
   - Year: 2025
   - arXiv: 2508.01491
   - Why relevant: Comprehensive review of LLM homogenization across domains (research ideas, essays, survey responses, creative ideation, art).

## CoT Faithfulness

7. **Language Models Don't Always Say What They Think** ([PDF](2305.04388_turpin2023_unfaithful_explanations_cot.pdf))
   - Authors: Turpin et al. (NYU / Anthropic)
   - Year: 2023 (NeurIPS)
   - arXiv: 2305.04388
   - Why relevant: Foundational unfaithfulness paper. CoT can hide biases that drive convergence on wrong answers.

8. **Reasoning Models Don't Always Say What They Think** ([PDF](2505.05410_baker2025_reasoning_models_dont_say_think.pdf))
   - Authors: Chen et al. (Anthropic)
   - Year: 2025
   - arXiv: 2505.05410
   - Why relevant: Extends to reasoning models. 61-75% unfaithful. Outcome-based RL plateaus. Reward hacking invisible in CoT.

9. **Chain-of-Thought Reasoning In The Wild Is Not Always Faithful** ([PDF](2503.08679_bao2025_cot_wild_not_faithful.pdf))
   - Authors: Bao et al.
   - Year: 2025
   - arXiv: 2503.08679
   - Why relevant: Measures unfaithfulness rates (0.04-13%) in realistic settings without artificial perturbations.

10. **Dissociation of Faithful and Unfaithful Reasoning in LLMs** ([PDF](2405.15092_chen2024_dissociation_faithful_unfaithful.pdf))
    - Authors: Chen et al.
    - Year: 2024
    - arXiv: 2405.15092
    - Why relevant: Analyzes mechanisms behind faithful vs. unfaithful CoT.

11. **Measuring Chain-of-Thought Faithfulness and Verbosity** ([PDF](2510.27378_liu2025_measuring_cot_faithfulness_verbosity.pdf))
    - Authors: Liu et al.
    - Year: 2025
    - arXiv: 2510.27378
    - Why relevant: Novel unlearning-based faithfulness metric.

12. **Is Chain-of-Thought Reasoning of LLMs Faithful?** ([PDF](2508.01191_su2025_is_cot_reasoning_faithful.pdf))
    - Authors: Su et al.
    - Year: 2025 (ACL Findings)
    - arXiv: 2508.01191
    - Why relevant: Three faithfulness measurement approaches with open-source code.

## CoT Internal Representations

13. **Knowing Before Saying: LLM Representations Encode Information About CoT Success** ([PDF](2505.24362_afzal2025_knowing_before_saying.pdf))
    - Authors: Afzal et al. (TU Munich / Nvidia)
    - Year: 2025 (ACL Findings)
    - arXiv: 2505.24362
    - Why relevant: Models predict CoT success before generating any tokens (60-76.4% accuracy). Suggests convergence happens at representation level, not during CoT.

14. **How does Chain of Thought Think? Mechanistic Interpretability of CoT** ([PDF](2507.22928_chen2025_mechanistic_interp_cot.pdf))
    - Authors: Chen et al.
    - Year: 2025
    - arXiv: 2507.22928
    - Why relevant: Uses sparse autoencoders to probe CoT internal representations.

15. **Towards Faithful Chain-of-Thought: LLMs are Bridging Reasoners** ([PDF](2405.18915_xiao2024_towards_faithful_cot.pdf))
    - Authors: Xiao et al.
    - Year: 2024
    - arXiv: 2405.18915
    - Why relevant: Explores approaches to make CoT more faithful.

## Surveys and Broader Context

16. **Latent Chain-of-Thought Reasoning: A Comprehensive Survey** ([PDF](2505.16782_xu2025_latent_cot_reasoning_survey.pdf))
    - Authors: Xu et al.
    - Year: 2025
    - arXiv: 2505.16782
    - Why relevant: Survey of latent (non-verbalized) CoT approaches.

17. **Towards Reasoning Era: A Survey of Long Chain-of-Thought for Reasoning LLMs** ([PDF](2503.09567_sui2025_towards_reasoning_era_survey.pdf))
    - Authors: Sui et al.
    - Year: 2025
    - arXiv: 2503.09567
    - Why relevant: Comprehensive survey of long CoT reasoning approaches.

18. **Representation Engineering for LLMs: Survey** ([PDF](2502.17601_chen2025_representation_engineering_survey.pdf))
    - Authors: Chen et al.
    - Year: 2025
    - arXiv: 2502.17601
    - Why relevant: Survey of techniques for understanding and manipulating LLM representations.

## Other Relevant Papers

19. **Active Prompting with Chain-of-Thought for LLMs** ([PDF](2302.12246_diao2023_active_prompting_cot.pdf))
    - Authors: Diao et al.
    - Year: 2023 (ACL 2024)
    - arXiv: 2302.12246
    - Why relevant: Methods for adaptively selecting CoT exemplars.

20. **Flow of Reasoning: Training LLMs for Divergent Reasoning** ([PDF](2406.05673_yu2024_flow_reasoning_divergent.pdf))
    - Authors: Yu et al.
    - Year: 2024
    - arXiv: 2406.05673
    - Why relevant: Explores divergent (diverse) reasoning in LLMs.

21. **Measuring Faithfulness in Chain-of-Thought Reasoning** ([PDF](2307.13702_lanham2023_measuring_faithfulness_cot.pdf))
    - Authors: Lanham et al. (Anthropic)
    - Year: 2023
    - arXiv: 2307.13702
    - Why relevant: Early systematic measurement of CoT faithfulness with multiple intervention methods.
