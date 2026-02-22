# Does Chain of Thought Cause Models to Converge More?

An empirical study measuring whether Chain of Thought (CoT) prompting increases output similarity across different LLM families.

## Key Findings

- **CoT dramatically increases cross-model convergence** (Cohen's d = 0.93, p < 0.00001): Different LLMs produce 31% more similar outputs when using CoT vs. direct prompting.
- **CoT does NOT increase within-model convergence**: Individual models maintain their diversity across samples — the effect is purely cross-model.
- **The effect is strongest for reasoning tasks** (d = 1.19) but persists for creative (d = 0.76) and opinion tasks (d = 0.90).
- **CoT decreases answer agreement** on reasoning tasks (22.9% vs 44.8%) despite increasing semantic similarity — models "sound alike" but sometimes reach different conclusions.
- **CoT acts as a cross-model homogenizer**: It funnels different architectures toward shared reasoning templates and phrasings.

## Experiment Design

- **4 LLMs**: GPT-4.1, Claude Sonnet 4.5, Gemini 2.5 Flash, Llama 3.1 70B
- **2 conditions**: Direct prompting vs. Zero-shot CoT ("Let's think step by step")
- **3 task types**: Reasoning (BBH), Creative (NoveltyBench), Opinion (WVS)
- **46 questions x 4 models x 2 conditions x 3 samples = 1,104 API calls**
- **Metrics**: Pairwise cosine similarity (sentence-transformers), answer agreement, lexical diversity

## Reproduce

```bash
# Setup
uv venv && source .venv/bin/activate
uv add openai httpx numpy scipy scikit-learn matplotlib seaborn sentence-transformers datasets tqdm

# Set API keys
export OPENAI_API_KEY="..."
export OPENROUTER_API_KEY="..."

# Run
python src/prepare_data.py    # Sample questions from datasets
python src/run_experiment.py  # Query LLMs (~8 min with async)
python src/analyze_results.py # Compute metrics, stats, plots
```

## File Structure

```
├── REPORT.md              # Full research report with results
├── README.md              # This file
├── planning.md            # Experimental design and motivation
├── literature_review.md   # Pre-gathered literature synthesis
├── resources.md           # Catalog of datasets, papers, code
├── src/
│   ├── prepare_data.py    # Dataset sampling
│   ├── run_experiment.py  # LLM API calls (async)
│   └── analyze_results.py # Analysis, stats, visualization
├── results/
│   ├── questions.json     # 46 sampled questions
│   ├── responses.json     # 1,104 model responses
│   └── metrics.json       # Computed metrics and statistical tests
├── figures/
│   ├── cross_model_convergence.png
│   ├── within_model_convergence.png
│   ├── effect_sizes_summary.png
│   ├── model_similarity_heatmap.png
│   └── response_lengths.png
├── datasets/              # Pre-downloaded datasets (BBH, NoveltyBench, WVS)
├── papers/                # Downloaded research papers
└── code/                  # Cloned reference repositories
```

See [REPORT.md](REPORT.md) for the full analysis.
