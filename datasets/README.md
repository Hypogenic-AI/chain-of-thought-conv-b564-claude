# Downloaded Datasets

This directory contains datasets for the research project on whether Chain of Thought causes models to converge more. Data files are NOT committed to git due to size. Follow the download instructions below.

## Dataset 1: BIG-Bench Hard (BBH)

### Overview
- **Source**: HuggingFace (`lukaemon/bbh`)
- **Size**: 3,187 examples across 13 tasks
- **Format**: HuggingFace Dataset (arrow files)
- **Task**: Multi-task reasoning (logical deduction, arithmetic, word sorting, etc.)
- **Splits**: test only
- **License**: Apache 2.0

### Download Instructions

**Using HuggingFace:**
```python
from datasets import load_dataset

tasks = [
    'boolean_expressions', 'causal_judgement', 'date_understanding',
    'disambiguation_qa', 'logical_deduction_five_objects',
    'multistep_arithmetic_two', 'navigate', 'object_counting',
    'reasoning_about_colored_objects', 'tracking_shuffled_objects_three_objects',
    'word_sorting', 'web_of_lies', 'sports_understanding'
]

for task in tasks:
    ds = load_dataset("lukaemon/bbh", task)
    ds.save_to_disk(f"datasets/bigbenchhard/{task}")
```

### Loading the Dataset
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/bigbenchhard/boolean_expressions")
```

### Sample Data
```json
{
  "input": "not ( ( not not True ) ) is",
  "target": "False"
}
```

### Notes
- CoT and standard prompts available via the original BBH paper
- 13 of the 27 BBH tasks are included (the most commonly studied ones)
- Used extensively in CoT faithfulness research (Turpin et al. 2023)

---

## Dataset 2: MMLU-Pro

### Overview
- **Source**: HuggingFace (`TIGER-Lab/MMLU-Pro`)
- **Size**: 12,032 test + 70 validation examples
- **Format**: HuggingFace Dataset
- **Task**: Multi-domain knowledge questions (10 multiple-choice options)
- **Splits**: test, validation
- **License**: MIT

### Download Instructions

```python
from datasets import load_dataset
ds = load_dataset("TIGER-Lab/MMLU-Pro")
ds.save_to_disk("datasets/mmlu_pro/data")
```

### Loading the Dataset
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/mmlu_pro/data")
```

### Sample Data
See `datasets/mmlu_pro/samples.json` for examples.

### Notes
- Designed to require CoT reasoning (performance drops 19% without CoT)
- 10 answer options per question (more challenging than standard MMLU's 4)
- Used in Chen et al. (2025) for faithfulness evaluation
- Covers 14 subject domains

---

## Dataset 3: NoveltyBench

### Overview
- **Source**: Copied from `code/multilingual-lot-diversity/data/novelty-bench/`
- **Size**: 100 open-ended questions (curated + wildchat splits)
- **Format**: HuggingFace Dataset (arrow files)
- **Task**: Diversity evaluation — no ground-truth answers
- **Splits**: curated, wildchat

### Download Instructions

```python
# Option 1: Load from the multilingual-lot-diversity repo
from datasets import load_from_disk
ds = load_from_disk("datasets/noveltybench")

# Option 2: Clone the repo and copy
# git clone https://github.com/iNLP-Lab/Multilingual-LoT-Diversity.git
# cp -r Multilingual-LoT-Diversity/data/novelty-bench/ datasets/noveltybench/
```

### Loading the Dataset
```python
from datasets import load_from_disk
ds = load_from_disk("datasets/noveltybench")
curated = ds['curated']
wildchat = ds['wildchat']
```

### Notes
- Primary benchmark for measuring output diversity in LLMs
- Used in Xu & Zhang (2026) and Jain et al. (2025)
- Questions designed to have multiple valid, distinct answers

---

## Dataset 4: BLEND (Cultural Knowledge)

### Overview
- **Source**: Copied from `code/multilingual-lot-diversity/data/blend.json`
- **Size**: 402 multiple-choice questions
- **Format**: JSON
- **Task**: Cultural knowledge questions with country-mapped options

### Download Instructions

```python
# Clone the repo and copy
# git clone https://github.com/iNLP-Lab/Multilingual-LoT-Diversity.git
# cp Multilingual-LoT-Diversity/data/blend.json datasets/blend/

import json
with open("datasets/blend/blend.json") as f:
    data = json.load(f)
```

### Notes
- Measures cultural pluralism via normalized entropy over country distribution
- Used in Xu & Zhang (2026) for evaluating language-of-thought effects on cultural diversity

---

## Dataset 5: World Values Survey (WVS)

### Overview
- **Source**: Copied from `code/multilingual-lot-diversity/data/wvs.json`
- **Size**: 283 multiple-choice questions
- **Format**: JSON
- **Task**: Value/opinion questions with predefined options

### Download Instructions

```python
# Clone the repo and copy
# git clone https://github.com/iNLP-Lab/Multilingual-LoT-Diversity.git
# cp Multilingual-LoT-Diversity/data/wvs.json datasets/wvs/

import json
with open("datasets/wvs/wvs.json") as f:
    data = json.load(f)
```

### Notes
- Measures opinion diversity via normalized entropy over value-option distribution
- Used in Xu & Zhang (2026); WVS showed largest gains from Mixed-Language Sampling (+10.3 to +21.0)

---

## Dataset Summary

| Name | Size | Task | Primary Use |
|------|------|------|-------------|
| BIG-Bench Hard | 3,187 | Reasoning | CoT vs. standard prompting comparison |
| MMLU-Pro | 12,032 | Knowledge QA | CoT-dependent performance |
| NoveltyBench | 100 | Diversity eval | Output diversity measurement |
| BLEND | 402 | Cultural knowledge | Cultural convergence |
| WVS | 283 | Value questions | Opinion convergence |

**Note:** GPQA requires authenticated HuggingFace access and could not be downloaded without a token. It can be obtained from `Idavidrein/gpqa` on HuggingFace with proper authentication.
