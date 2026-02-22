"""
Prepare question sets from BBH, NoveltyBench, and WVS datasets.
Samples a balanced subset for the convergence experiment.
"""

import json
import random
import os
from datasets import load_from_disk

SEED = 42
random.seed(SEED)

BASE_DIR = "/data/hypogenicai/workspaces/chain-of-thought-conv-b564-claude"
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
OUTPUT_PATH = os.path.join(BASE_DIR, "results", "questions.json")


def load_bbh_questions(n_per_task=2, max_tasks=8):
    """Load BBH questions, sampling across diverse reasoning tasks."""
    bbh_dir = os.path.join(DATASETS_DIR, "bigbenchhard")

    # Select tasks that require diverse reasoning types
    target_tasks = [
        "date_understanding",
        "causal_judgement",
        "disambiguation_qa",
        "logical_deduction_five_objects",
        "navigate",
        "reasoning_about_colored_objects",
        "sports_understanding",
        "web_of_lies",
    ]

    questions = []
    for task in target_tasks[:max_tasks]:
        task_dir = os.path.join(bbh_dir, task, "test")
        if not os.path.isdir(task_dir):
            print(f"  Skipping {task}: no test dir")
            continue
        ds = load_from_disk(task_dir)
        indices = random.sample(range(len(ds)), min(n_per_task, len(ds)))
        for idx in indices:
            item = ds[idx]
            questions.append({
                "id": f"bbh_{task}_{idx}",
                "category": "reasoning",
                "source": "bbh",
                "task": task,
                "question": item["input"],
                "answer": item["target"],
            })

    print(f"  BBH: {len(questions)} questions from {len(target_tasks)} tasks")
    return questions


def load_noveltybench_questions(n=15):
    """Load creative prompts from NoveltyBench."""
    ds = load_from_disk(os.path.join(DATASETS_DIR, "noveltybench", "curated"))
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    questions = []
    for idx in indices:
        item = ds[idx]
        questions.append({
            "id": f"nb_{item['id']}",
            "category": "creative",
            "source": "noveltybench",
            "question": item["prompt"],
            "answer": None,
        })

    print(f"  NoveltyBench: {len(questions)} questions")
    return questions


def load_wvs_questions(n=15):
    """Load opinion/value questions from World Values Survey."""
    with open(os.path.join(DATASETS_DIR, "wvs", "wvs.json")) as f:
        data = json.load(f)

    indices = random.sample(range(len(data)), min(n, len(data)))

    questions = []
    for idx in indices:
        item = data[idx]
        questions.append({
            "id": f"wvs_{item['Q_id']}",
            "category": "opinion",
            "source": "wvs",
            "question": item["question"],
            "options": item["option"],
            "answer": None,
        })

    print(f"  WVS: {len(questions)} questions")
    return questions


def main():
    print("Preparing question sets...")

    bbh = load_bbh_questions(n_per_task=2, max_tasks=8)
    nb = load_noveltybench_questions(n=15)
    wvs = load_wvs_questions(n=15)

    all_questions = bbh + nb + wvs
    print(f"\nTotal: {len(all_questions)} questions")
    print(f"  Reasoning: {sum(1 for q in all_questions if q['category'] == 'reasoning')}")
    print(f"  Creative:  {sum(1 for q in all_questions if q['category'] == 'creative')}")
    print(f"  Opinion:   {sum(1 for q in all_questions if q['category'] == 'opinion')}")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(all_questions, f, indent=2)

    print(f"\nSaved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
