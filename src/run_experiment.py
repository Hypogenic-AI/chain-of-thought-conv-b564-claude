"""
Core experiment: Query multiple LLMs with and without Chain of Thought.
Collects responses for convergence analysis.
Uses async concurrency for speed.
"""

import json
import os
import time
import random
import asyncio
import functools
from datetime import datetime

import openai
import httpx

print = functools.partial(print, flush=True)

SEED = 42
random.seed(SEED)

BASE_DIR = "/data/hypogenicai/workspaces/chain-of-thought-conv-b564-claude"
QUESTIONS_PATH = os.path.join(BASE_DIR, "results", "questions.json")
RESPONSES_PATH = os.path.join(BASE_DIR, "results", "responses.json")

N_SAMPLES = 3
TEMPERATURE = 0.7
MAX_TOKENS = 400

# Model configurations
MODELS = {
    "gpt-4.1": {
        "provider": "openai",
        "model_id": "gpt-4.1",
    },
    "claude-sonnet-4-5": {
        "provider": "openrouter",
        "model_id": "anthropic/claude-sonnet-4-5",
    },
    "gemini-2.5-flash": {
        "provider": "openrouter",
        "model_id": "google/gemini-2.5-flash",
    },
    "llama-3.1-70b": {
        "provider": "openrouter",
        "model_id": "meta-llama/llama-3.1-70b-instruct",
    },
}

DIRECT_SYSTEM = "Answer the question directly and concisely. Do not explain your reasoning."
COT_SYSTEM = "Answer the question by thinking step by step. Show your reasoning before giving your final answer."
DIRECT_SUFFIX = "\n\nAnswer directly and concisely."
COT_SUFFIX = "\n\nLet's think step by step."


def get_openai_client():
    return openai.AsyncOpenAI(api_key=os.environ["OPENAI_API_KEY"])


def get_openrouter_client():
    return openai.AsyncOpenAI(
        api_key=os.environ.get("OPENROUTER_API_KEY", os.environ.get("OPENROUTER_KEY")),
        base_url="https://openrouter.ai/api/v1",
    )


async def call_model(client, model_id, system_msg, user_msg, temperature, max_tokens, semaphore):
    """Make a single async API call with retry logic."""
    async with semaphore:
        for attempt in range(4):
            try:
                response = await client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content
            except Exception as e:
                wait = min(2 ** attempt * 2, 30)
                print(f"    Retry {attempt+1}: {model_id}: {str(e)[:80]}")
                await asyncio.sleep(wait)
    return None


async def run_experiment():
    """Run the full experiment with async concurrency."""
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)

    print(f"Loaded {len(questions)} questions")
    print(f"Models: {list(MODELS.keys())}")
    print(f"Samples per condition: {N_SAMPLES}")
    total_calls = len(questions) * len(MODELS) * 2 * N_SAMPLES
    print(f"Total API calls: {total_calls}")

    # Load existing responses if any (for resume)
    responses = []
    completed_keys = set()
    if os.path.exists(RESPONSES_PATH):
        with open(RESPONSES_PATH) as f:
            responses = json.load(f)
        for r in responses:
            completed_keys.add((r["question_id"], r["model"], r["condition"], r["sample_idx"]))
        print(f"Resuming: {len(responses)} responses already collected")

    openai_client = get_openai_client()
    openrouter_client = get_openrouter_client()

    # Semaphores to limit concurrency per provider
    openai_sem = asyncio.Semaphore(5)
    openrouter_sem = asyncio.Semaphore(8)

    start_time = time.time()
    completed = 0

    # Process questions in batches for manageable concurrency
    for qi, question in enumerate(questions):
        qid = question["id"]
        q_text = question["question"]
        print(f"\n[Q {qi+1}/{len(questions)}] {qid} ({question['category']})")

        tasks = []
        task_info = []

        for model_name, model_cfg in MODELS.items():
            client = openai_client if model_cfg["provider"] == "openai" else openrouter_client
            sem = openai_sem if model_cfg["provider"] == "openai" else openrouter_sem
            model_id = model_cfg["model_id"]

            for condition in ["direct", "cot"]:
                system_msg = DIRECT_SYSTEM if condition == "direct" else COT_SYSTEM
                user_msg = q_text + (DIRECT_SUFFIX if condition == "direct" else COT_SUFFIX)

                for sample_idx in range(N_SAMPLES):
                    key = (qid, model_name, condition, sample_idx)
                    if key in completed_keys:
                        continue

                    task = call_model(client, model_id, system_msg, user_msg,
                                      TEMPERATURE, MAX_TOKENS, sem)
                    tasks.append(task)
                    task_info.append({
                        "question_id": qid,
                        "category": question["category"],
                        "model": model_name,
                        "model_id": model_id,
                        "condition": condition,
                        "sample_idx": sample_idx,
                    })

        if not tasks:
            print(f"  Skipping (already done)")
            continue

        # Run all calls for this question concurrently
        results = await asyncio.gather(*tasks)

        for info, result in zip(task_info, results):
            if result is not None:
                record = {**info, "response": result, "timestamp": datetime.now().isoformat()}
                responses.append(record)
                completed += 1
            else:
                print(f"  FAILED: {info['model']}/{info['condition']}/{info['sample_idx']}")

        # Save after each question
        with open(RESPONSES_PATH, "w") as f:
            json.dump(responses, f, indent=2)

        elapsed = time.time() - start_time
        rate = completed / elapsed if elapsed > 0 else 0
        remaining = total_calls - len(completed_keys) - completed
        eta = remaining / rate if rate > 0 else 0
        print(f"  Done: {completed} calls, {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    elapsed = time.time() - start_time
    print(f"\nExperiment complete!")
    print(f"  Total responses: {len(responses)}")
    print(f"  Time: {elapsed:.0f}s")

    from collections import Counter
    model_counts = Counter(r["model"] for r in responses)
    cond_counts = Counter(r["condition"] for r in responses)
    cat_counts = Counter(r["category"] for r in responses)
    print(f"\n  By model: {dict(model_counts)}")
    print(f"  By condition: {dict(cond_counts)}")
    print(f"  By category: {dict(cat_counts)}")


if __name__ == "__main__":
    asyncio.run(run_experiment())
