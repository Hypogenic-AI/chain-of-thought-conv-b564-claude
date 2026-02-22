"""
Analyze convergence: compute semantic similarity, lexical diversity, and answer agreement.
Generates statistical tests and visualizations.
"""

import json
import os
import random
import re
from collections import defaultdict
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sentence_transformers import SentenceTransformer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

BASE_DIR = "/data/hypogenicai/workspaces/chain-of-thought-conv-b564-claude"
RESPONSES_PATH = os.path.join(BASE_DIR, "results", "responses.json")
QUESTIONS_PATH = os.path.join(BASE_DIR, "results", "questions.json")
METRICS_PATH = os.path.join(BASE_DIR, "results", "metrics.json")
FIGURES_DIR = os.path.join(BASE_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


def load_data():
    with open(RESPONSES_PATH) as f:
        responses = json.load(f)
    with open(QUESTIONS_PATH) as f:
        questions = json.load(f)
    return responses, questions


def compute_embeddings(responses, model_name="all-MiniLM-L6-v2"):
    """Compute sentence embeddings for all responses using a local model."""
    print(f"Loading embedding model: {model_name}...")
    model = SentenceTransformer(model_name, device="cuda")

    texts = [r["response"] for r in responses]
    print(f"Computing embeddings for {len(texts)} responses...")
    embeddings = model.encode(texts, batch_size=64, show_progress_bar=True, normalize_embeddings=True)

    return embeddings


def cosine_similarity(a, b):
    """Cosine similarity between two normalized vectors."""
    return float(np.dot(a, b))


def pairwise_cosine_similarities(embeddings_list):
    """Average pairwise cosine similarity for a list of embeddings."""
    if len(embeddings_list) < 2:
        return []
    sims = []
    for i, j in combinations(range(len(embeddings_list)), 2):
        sims.append(cosine_similarity(embeddings_list[i], embeddings_list[j]))
    return sims


def compute_distinct_n(texts, n=1):
    """Compute Distinct-n metric: ratio of unique n-grams to total n-grams."""
    all_ngrams = []
    for text in texts:
        words = text.lower().split()
        ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
        all_ngrams.extend(ngrams)
    if len(all_ngrams) == 0:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_type_token_ratio(texts):
    """Average type-token ratio across texts."""
    ttrs = []
    for text in texts:
        words = text.lower().split()
        if len(words) == 0:
            continue
        ttrs.append(len(set(words)) / len(words))
    return np.mean(ttrs) if ttrs else 0.0


def extract_answer(response, category):
    """Extract the final answer from a response (for BBH reasoning tasks)."""
    if category != "reasoning":
        return response.strip()

    # Look for pattern like "(A)", "(B)", etc. — take the last one
    matches = re.findall(r'\(([A-F])\)', response)
    if matches:
        return f"({matches[-1]})"

    # Try "answer is X" pattern
    match = re.search(r'(?:answer|Answer)\s*(?:is|:)\s*\(?([A-F])\)?', response)
    if match:
        return f"({match.group(1)})"

    return response.strip()[:100]


def analyze_within_model_convergence(responses, embeddings):
    """
    For each (model, condition, question), compute pairwise similarity
    among the N samples. Compare CoT vs direct.
    """
    # Group responses by (model, condition, question_id)
    groups = defaultdict(list)
    for i, r in enumerate(responses):
        key = (r["model"], r["condition"], r["question_id"])
        groups[key].append(i)

    # Compute per-question within-model similarity
    records = []
    for (model, condition, qid), indices in groups.items():
        if len(indices) < 2:
            continue
        embs = [embeddings[i] for i in indices]
        sims = pairwise_cosine_similarities(embs)
        cat = responses[indices[0]]["category"]
        records.append({
            "model": model,
            "condition": condition,
            "question_id": qid,
            "category": cat,
            "mean_similarity": float(np.mean(sims)),
            "n_pairs": len(sims),
        })

    return records


def analyze_cross_model_convergence(responses, embeddings):
    """
    For each (condition, question_id), compute pairwise similarity
    across different models. Compare CoT vs direct.
    """
    # Group by (condition, question_id, model) and take the mean embedding per model
    groups = defaultdict(list)
    for i, r in enumerate(responses):
        key = (r["condition"], r["question_id"], r["model"])
        groups[key].append(i)

    # Get mean embedding per (condition, question, model)
    model_embs = {}
    for (condition, qid, model), indices in groups.items():
        embs = np.array([embeddings[i] for i in indices])
        mean_emb = embs.mean(axis=0)
        mean_emb = mean_emb / np.linalg.norm(mean_emb)
        model_embs[(condition, qid, model)] = mean_emb

    # For each (condition, question), compute cross-model similarity
    records = []
    conditions_questions = defaultdict(set)
    for (condition, qid, model) in model_embs:
        conditions_questions[(condition, qid)].add(model)

    for (condition, qid), models in conditions_questions.items():
        if len(models) < 2:
            continue
        model_list = sorted(models)
        sims = []
        for m1, m2 in combinations(model_list, 2):
            e1 = model_embs[(condition, qid, m1)]
            e2 = model_embs[(condition, qid, m2)]
            sims.append(cosine_similarity(e1, e2))

        # Get category
        cat = None
        for r in responses:
            if r["question_id"] == qid:
                cat = r["category"]
                break

        records.append({
            "condition": condition,
            "question_id": qid,
            "category": cat,
            "mean_cross_similarity": float(np.mean(sims)),
            "n_model_pairs": len(sims),
        })

    return records


def analyze_answer_agreement(responses, questions):
    """For BBH questions, check whether models agree on the answer more with CoT."""
    q_answers = {q["id"]: q.get("answer") for q in questions}

    # Group by (condition, question_id, model)
    groups = defaultdict(list)
    for r in responses:
        if r["category"] == "reasoning":
            key = (r["condition"], r["question_id"], r["model"])
            answer = extract_answer(r["response"], r["category"])
            groups[key].append(answer)

    # For each (condition, question), get majority answer per model, then check agreement
    records = []
    cq_models = defaultdict(dict)
    for (condition, qid, model), answers in groups.items():
        # Majority answer from this model
        from collections import Counter
        majority = Counter(answers).most_common(1)[0][0]
        cq_models[(condition, qid)][model] = majority

    for (condition, qid), model_answers in cq_models.items():
        if len(model_answers) < 2:
            continue
        answers_list = list(model_answers.values())
        # Agreement rate: fraction of model pairs that agree
        pairs = list(combinations(answers_list, 2))
        agreement = sum(1 for a, b in pairs if a == b) / len(pairs) if pairs else 0
        records.append({
            "condition": condition,
            "question_id": qid,
            "agreement_rate": agreement,
            "n_models": len(model_answers),
            "ground_truth": q_answers.get(qid),
        })

    return records


def analyze_lexical_diversity(responses):
    """Compute lexical diversity metrics per (model, condition, question)."""
    groups = defaultdict(list)
    for r in responses:
        key = (r["model"], r["condition"], r["question_id"])
        groups[key].append(r["response"])

    records = []
    for (model, condition, qid), texts in groups.items():
        cat = None
        for r in responses:
            if r["question_id"] == qid:
                cat = r["category"]
                break
        records.append({
            "model": model,
            "condition": condition,
            "question_id": qid,
            "category": cat,
            "distinct_1": compute_distinct_n(texts, 1),
            "distinct_2": compute_distinct_n(texts, 2),
            "ttr": compute_type_token_ratio(texts),
            "avg_length": np.mean([len(t.split()) for t in texts]),
        })

    return records


def statistical_tests(within_model, cross_model, answer_agreement, lexical):
    """Run paired statistical tests comparing CoT vs direct."""
    results = {}

    # 1. Within-model: paired by (model, question)
    print("\n=== Statistical Tests ===\n")

    # Group within-model by (model, question) → compare conditions
    wm_pairs = defaultdict(dict)
    for r in within_model:
        key = (r["model"], r["question_id"])
        wm_pairs[key][r["condition"]] = r["mean_similarity"]

    direct_sims = []
    cot_sims = []
    for key, conditions in wm_pairs.items():
        if "direct" in conditions and "cot" in conditions:
            direct_sims.append(conditions["direct"])
            cot_sims.append(conditions["cot"])

    if len(direct_sims) >= 5:
        stat, p = stats.wilcoxon(cot_sims, direct_sims)
        d = (np.mean(cot_sims) - np.mean(direct_sims)) / np.std(np.array(cot_sims) - np.array(direct_sims))
        print(f"Within-model similarity (CoT vs Direct):")
        print(f"  Direct mean: {np.mean(direct_sims):.4f} ± {np.std(direct_sims):.4f}")
        print(f"  CoT mean:    {np.mean(cot_sims):.4f} ± {np.std(cot_sims):.4f}")
        print(f"  Wilcoxon: W={stat:.0f}, p={p:.6f}")
        print(f"  Cohen's d: {d:.3f}")
        results["within_model"] = {
            "direct_mean": float(np.mean(direct_sims)),
            "direct_std": float(np.std(direct_sims)),
            "cot_mean": float(np.mean(cot_sims)),
            "cot_std": float(np.std(cot_sims)),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "cohens_d": float(d),
            "n_pairs": len(direct_sims),
        }

    # By category
    for cat in ["reasoning", "creative", "opinion"]:
        d_sims = []
        c_sims = []
        for key, conditions in wm_pairs.items():
            qid = key[1]
            r_cat = None
            for r in within_model:
                if r["question_id"] == qid:
                    r_cat = r["category"]
                    break
            if r_cat == cat and "direct" in conditions and "cot" in conditions:
                d_sims.append(conditions["direct"])
                c_sims.append(conditions["cot"])

        if len(d_sims) >= 5:
            stat, p = stats.wilcoxon(c_sims, d_sims)
            d_eff = (np.mean(c_sims) - np.mean(d_sims)) / np.std(np.array(c_sims) - np.array(d_sims)) if np.std(np.array(c_sims) - np.array(d_sims)) > 0 else 0
            print(f"\n  Within-model ({cat}):")
            print(f"    Direct: {np.mean(d_sims):.4f} ± {np.std(d_sims):.4f}")
            print(f"    CoT:    {np.mean(c_sims):.4f} ± {np.std(c_sims):.4f}")
            print(f"    p={p:.6f}, d={d_eff:.3f}")
            results[f"within_model_{cat}"] = {
                "direct_mean": float(np.mean(d_sims)),
                "cot_mean": float(np.mean(c_sims)),
                "p_value": float(p),
                "cohens_d": float(d_eff),
                "n": len(d_sims),
            }

    # 2. Cross-model similarity
    cm_pairs = defaultdict(dict)
    for r in cross_model:
        cm_pairs[r["question_id"]][r["condition"]] = r["mean_cross_similarity"]

    direct_cm = []
    cot_cm = []
    for qid, conditions in cm_pairs.items():
        if "direct" in conditions and "cot" in conditions:
            direct_cm.append(conditions["direct"])
            cot_cm.append(conditions["cot"])

    if len(direct_cm) >= 5:
        stat, p = stats.wilcoxon(cot_cm, direct_cm)
        d = (np.mean(cot_cm) - np.mean(direct_cm)) / np.std(np.array(cot_cm) - np.array(direct_cm)) if np.std(np.array(cot_cm) - np.array(direct_cm)) > 0 else 0
        print(f"\nCross-model similarity (CoT vs Direct):")
        print(f"  Direct mean: {np.mean(direct_cm):.4f} ± {np.std(direct_cm):.4f}")
        print(f"  CoT mean:    {np.mean(cot_cm):.4f} ± {np.std(cot_cm):.4f}")
        print(f"  Wilcoxon: W={stat:.0f}, p={p:.6f}")
        print(f"  Cohen's d: {d:.3f}")
        results["cross_model"] = {
            "direct_mean": float(np.mean(direct_cm)),
            "direct_std": float(np.std(direct_cm)),
            "cot_mean": float(np.mean(cot_cm)),
            "cot_std": float(np.std(cot_cm)),
            "wilcoxon_stat": float(stat),
            "p_value": float(p),
            "cohens_d": float(d),
            "n_pairs": len(direct_cm),
        }

    # By category for cross-model
    for cat in ["reasoning", "creative", "opinion"]:
        d_cm = []
        c_cm = []
        for qid, conditions in cm_pairs.items():
            r_cat = None
            for r in cross_model:
                if r["question_id"] == qid:
                    r_cat = r["category"]
                    break
            if r_cat == cat and "direct" in conditions and "cot" in conditions:
                d_cm.append(conditions["direct"])
                c_cm.append(conditions["cot"])

        if len(d_cm) >= 5:
            stat, p = stats.wilcoxon(c_cm, d_cm)
            d_eff = (np.mean(c_cm) - np.mean(d_cm)) / np.std(np.array(c_cm) - np.array(d_cm)) if np.std(np.array(c_cm) - np.array(d_cm)) > 0 else 0
            print(f"\n  Cross-model ({cat}):")
            print(f"    Direct: {np.mean(d_cm):.4f} ± {np.std(d_cm):.4f}")
            print(f"    CoT:    {np.mean(c_cm):.4f} ± {np.std(c_cm):.4f}")
            print(f"    p={p:.6f}, d={d_eff:.3f}")
            results[f"cross_model_{cat}"] = {
                "direct_mean": float(np.mean(d_cm)),
                "cot_mean": float(np.mean(c_cm)),
                "p_value": float(p),
                "cohens_d": float(d_eff),
                "n": len(d_cm),
            }

    # 3. Answer agreement
    aa_pairs = defaultdict(dict)
    for r in answer_agreement:
        aa_pairs[r["question_id"]][r["condition"]] = r["agreement_rate"]

    direct_aa = []
    cot_aa = []
    for qid, conditions in aa_pairs.items():
        if "direct" in conditions and "cot" in conditions:
            direct_aa.append(conditions["direct"])
            cot_aa.append(conditions["cot"])

    if len(direct_aa) >= 5:
        stat, p = stats.wilcoxon(cot_aa, direct_aa)
        print(f"\nAnswer Agreement (CoT vs Direct, reasoning only):")
        print(f"  Direct mean: {np.mean(direct_aa):.4f}")
        print(f"  CoT mean:    {np.mean(cot_aa):.4f}")
        print(f"  Wilcoxon: p={p:.6f}")
        results["answer_agreement"] = {
            "direct_mean": float(np.mean(direct_aa)),
            "cot_mean": float(np.mean(cot_aa)),
            "p_value": float(p),
            "n": len(direct_aa),
        }

    # 4. Lexical diversity
    lex_pairs = defaultdict(dict)
    for r in lexical:
        key = (r["model"], r["question_id"])
        lex_pairs[key][r["condition"]] = r

    direct_d1 = []
    cot_d1 = []
    direct_d2 = []
    cot_d2 = []
    for key, conditions in lex_pairs.items():
        if "direct" in conditions and "cot" in conditions:
            direct_d1.append(conditions["direct"]["distinct_1"])
            cot_d1.append(conditions["cot"]["distinct_1"])
            direct_d2.append(conditions["direct"]["distinct_2"])
            cot_d2.append(conditions["cot"]["distinct_2"])

    if len(direct_d1) >= 5:
        stat1, p1 = stats.wilcoxon(cot_d1, direct_d1)
        stat2, p2 = stats.wilcoxon(cot_d2, direct_d2)
        print(f"\nLexical Diversity (CoT vs Direct):")
        print(f"  Distinct-1: Direct={np.mean(direct_d1):.4f}, CoT={np.mean(cot_d1):.4f}, p={p1:.6f}")
        print(f"  Distinct-2: Direct={np.mean(direct_d2):.4f}, CoT={np.mean(cot_d2):.4f}, p={p2:.6f}")
        results["lexical_diversity"] = {
            "distinct1_direct": float(np.mean(direct_d1)),
            "distinct1_cot": float(np.mean(cot_d1)),
            "distinct1_p": float(p1),
            "distinct2_direct": float(np.mean(direct_d2)),
            "distinct2_cot": float(np.mean(cot_d2)),
            "distinct2_p": float(p2),
        }

    return results


def plot_within_model(within_model):
    """Plot within-model convergence comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    categories = ["reasoning", "creative", "opinion"]
    titles = ["Reasoning (BBH)", "Creative (NoveltyBench)", "Opinion (WVS)"]

    for ax, cat, title in zip(axes, categories, titles):
        data_direct = [r["mean_similarity"] for r in within_model
                       if r["category"] == cat and r["condition"] == "direct"]
        data_cot = [r["mean_similarity"] for r in within_model
                    if r["category"] == cat and r["condition"] == "cot"]

        if not data_direct or not data_cot:
            ax.set_title(f"{title}\n(no data)")
            continue

        bp = ax.boxplot([data_direct, data_cot], labels=["Direct", "CoT"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#4ECDC4")
        bp["boxes"][1].set_facecolor("#FF6B6B")

        ax.set_title(title, fontsize=13)
        ax.set_ylabel("Within-Model Pairwise Similarity")
        ax.set_ylim(-0.1, 1.1)

    plt.suptitle("Within-Model Convergence: Direct vs CoT", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "within_model_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: within_model_convergence.png")


def plot_cross_model(cross_model):
    """Plot cross-model convergence comparison."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    categories = ["reasoning", "creative", "opinion"]
    titles = ["Reasoning (BBH)", "Creative (NoveltyBench)", "Opinion (WVS)"]

    for ax, cat, title in zip(axes, categories, titles):
        data_direct = [r["mean_cross_similarity"] for r in cross_model
                       if r["category"] == cat and r["condition"] == "direct"]
        data_cot = [r["mean_cross_similarity"] for r in cross_model
                    if r["category"] == cat and r["condition"] == "cot"]

        if not data_direct or not data_cot:
            ax.set_title(f"{title}\n(no data)")
            continue

        bp = ax.boxplot([data_direct, data_cot], labels=["Direct", "CoT"],
                        patch_artist=True, widths=0.5)
        bp["boxes"][0].set_facecolor("#4ECDC4")
        bp["boxes"][1].set_facecolor("#FF6B6B")

        ax.set_title(title, fontsize=13)
        ax.set_ylabel("Cross-Model Pairwise Similarity")
        ax.set_ylim(-0.1, 1.1)

    plt.suptitle("Cross-Model Convergence: Direct vs CoT", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "cross_model_convergence.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: cross_model_convergence.png")


def plot_overall_summary(test_results):
    """Bar chart summary of effect sizes."""
    categories = []
    effect_sizes = []
    colors = []
    labels = []

    for key in ["within_model_reasoning", "within_model_creative", "within_model_opinion",
                "cross_model_reasoning", "cross_model_creative", "cross_model_opinion"]:
        if key in test_results:
            d = test_results[key]["cohens_d"]
            p = test_results[key]["p_value"]
            effect_sizes.append(d)
            nice_key = key.replace("_", " ").title()
            sig = "*" if p < 0.05 else ""
            labels.append(f"{nice_key}{sig}")
            colors.append("#FF6B6B" if d > 0 else "#4ECDC4")

    if not labels:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(range(len(labels)), effect_sizes, color=colors, edgecolor="black", height=0.6)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel("Cohen's d (positive = CoT more similar)", fontsize=12)
    ax.axvline(x=0, color="black", linewidth=1)
    ax.set_title("Effect of CoT on Convergence by Task Type\n(* = p < 0.05)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "effect_sizes_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: effect_sizes_summary.png")


def plot_model_heatmap(responses, embeddings):
    """Heatmap of cross-model similarity for CoT vs Direct."""
    models = sorted(set(r["model"] for r in responses))
    conditions = ["direct", "cot"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, condition in zip(axes, conditions):
        # Get mean embedding per model (averaged over all questions and samples)
        model_embs = {}
        for model in models:
            indices = [i for i, r in enumerate(responses) if r["model"] == model and r["condition"] == condition]
            if indices:
                embs = np.array([embeddings[i] for i in indices])
                mean_emb = embs.mean(axis=0)
                mean_emb = mean_emb / np.linalg.norm(mean_emb)
                model_embs[model] = mean_emb

        # Build similarity matrix
        n = len(models)
        sim_matrix = np.zeros((n, n))
        for i, m1 in enumerate(models):
            for j, m2 in enumerate(models):
                if m1 in model_embs and m2 in model_embs:
                    sim_matrix[i, j] = cosine_similarity(model_embs[m1], model_embs[m2])

        sns.heatmap(sim_matrix, annot=True, fmt=".3f", xticklabels=models, yticklabels=models,
                    cmap="RdYlBu_r", vmin=0, vmax=1, ax=ax, square=True)
        ax.set_title(f"{condition.upper()}", fontsize=13)

    plt.suptitle("Cross-Model Similarity Heatmap", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "model_similarity_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: model_similarity_heatmap.png")


def plot_response_length(responses):
    """Compare response lengths between conditions."""
    data = defaultdict(lambda: defaultdict(list))
    for r in responses:
        length = len(r["response"].split())
        data[r["condition"]][r["category"]].append(length)

    fig, ax = plt.subplots(figsize=(10, 5))
    categories = ["reasoning", "creative", "opinion"]
    x = np.arange(len(categories))
    width = 0.35

    direct_means = [np.mean(data["direct"].get(c, [0])) for c in categories]
    cot_means = [np.mean(data["cot"].get(c, [0])) for c in categories]
    direct_stds = [np.std(data["direct"].get(c, [0])) for c in categories]
    cot_stds = [np.std(data["cot"].get(c, [0])) for c in categories]

    ax.bar(x - width/2, direct_means, width, yerr=direct_stds, label="Direct", color="#4ECDC4",
           capsize=5, edgecolor="black")
    ax.bar(x + width/2, cot_means, width, yerr=cot_stds, label="CoT", color="#FF6B6B",
           capsize=5, edgecolor="black")

    ax.set_xticks(x)
    ax.set_xticklabels(["Reasoning", "Creative", "Opinion"], fontsize=12)
    ax.set_ylabel("Response Length (words)", fontsize=12)
    ax.set_title("Average Response Length by Condition", fontsize=14, fontweight="bold")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "response_lengths.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved: response_lengths.png")


def main():
    print("=" * 60)
    print("CONVERGENCE ANALYSIS")
    print("=" * 60)

    responses, questions = load_data()
    print(f"Loaded {len(responses)} responses for {len(questions)} questions")

    # Compute embeddings
    embeddings = compute_embeddings(responses)

    # Compute metrics
    print("\nComputing within-model convergence...")
    within_model = analyze_within_model_convergence(responses, embeddings)

    print("Computing cross-model convergence...")
    cross_model = analyze_cross_model_convergence(responses, embeddings)

    print("Computing answer agreement...")
    answer_agreement = analyze_answer_agreement(responses, questions)

    print("Computing lexical diversity...")
    lexical = analyze_lexical_diversity(responses)

    # Statistical tests
    test_results = statistical_tests(within_model, cross_model, answer_agreement, lexical)

    # Save all metrics
    all_metrics = {
        "within_model": within_model,
        "cross_model": cross_model,
        "answer_agreement": answer_agreement,
        "lexical_diversity": lexical,
        "statistical_tests": test_results,
    }
    with open(METRICS_PATH, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nMetrics saved to {METRICS_PATH}")

    # Generate plots
    print("\nGenerating visualizations...")
    plot_within_model(within_model)
    plot_cross_model(cross_model)
    plot_overall_summary(test_results)
    plot_model_heatmap(responses, embeddings)
    plot_response_length(responses)

    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
