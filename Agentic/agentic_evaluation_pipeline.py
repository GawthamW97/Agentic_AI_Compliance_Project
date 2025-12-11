import pandas as pd
import json
import csv
import os
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from agentic_hierarchy import agentic_hs_classifier
from agentic_pipeline import agentic
import warnings
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than",
    category=UserWarning
)

def evaluate_agentic_pipeline(
    ground_truth_csv,
    sample_size=1000,
    top_k=20,
    output_dir="./logs/agentic_logs"
):
    print("Begin agentic flow...")

    def safe_int(x):
        """Convert any value to an integer. Invalid values become 0."""
        try:
            return int(str(x).strip())
        except:
            return 0

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_file = os.path.join(output_dir, f"agentic_eval_{timestamp}.csv")
    jsonl_file = os.path.join(output_dir, f"agentic_eval_{timestamp}.jsonl")

    df = pd.read_csv(ground_truth_csv)
    sample_size = len(df)

    y_true = []
    y_pred = []
    top_k_correct = 0
    logs = []

    total_start_time = time.time()

    for idx, row in df.iterrows():

        true_code = safe_int(row["hscode"])
        desc = str(row["description"])

        start_time = time.time()
        result = agentic(desc,top_k)
        latency = time.time() - start_time

        final_code = safe_int(result.get("final_code"))

        # Extract Top-K codes
        hs_codes_k = []
        try:
            hs_codes_k.append(final_code)
        except:
            pass

        if true_code in hs_codes_k:
            top_k_correct += 1

        # Append true + predicted
        y_true.append(true_code)
        y_pred.append(final_code)

        # Build JSONL log entry
        log_entry = {
            "index": idx,
            "true_hscode": true_code,
            "predicted_hscode": final_code,
            "confidence": result.get("confidence", None),
            "latency_seconds": round(latency, 4),
            "product_description": desc,
            "candidates": hs_codes_k
        }

        logs.append(log_entry)

        with open(jsonl_file, "a", encoding="utf8") as f_json:
            f_json.write(json.dumps(log_entry) + "\n")

    #  Compute metrics
    top1_accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    topk_accuracy = top_k_correct / sample_size
    total_eval_time = time.time() - total_start_time

    #  Write CSV summary
    with open(csv_file, "w", newline="", encoding="utf8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "Total Samples",
            "Top-1 Accuracy",
            f"Top-{top_k} Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "Total Eval Time (sec)",
            "Avg Latency (sec)"
        ])
        writer.writerow([
            sample_size,
            round(top1_accuracy, 4),
            round(topk_accuracy, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(total_eval_time, 2),
            round(total_eval_time / sample_size, 4)
        ])

    print("\n========== Agentic Evaluation Complete ==========")
    print(f"Samples: {sample_size}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    print(f"Top-{top_k} Accuracy: {topk_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Total Evaluation Time: {total_eval_time:.2f} sec")
    print(f"Avg Latency: {total_eval_time/sample_size:.4f} sec/sample")
    print(f"CSV Log: {csv_file}")
    print(f"JSONL Log: {jsonl_file}")

    return {
        "top1_accuracy": float(top1_accuracy),
        "topk_accuracy": float(topk_accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "total_eval_time": float(total_eval_time),
        "avg_latency": float(total_eval_time / sample_size),
        "csv_file": csv_file,
        "jsonl_file": jsonl_file
    }

def normalize_hscode(code: str):
    if code is None:
        return None

    # Convert to string safely
    code = str(code).strip()

    # Remove any non-digit characters just in case
    code = ''.join(filter(str.isdigit, code))

    # Pad to 6 digits (HS codes are always 6-digit minimum)
    return code.zfill(6)

def evaluate_agentic_hierarchy(
    ground_truth_csv,
    sample_size=1000,
    top_k=5,
    k6=20,
    output_dir="./logs/agentic_hier_logs"
):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"agentic_eval_{timestamp}.csv")
    jsonl_file = os.path.join(output_dir, f"agentic_eval_{timestamp}.jsonl")


    df = pd.read_csv(ground_truth_csv)
    df["hscode"] = df["hscode"].astype(str)
    # df = df[df["hscode"].str.len() >= 5]    
    # # Random sampling
    # df_sample = df.sample(n=sample_size, random_state=42)

    y_true = []
    y_pred = []
    topk_correct = 0

    total_samples = len(df)
    total_start = time.time()

    # Open JSONL writer once
    jsonl_f = open(jsonl_file, "w", encoding="utf8")

    for idx, row in df.iterrows():

        true_code = normalize_hscode(row["hscode"])
        desc = str(row["description"])

        start = time.time()

        result = agentic_hs_classifier(
            desc,
            k6=k6,
            confidence_threshold=0.75
        )

        latency = time.time() - start

        final_answer = result["final_answer"]
        pred_code = normalize_hscode(final_answer.get("final_code"))
        confidence = final_answer.get("confidence")

        # If fallback failed for some reason (should not happen)
        if pred_code is None:
            pred_code = normalize_hscode(result["retrieved_tree"]["subheadings"][0]["hs6"])

        # Build Top-K list from retrieval
        sub_candidates = [
            normalize_hscode(s["hs6"])
            for s in result["retrieved_tree"]["subheadings"]
        ]

        # Deduplicate while keeping order
        seen = set()
        dedup_candidates = []
        for c in sub_candidates:
            if c not in seen:
                dedup_candidates.append(c)
                seen.add(c)

        topk_candidates = dedup_candidates[:top_k]

        if true_code in topk_candidates:
            topk_correct += 1

        y_true.append(true_code)
        y_pred.append(pred_code)

        # Write JSONL log record
        log_entry = {
            "index": int(idx),
            "true_hscode": true_code,
            "predicted_hscode": pred_code,
            "confidence": confidence,
            "latency_seconds": round(latency, 4),
            "product_description": desc,
            "topk_candidates": topk_candidates,
            "retrieval_tree": result["retrieved_tree"],
            "initial_answer": result["initial_answer"],
            "final_answer": final_answer
        }

        jsonl_f.write(json.dumps(log_entry) + "\n")

    jsonl_f.close()

    # Accuracy, precision, recall, F1:
    top1_acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    # Top-K accuracy
    topk_acc = topk_correct / total_samples

    total_time = time.time() - total_start
    avg_latency = total_time / total_samples

    # Write CSV summary
    with open(csv_file, "w", newline="", encoding="utf8") as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow([
            "Total Samples",
            "Top-1 Accuracy",
            f"Top-{top_k} Accuracy",
            "Precision (macro)",
            "Recall (macro)",
            "F1 Score (macro)",
            "Total Eval Time (sec)",
            "Avg Latency (sec)"
        ])

        writer.writerow([
            total_samples,
            round(top1_acc, 4),
            round(topk_acc, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(total_time, 2),
            round(avg_latency, 4)
        ])

    # Print results
    print("\n========== Agentic 6-Digit First Evaluation ==========")
    print(f"Total samples: {total_samples}")
    print(f"Top-1 Accuracy: {top1_acc:.4f}")
    print(f"Top-{top_k} Accuracy: {topk_acc:.4f}")
    print(f"Precision (macro): {precision:.4f}")
    print(f"Recall (macro): {recall:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")
    print(f"Total Eval Time (sec): {total_time:.2f}")
    print(f"Avg Latency (sec/sample): {avg_latency:.4f}")
    print(f"CSV summary: {csv_file}")
    print(f"JSONL log: {jsonl_file}")

    return {
        "top1_accuracy": top1_acc,
        "topk_accuracy": topk_acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_eval_time": total_time,
        "avg_latency": avg_latency,
        "csv_file": csv_file,
        "jsonl_file": jsonl_file
    }

# Canonical Description evaluation
# evaluate_agentic_pipeline("./dataset/test_hs_codes.csv")

# Additional dataset evaluation
# evaluate_agentic_pipeline("./dataset/test_ext_hs_codes.csv")

#synthetic dataset of product descriptions
# evaluate_agentic_pipeline("./dataset/synthetic_product_descriptions.csv")

# evaluate_agentic_hierarchy("./dataset/test_ext_hs_codes.csv")
# evaluate_agentic_hierarchy("./dataset/synthetic_product_descriptions.csv")