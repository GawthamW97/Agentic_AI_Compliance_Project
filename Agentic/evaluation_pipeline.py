import pandas as pd
import json
import csv
import os
import time
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from agentic_hierarchy import hierarchical_retrieval,agent_reason,reflect_and_correct,agentic_hs_classifier
from flat_agentic import flat_rag_retrieve,agent_reason_flat,reflect_and_correct_flat,agentic_hs_classifier_flat
from agentic_pipeline import agentic_hier
import warnings
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than",
    category=UserWarning
)

def evaluate_agentic_pipeline(
    ground_truth_csv,
    sample_size=500,
    top_k=5,
    output_dir="./agentic_logs"
):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    csv_file = os.path.join(output_dir, f"agentic_eval_{timestamp}.csv")
    jsonl_file = os.path.join(output_dir, f"agentic_eval_{timestamp}.jsonl")

    df = pd.read_csv(ground_truth_csv)
    df["hscode"] = df["hscode"].astype(str)
    df = df[df["hscode"].str.len() >= 5]    
    # Random sampling
    df_sample = df.sample(n=sample_size, random_state=42)

    y_true = []
    y_pred = []
    topk_correct = 0
    logs = []

    total_start_time = time.time()

    for idx, row in df_sample.iterrows():
        true_code = int(row["hscode"])
        desc = str(row["description"])

        start_time = time.time()
        result = agentic_hier(desc)
        latency = time.time() - start_time

        final_code = int(result["final_code"])
        confidence = None


        # Retrieve all subheading candidates
        # sub_candidates = [
        #     x[0] for x in result["retrieved_tree"]["subheadings"]
        # ]

        # # Track Top-K hit
        # if true_code in sub_candidates[:top_k]:
        #     topk_correct += 1

        # Append metric vectors
        y_true.append(true_code)
        y_pred.append(final_code if final_code else "UNCERTAIN")


        # Build log entry
        log_entry = {
            "index": idx,
            "true_hscode": true_code,
            "predicted_hscode": final_code,
            "confidence": confidence,
            "latency_seconds": round(latency, 4),
            # "chapters": result["retrieved_tree"]["chapters"],
            # "headings": result["retrieved_tree"]["headings"],
            # "subheadings": result["retrieved_tree"]["subheadings"],
            # "initial_answer": result["initial_answer"],
            # "final_answer": result["final_answer"],
            "product_description": desc
        }

        logs.append(log_entry)

        # Write each sample log to JSONL (streaming)
        with open(jsonl_file, "a", encoding="utf8") as f_json:
            f_json.write(json.dumps(log_entry) + "\n")

    # ---- Compute metrics ----
    filtered_idx = [i for i, p in enumerate(y_pred) if p != "UNCERTAIN"]
    if filtered_idx:
        y_true_filtered = [y_true[i] for i in filtered_idx]
        y_pred_filtered = [y_pred[i] for i in filtered_idx]
    else:
        y_true_filtered = y_true
        y_pred_filtered = y_pred

    top1_accuracy = accuracy_score(y_true_filtered, y_pred_filtered)
    precision = precision_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
    recall = recall_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average="macro", zero_division=0)

    # topk_accuracy = topk_correct / sample_size

    total_eval_time = time.time() - total_start_time

    # ---- Write CSV summary log ----
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
            # round(topk_accuracy, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(total_eval_time, 2),
            round(total_eval_time / sample_size, 4)
        ])

    print("\n========== Agentic Evaluation Complete ==========")
    print(f"Samples: {sample_size}")
    print(f"Top-1 Accuracy: {top1_accuracy:.4f}")
    # print(f"Top-{top_k} Accuracy: {topk_accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"Total Evaluation Time: {total_eval_time:.2f} sec")
    print(f"Avg Latency: {total_eval_time/sample_size:.4f} sec/sample")
    print(f"CSV Log: {csv_file}")
    print(f"JSONL Log: {jsonl_file}")

    return {
        "top1_accuracy": top1_accuracy,
        # "topk_accuracy": topk_accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "total_eval_time": total_eval_time,
        "avg_latency": total_eval_time / sample_size,
        "csv_file": csv_file,
        "jsonl_file": jsonl_file
    }

def evaluate_agentic_flat(
    ground_truth_csv,
    sample_size=100,
    top_k=5,
    k_retrieval=30,
    output_dir="./agentic_flat_logs"
):
    """
    Evaluate flat-RAG + Agentic AI on a random sample of HS-coded products.

    df: must contain columns ['hscode', 'description'].

    Produces:
      - Top-1 accuracy
      - Top-K accuracy
      - Precision / Recall / F1 (macro)
      - Total eval time
      - Avg latency

    Logs:
      - CSV summary
      - JSONL detailed logs (one per sample)
    """

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(output_dir, f"agentic_flat_eval_{timestamp}.csv")
    jsonl_file = os.path.join(output_dir, f"agentic_flat_eval_{timestamp}.jsonl")

    df = pd.read_csv(ground_truth_csv)
    df["hscode"] = df["hscode"].astype("string")
    df = df[df["hscode"].str.len() >= 5]

    df_sample = df.sample(
        n=min(sample_size, len(df)),
        random_state=42
    )

    y_true = []
    y_pred = []
    topk_correct = 0
    total_samples = len(df_sample)

    total_start = time.time()

    jsonl_f = open(jsonl_file, "w", encoding="utf8")

    for idx, row in df_sample.iterrows():
        true_code = str(row["hscode"])
        desc = str(row["description"])

        start = time.time()
        result = agentic_hs_classifier_flat(
            desc,
            k_retrieval=k_retrieval,
            confidence_threshold=0.5
        )
        latency = time.time() - start

        final = result["final_answer"]
        pred_code = final.get("final_code") or "UNCERTAIN"
        confidence = final.get("confidence", None)

        # Build Top-K candidate list from RAG retrieval
        candidate_codes = []
        for c in result["candidates"]:
            if c["hs_code"] not in candidate_codes:
                candidate_codes.append(c["hs_code"])
        topk_candidates = candidate_codes[:top_k]

        if true_code in topk_candidates:
            topk_correct += 1

        y_true.append(true_code)
        y_pred.append(pred_code)

        log_entry = {
            "index": int(idx),
            "product_description": desc,
            "true_hscode": true_code,
            "predicted_hscode": pred_code,
            "confidence": confidence,
            "latency_seconds": round(latency, 4),
            "topk_candidates": topk_candidates,
            "candidates": result["candidates"],
            "initial_answer": result["initial_answer"],
            "final_answer": result["final_answer"]
        }

        jsonl_f.write(json.dumps(log_entry) + "\n")

    jsonl_f.close()

    # Filter out UNCERTAIN for classification metrics
    valid_idx = [i for i, p in enumerate(y_pred) if p != "UNCERTAIN"]
    if valid_idx:
        y_true_f = [y_true[i] for i in valid_idx]
        y_pred_f = [y_pred[i] for i in valid_idx]
    else:
        y_true_f = y_true
        y_pred_f = y_pred

    top1_acc = accuracy_score(y_true_f, y_pred_f)
    precision = precision_score(y_true_f, y_pred_f, average="macro", zero_division=0)
    recall = recall_score(y_true_f, y_pred_f, average="macro", zero_division=0)
    f1 = f1_score(y_true_f, y_pred_f, average="macro", zero_division=0)
    topk_acc = topk_correct / total_samples

    total_time = time.time() - total_start
    avg_latency = total_time / total_samples

    # CSV summary
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

    print("\n========== Flat-RAG + Agentic Evaluation ==========")
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

def normalize_hscode(code: str):
    """
    Ensure HS code is treated as a 6-digit zero-padded string.
    Handles cases where code is int, float, or None.
    """
    if code is None:
        return None

    # Convert to string safely
    code = str(code).strip()

    # Remove any non-digit characters just in case
    code = ''.join(filter(str.isdigit, code))

    # Pad to 6 digits (HS codes are always 6-digit minimum)
    return code.zfill(6)

# Canonical Description evaluation
# evaluate_agentic_pipeline("./dataset/harmonized-system.csv")

# Additional dataset evaluation
evaluate_agentic_pipeline("./dataset/test_hs_codes.csv")

# evaluate_agentic_flat("./dataset/test_hs_codes.csv")