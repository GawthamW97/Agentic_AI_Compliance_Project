import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from openai import OpenAI
from dotenv import load_dotenv
from rag_pipeline import rag_only,rag_with_llm
from rag_default import classify_code_without_llm_v2
import chromadb
import os
import json

import warnings
warnings.filterwarnings(
    "ignore",
    message="The number of unique classes is greater than",
    category=UserWarning
)

def evaluate_rag_pipeline(
    ground_truth_csv,
    k=3,
    confidence_threshold=None,
    output_file="./output/rag_predictions.csv",
    collection="hs_codes"
):

    df = pd.read_csv(ground_truth_csv)

    # df = df.sample(n=1000, random_state=42)
    y_true = []
    y_pred_top1 = []
    top_k_correct = 0
    total = len(df)

    prediction_logs = []

    for idx, row in df.iterrows():
        desc = str(row["description"])
        true_code = str(row["hscode"])

        results = rag_only(desc)
        top1 = results[0]

        is_low_confidence = (
            confidence_threshold and top1["similarity"] < confidence_threshold
        )
        if is_low_confidence:
            pred_code = "UNCERTAIN"
        else:
            pred_code = top1["hs_code"]

        y_true.append(true_code)
        y_pred_top1.append(pred_code)

        # Top-K match check
        hs_codes_k = [r["hs_code"] for r in results]
        if true_code in hs_codes_k:
            top_k_correct += 1

        # Log predictions
        prediction_logs.append({
            "index": idx,
            "product_description": desc,
            "true_hscode": true_code,
            "predicted_hscode": pred_code,
            "top1_similarity": round(top1["similarity"], 4),
            "is_low_confidence": is_low_confidence,
            "top_k_candidates": json.dumps([{
                "rank": r["rank"],
                "hs_code": r["hs_code"],
                "similarity": round(r["similarity"], 4)
            } for r in results])
        })

    valid_idx = [i for i, p in enumerate(y_pred_top1) if p != "UNCERTAIN"]
    if valid_idx:
        y_true_filtered = [y_true[i] for i in valid_idx]
        y_pred_filtered = [y_pred_top1[i] for i in valid_idx]
    else:
        y_true_filtered = y_true
        y_pred_filtered = y_pred_top1

    acc = accuracy_score(y_true_filtered, y_pred_filtered)
    prec = precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    rec = recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    topk_acc = top_k_correct / total

    print("\n========== Evaluation Results ==========")
    print(f"Total samples: {total}")
    print(f"Top-1 Accuracy: {acc:.4f}")
    print(f"Top-{k} Accuracy: {topk_acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")

    hier_metrics = compute_hierarchical_metrics(y_true_filtered, y_pred_filtered)

    print("\n========== Hierarchical (Aggregated) Metrics ==========")
    for level, vals in hier_metrics.items():
        print(f"{level.upper()} → "
              f"Acc: {vals['accuracy']:.4f}, "
              f"Prec: {vals['precision']:.4f}, "
              f"Rec: {vals['recall']:.4f}, "
              f"F1: {vals['f1']:.4f}")

    # if confidence_threshold:
    #     print(f"Confidence threshold applied: {confidence_threshold}")
    #     print(f"Samples classified as UNCERTAIN: {total - len(valid_idx)}")

    # print("\nDetailed Classification Report:")
    # print(classification_report(y_true_filtered, y_pred_filtered, zero_division=0))

    # Save prediction log to CSV
    pd.DataFrame(prediction_logs).to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")

    return {
        "top1_accuracy": acc,
        "topk_accuracy": topk_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "total_samples": total,
        "classified_samples": len(valid_idx)
    }

def evaluate_rag_llm_pipeline(
    ground_truth_csv,
    k=5,
    model="gpt-4o-mini",
    output_file="rag_llm_predictions.csv",
    collection="hs_codes"
):

    df = pd.read_csv(ground_truth_csv)
    # df = df.sample(n=1000,random_state=42)
    y_true = []
    y_pred = []
    top_k_correct = 0
    total = len(df)

    prediction_logs = []

    for idx, row in df.iterrows():
        desc = str(row["description"])
        true_code = str(row["hscode"])

        pred_code, llm_raw, candidates = rag_with_llm(desc)

        retrieved_codes = [c["hs_code"] for c in candidates]
        if true_code in retrieved_codes:
            top_k_correct += 1

        y_true.append(true_code)
        y_pred.append(pred_code)

        prediction_logs.append({
            "index": idx,
            "product_description": desc,
            "true_hscode": true_code,
            "predicted_hscode": pred_code,
            "top_k_candidates": json.dumps(candidates),
            "llm_raw_output": llm_raw
        })

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    topk_acc = top_k_correct / total

    print("\n========== RAG + LLM Evaluation ==========")
    print(f"Total samples: {total}")
    print(f"Top-1 Accuracy (LLM): {acc:.4f}")
    print(f"Top-{k} Retrieval Accuracy: {topk_acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")

    # print("\nDetailed Classification Report:")
    # print(classification_report(y_true, y_pred, zero_division=0))

    pd.DataFrame(prediction_logs).to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")

    return {
        "top1_accuracy": acc,
        "topk_accuracy": topk_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "total_samples": total
    }

def evaluate_rag_pipeline_hybrid(
    ground_truth_csv,
    k=3,
    confidence_threshold=None,
    output_file="./output/rag_predictions_hybrid.csv",
    retriever_func=None,   
    alpha=0.7            
):

    df = pd.read_csv(ground_truth_csv)
    y_true, y_pred_top1 = [], []
    top_k_correct, total = 0, len(df)
    prediction_logs = []

    for idx, row in df.iterrows():
        desc = str(row["description"])
        true_code = str(row["hscode"])

        results = retriever_func(desc, k=k)
        top1 = results[0]

        if "hybrid_score" in top1:
            confidence = top1["hybrid_score"]
        elif "semantic_score" in top1 and "lexical_score" in top1:
            confidence = alpha * top1["semantic_score"] + (1 - alpha) * top1["lexical_score"]
        else:
            confidence = top1.get("similarity", 0)

        # Confidence threshold check
        is_low_confidence = (
            confidence_threshold is not None and confidence < confidence_threshold
        )

        if is_low_confidence:
            pred_code = "UNCERTAIN"
        else:
            pred_code = top1.get("hs_code", top1.get("id", "UNKNOWN"))

        y_true.append(true_code)
        y_pred_top1.append(pred_code)

        hs_codes_k = [r.get("hs_code", r.get("id")) for r in results]
        if true_code in hs_codes_k:
            top_k_correct += 1

        def safe_round(value):
            try:
                return round(float(value), 4)
            except:
                return None

        prediction_logs.append({
            "index": idx,
            "product_description": desc,
            "true_hscode": true_code,
            "predicted_hscode": pred_code,
            "semantic_score": safe_round(top1.get("semantic_score")),
            "lexical_score": safe_round(top1.get("lexical_score")),
            "hybrid_score": safe_round(top1.get("hybrid_score")),
            "confidence": safe_round(confidence),
            "is_low_confidence": is_low_confidence,
            "top_k_candidates": json.dumps([
                {
                    "rank": r.get("rank"),
                    "hs_code": r.get("hs_code", r.get("id")),
                    "semantic_score": safe_round(r.get("semantic_score")),
                    "lexical_score": safe_round(r.get("lexical_score")),
                    "hybrid_score": safe_round(r.get("hybrid_score")),
                }
                for r in results
            ])
        })


    valid_idx = [i for i, p in enumerate(y_pred_top1) if p != "UNCERTAIN"]
    if valid_idx:
        y_true_filtered = [y_true[i] for i in valid_idx]
        y_pred_filtered = [y_pred_top1[i] for i in valid_idx]
    else:
        y_true_filtered = y_true
        y_pred_filtered = y_pred_top1

    acc = accuracy_score(y_true_filtered, y_pred_filtered)
    prec = precision_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    rec = recall_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    f1 = f1_score(y_true_filtered, y_pred_filtered, average='macro', zero_division=0)
    topk_acc = top_k_correct / total

    print("\n========== Evaluation Results ==========")
    print(f"Total samples: {total}")
    print(f"Top-1 Accuracy: {acc:.4f}")
    print(f"Top-{k} Accuracy: {topk_acc:.4f}")
    print(f"Precision (macro): {prec:.4f}")
    print(f"Recall (macro): {rec:.4f}")
    print(f"F1 Score (macro): {f1:.4f}")

    hier_metrics = compute_hierarchical_metrics(y_true_filtered, y_pred_filtered)

    print("\n========== Hierarchical (Aggregated) Metrics ==========")
    for level, vals in hier_metrics.items():
        print(f"{level.upper()} → "
              f"Acc: {vals['accuracy']:.4f}, "
              f"Prec: {vals['precision']:.4f}, "
              f"Rec: {vals['recall']:.4f}, "
              f"F1: {vals['f1']:.4f}")

    if confidence_threshold:
        print(f"Confidence threshold applied: {confidence_threshold}")
        print(f"Samples classified as UNCERTAIN: {total - len(valid_idx)}")

    pd.DataFrame(prediction_logs).to_csv(output_file, index=False)
    print(f"\nPredictions saved to: {output_file}")

    return {
        "top1_accuracy": acc,
        "topk_accuracy": topk_acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "total_samples": total,
        "classified_samples": len(valid_idx),
        "hierarchical_metrics": hier_metrics
    }


def compute_hierarchical_metrics(y_true, y_pred):
    levels = [2, 4, 6]
    results = {}

    for digits in levels:
        y_true_lvl = [truncate_code(c, digits) for c in y_true]
        y_pred_lvl = [truncate_code(c, digits) for c in y_pred]

        acc = accuracy_score(y_true_lvl, y_pred_lvl)
        prec = precision_score(y_true_lvl, y_pred_lvl, average='macro', zero_division=0)
        rec = recall_score(y_true_lvl, y_pred_lvl, average='macro', zero_division=0)
        f1 = f1_score(y_true_lvl, y_pred_lvl, average='macro', zero_division=0)

        results[f"{digits}-digit"] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

    return results

def truncate_code(code, digits=2):
    if not isinstance(code, str):
        code = str(code)
    return code[:digits]

# evaluate_rag_pipeline("./dataset/harmonized-system.csv",3,collection="hs_codes_hierarchy")
# evaluate_rag_pipeline("./dataset/test_hs_codes.csv",3)
# evaluate_rag_pipeline("./dataset/test_hs_codes.csv",3,collection="hs_codes_hierarchy")
evaluate_rag_pipeline("./dataset/synthetic_product_descriptions.csv",3)
# evaluate_rag_pipeline("./dataset/synthetic_product_descriptions.csv",3,collection="hs_codes_hierarchy")

# evaluate_rag_llm_pipeline("./dataset/test_hs_codes.csv",3)
# evaluate_rag_llm_pipeline("./dataset/test_hs_codes.csv",3,collection="hs_codes_hierarchy")
# evaluate_rag_llm_pipeline("./dataset/synthetic_product_descriptions.csv",3)
# evaluate_rag_llm_pipeline("./dataset/synthetic_product_descriptions.csv",3,collection="hs_codes_hierarchy")

# evaluate_rag_pipeline_hybrid("./dataset/test_hs_codes.csv",3,retriever_func=classify_code_without_llm_v2)

