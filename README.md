# Agentic AI vs RAG for Automated HS Classification & Compliance

> **Production-ready AI research project demonstrating advanced RAG, Agentic AI, and LLM evaluation for regulated enterprise systems (ERP/CRM, trade & tax compliance).**

---

## 🚀 Why This Project Matters (Recruiter Summary)

This project demonstrates **end-to-end applied AI engineering**, not just model experimentation. It shows how Large Language Models can be **safely, reliably, and measurably deployed** in **compliance‑critical enterprise environments**.

**What this proves:**
- You can design **AI systems, not just prompts**
- You understand **evaluation, metrics, and reproducibility**
- You can mitigate **hallucinations in real-world AI workflows**
- You can integrate AI into **ERP/CRM-style architectures**

---

## 🧠 Core Skills Demonstrated

**AI / ML**
- Retrieval-Augmented Generation (RAG)
- Agentic AI (multi-agent reasoning, hierarchical workflows)
- Hallucination detection & mitigation
- Embedding-based semantic search
- Prompt engineering & reasoning control

**Data & Evaluation**
- Experimental design & benchmarking
- Top-K accuracy, Precision, Recall, F1
- Latency & performance trade-off analysis
- Deterministic vs stochastic evaluation handling

**Engineering**
- Python modular architecture
- Vector databases (ChromaDB)
- YAML-based configuration management
- Structured logging (CSV / JSONL)
- Reproducible pipelines

**Enterprise & Compliance Context**
- HS / CN code classification
- EU VAT & TARIC alignment
- ERP/CRM integration patterns
- GDPR-aware system design

---

## 🎯 Project Objective

To **compare and quantify** the effectiveness of three AI architectures for automated product classification:

1. **Baseline RAG** – fast, retrieval-only
2. **RAG + LLM** – retrieval with reasoning
3. **Agentic AI (Proposed)** – self-verifying multi-agent system

The goal is to identify which approach delivers the **best balance of accuracy, reliability, and latency** for enterprise adoption.

---

## 🏗️ High-Level Architecture

```text
Input Product Description
        │
        ▼
Embedding Model (OpenAI)
        │
        ▼
Vector DB (ChromaDB)
        │
        ├── RAG (Retrieval Only)
        ├── RAG + LLM (Reasoning)
        └── Agentic AI (Hierarchical Agents)
                │
                ▼
        Final HS/CN Code Prediction
```

---

## 📂 Repository Structure

```text
Agentic_AI_Compliance_Project/
│
├── dataset/                # Synthetic & ground-truth datasets
├── RAG/                    # Baseline & RAG+LLM pipelines
├── Agentic/                # Multi-agent hierarchical pipeline
├── config/                 # YAML-based configuration
├── evaluation_logs/        # CSV & JSONL experiment outputs
├── main.py                 # Unified experiment runner
└── README.md
```

---

## 📊 Key Results (Headline)

| Architecture | Top‑1 Accuracy | F1 Score | Reliability |
|-------------|---------------|---------|------------|
| RAG | Low–Medium | Low | Retrieval‑dependent |
| RAG + LLM | Medium | Medium | Prompt‑sensitive |
| **Agentic AI** | **High** | **High** | **Most consistent** |

**Takeaway:** Agentic AI delivers **substantially better correctness and stability**, at the cost of higher—but predictable—latency.

---

## 🧪 Evaluation Highlights

- Fixed test sets to ensure fair comparison
- Multiple Top‑K retrieval depths (3, 7, 14, 20)
- Case‑based testing across description complexity
- Full experiment logs for auditability

This mirrors **industry-grade ML validation**, not academic toy experiments.

---


## 💼 Real-World Use Cases

- Automated product classification in ERP systems
- VAT & TARIC validation pipelines
- AI-assisted customs & trade compliance
- Safer LLM deployment in regulated domains


---

## 👤 Author

**Gawtham Wayne**  
MSc Data Analytics – National College of Ireland  
Focus: Applied AI, LLM Systems
