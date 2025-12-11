from agentic_evaluation_pipeline import evaluate_agentic_pipeline

#------------------ DATATSET 02 ------------------
print("DATATSET 02 results")
# #Flat RAG + Agentic AI
evaluate_agentic_pipeline("./dataset/test_ext_hs_codes.csv",top_k=20)


#------------------ Synthetic DATASET ------------------
print("Synthetic Dataset results")
# #Flat RAG + Agentic AI
evaluate_agentic_pipeline("./dataset/synthetic_product_descriptions.csv",top_k=20)