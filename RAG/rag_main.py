from rag_evaluation_pipeline import main_RAG_Evaluation, main_RAG_LLM_Evaluation

#------------------ DATATSET 02 ------------------
print("DATATSET 02 results")
# #RAG
main_RAG_Evaluation("./dataset/test_ext_hs_codes.csv",14)
# #RAG + LLM
main_RAG_LLM_Evaluation("./dataset/test_ext_hs_codes.csv",14)


#------------------ Synthetic DATASET ------------------
print("Synthetic Dataset results")
#RAG
main_RAG_Evaluation("./dataset/synthetic_product_descriptions.csv",14)
# #RAG + LLM
main_RAG_LLM_Evaluation("./dataset/synthetic_product_descriptions.csv",14)
