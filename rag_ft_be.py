from time import time
from fastapi import FastAPI, Request
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel
from typing import List
import torch
import faiss
import json
import numpy as np
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import string
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from sentence_transformers import SentenceTransformer
import time as t
from gaurdrails import InputQueryValidator,NumericAnswerGuard
# Load all resources at startup
print("\n🔧 Loading resources...")

# FAISS index
faiss_index = faiss.read_index("faiss_index.idx")

# Document metadata
with open("faiss_doc_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Full documents (assumed same order)
with open("dataset_netflix/output_texts/documents_with_metadata.jsonl", "r", encoding="utf-8") as f:
    documents = [json.loads(line.strip()) for line in f]

# BM25 setup
stop_words = set(stopwords.words("english"))
tokenized_corpus = [word_tokenize(doc["text"].lower()) for doc in documents]
bm25 = BM25Okapi(tokenized_corpus)

# Sentence embedding model
embed_model = SentenceTransformer("all-MiniLM-L6-v2")


# TinyLlama base model for RAG
llm_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
adapter_dir = "lora-tinyllama-raft"
tokenizer = AutoTokenizer.from_pretrained(llm_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
llm = AutoModelForCausalLM.from_pretrained(
    llm_name,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
).to(device)
# Finetuned model (base + LoRA adapter)
model = PeftModel.from_pretrained(llm, adapter_dir).to(device)

# FastAPI app
app = FastAPI()

# Request schema
class QueryRequest(BaseModel):
    query: str
    top_k: int = 5

# Helper functions
def preprocess(text):
    tokens = word_tokenize(text.lower())
    return [t for t in tokens if t not in stop_words and t not in string.punctuation]

def hybrid_retrieve(query, top_k=5):
    # Preprocess
    tokens = preprocess(query)
    query_vec = embed_model.encode([query]).astype("float32")

    # FAISS search
    dense_scores, dense_idxs = faiss_index.search(query_vec, top_k)
    dense_result = {i: dense_scores[0][rank] for rank, i in enumerate(dense_idxs[0])}
    print(dense_result)
    # BM25 search
    sparse_scores = bm25.get_scores(tokens)
    sparse_top = np.argsort(sparse_scores)[::-1][:top_k]
    sparse_result = {i: sparse_scores[i] for i in sparse_top}
    print(sparse_scores)
    # # Union
    final_indices = set(dense_result.keys()).union(sparse_result.keys())
    print(final_indices)
    results = []
    for idx in final_indices:
        results.append({
            "doc": documents[idx],
            "faiss_score": dense_result.get(idx),
            "bm25_score": sparse_result.get(idx)
        })
    return results
def make_serializable(results):
    def convert(val):
        if isinstance(val, (np.floating, np.float32, np.float64)):
            return float(val)
        elif isinstance(val, (np.integer, np.int32, np.int64)):
            return int(val)
        return val

    serializable_results = []
    for item in results:
        doc = item["doc"]
        faiss_score = convert(item.get("faiss_score"))
        bm25_score = convert(item.get("bm25_score"))
        serializable_results.append({
            "doc": doc,
            "faiss_score": faiss_score,
            "bm25_score": bm25_score
        })
    return serializable_results

def generate_response(query, retrieved_docs, max_input_tokens=2048, max_new_tokens=200):
    context_blocks = []
    for i, item in enumerate(retrieved_docs):
        doc = item["doc"]
        context_blocks.append(f"[Document {i+1}]\n{doc['text'].strip()}\n")
    
    prompt = f"""You are a helpful assistant. Answer the question using the following context.

{''.join(context_blocks)}

Question: {query}
Answer:"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_input_tokens).to(device)
    with torch.no_grad():
        output_ids = llm.generate(
            inputs["input_ids"],
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id
        )

    output = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output.split("Answer:")[-1].strip()

@app.post("/rag")
def rag_endpoint(request: QueryRequest):
    start_time = t.time()
    print(f"\n🧠 New query: {request.query}")
    if not InputQueryValidator(request.query):
        return {
            "answer": "Query is out of scope for this financial assistant.",
            "confidence": 0.0,
            "method": "Blocked by Validator",
            "response_time": round(t.time() - start_time, 2),
            "retrieved_docs": []
        }
    retrieved = hybrid_retrieve(request.query, request.top_k)
    serializable_retrieved = make_serializable(retrieved)

     # Response generation
    raw_answer = generate_response(request.query, retrieved)
    numeric_answer = NumericAnswerGuard(raw_answer)

    # Confidence estimation
    faiss_scores = [r.get("faiss_score") or 0 for r in serializable_retrieved]
    bm25_scores = [r.get("bm25_score") or 0 for r in serializable_retrieved]
    norm_faiss = [s / max(faiss_scores) if max(faiss_scores) > 0 else 0 for s in faiss_scores]
    norm_bm25 = [s / max(bm25_scores) if max(bm25_scores) > 0 else 0 for s in bm25_scores]
    hybrid_confidence = sum(norm_faiss + norm_bm25) / (2 * len(serializable_retrieved))
    response_time = round(t.time() - start_time, 2)

    return {
        "answer": numeric_answer,
        "raw_answer": raw_answer,  # Optional: include for debugging
        "confidence": round(hybrid_confidence, 2),
        "method": "Hybrid (FAISS + BM25)",
        "response_time": response_time,
        "retrieved_docs": serializable_retrieved
    }
# Request schema
class FTQuery(BaseModel):
    question: str


# Inference function
def generate_finetuned_answer(question, max_new_tokens=64):
    prompt = f"Answer the question with a number only.\nQuestion: {question}\nAnswer:"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id
        )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response.split("Answer:")[-1].strip()


# API endpoint
@app.post("/finetuned")
def finetuned_inference(request: FTQuery):
    start_time = t.time()
    if not InputQueryValidator(request.question):
        return {
            "answer": "Query is out of scope for this financial assistant.",
            "confidence": 0.0,
            "method": "Blocked by Validator",
            "response_time": round(t.time() - start_time, 2)
        }
    raw_answer = generate_finetuned_answer(request.question)
    numeric_answer = NumericAnswerGuard(raw_answer)
    response_time = round(t.time() - start_time, 2)

    return {
        "answer": numeric_answer,
        "confidence": 0.95,  # Optional: replace with actual estimator if available
        "method": "Fine-Tuned",
        "response_time": response_time
    }