# =====================================================================
# Step 4: NVIDIA-style retrieval over the Milvus collections built by
# ingest.py, with NVIDIA RAG v2.2.0's settings:
#   1. search   top 100 chunks by embedding similarity   (VECTOR_DB_TOPK=100)
#               embedder nvidia/llama-nemotron-embed-1b-v2, "query: " prefix
#   2. rerank   all 100 with nvidia/llama-nemotron-rerank-1b-v2,
#               input "question:{q} \n \n passage:{p}", score = logit
#   3. keep     the best 10                              (APP_RETRIEVER_TOPK=10)
#
# Later steps import retrieve(question, collection).
# Run directly for a test: one question per collection, top 10 before and
# after reranking.
# Usage (inside .venv-rag, from ~/llm-safety/aiq_local; not while ingest.py runs):
#   python rag/retrieve.py
# =====================================================================
import os, time
import torch
import torch.nn.functional as F
from pymilvus import MilvusClient
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.getenv("MILVUS_DB", os.path.join(HERE, "milvus.db"))   # Dhabi: one copy per lane
EMBED = "nvidia/llama-nemotron-embed-1b-v2"
RERANK = "nvidia/llama-nemotron-rerank-1b-v2"
VECTOR_DB_TOPK, RETRIEVER_TOPK, RERANK_BATCH = 100, 10, 16
# NVIDIA GPU (Dhabi) if there is one, else the Mac GPU (mps), else the CPU
DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else "cpu")

_m = {}                                       # models and database, loaded once on first use

def _load():
    if _m:
        return _m
    _m["etok"] = AutoTokenizer.from_pretrained(EMBED, trust_remote_code=True)
    _m["emodel"] = AutoModel.from_pretrained(EMBED, trust_remote_code=True).to(DEVICE).eval()
    rtok = AutoTokenizer.from_pretrained(RERANK, trust_remote_code=True, padding_side="left")
    if rtok.pad_token is None:
        rtok.pad_token = rtok.eos_token
    rmodel = AutoModelForSequenceClassification.from_pretrained(RERANK, trust_remote_code=True).to(DEVICE).eval()
    if rmodel.config.pad_token_id is None:
        rmodel.config.pad_token_id = rtok.eos_token_id
    _m["rtok"], _m["rmodel"] = rtok, rmodel
    _m["db"] = MilvusClient(DB)
    return _m

def _embed_query(question):
    m = _load()
    batch = m["etok"](["query: " + question], padding=True, truncation=True,
                      max_length=8192, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        hidden = m["emodel"](**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * mask).sum(1) / mask.sum(1), p=2, dim=1).cpu().tolist()

def _rerank_scores(question, texts):
    m, scores = _load(), []
    for b in range(0, len(texts), RERANK_BATCH):
        pairs = [f"question:{question} \n \n passage:{t}" for t in texts[b:b + RERANK_BATCH]]
        batch = m["rtok"](pairs, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(DEVICE)
        with torch.inference_mode():
            scores += m["rmodel"](**batch).logits.view(-1).float().cpu().tolist()
    return scores

_loaded = set()                               # collections loaded into memory in this run

def search(question, collection, k=VECTOR_DB_TOPK):
    """Step 1 only: the k nearest chunks by embedding similarity."""
    db = _load()["db"]
    if collection not in _loaded:             # a fresh process finds collections "released"
        db.load_collection(collection)
        _loaded.add(collection)
    hits = db.search(collection, data=_embed_query(question), limit=k,
                                output_fields=["text", "source", "page"],
                                search_params={"metric_type": "IP"})[0]
    return [{"text": h["entity"]["text"], "source": h["entity"]["source"],
             "page": h["entity"]["page"], "vector_score": h["distance"]} for h in hits]

def retrieve(question, collection, top_k=VECTOR_DB_TOPK, keep=RETRIEVER_TOPK):
    """Steps 1-3: search top_k, rerank them all, return the best `keep`, best first."""
    hits = search(question, collection, top_k)
    for h, s in zip(hits, _rerank_scores(question, [h["text"] for h in hits])):
        h["rerank_score"] = s
    return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)[:keep]

if __name__ == "__main__":
    tests = [("Biomedical_Dataset", "What therapies restore CFTR protein function?"),
             ("Financial_Dataset", "How did Amazon's net income change in 2023?")]
    have = set(_load()["db"].list_collections())
    for collection, question in tests:
        if collection not in have:
            print(f"\n== {collection}: not in milvus.db yet, skipped")
            continue
        t0 = time.time()
        before = search(question, collection)
        after = retrieve(question, collection)
        print(f"\n== {collection}: {question}   ({time.time() - t0:.1f}s)")
        print("  BEFORE reranking (top 10 of the 100 by vector similarity):")
        for h in before[:RETRIEVER_TOPK]:
            print(f"    {h['vector_score']:.3f}  {h['source'][:38]:<38} p.{h['page']:<3} {' '.join(h['text'].split())[:80]}")
        print("  AFTER reranking (the 10 kept, best first):")
        for h in after:
            rank = next(i for i, b in enumerate(before, 1) if b is h or (b['text'] == h['text']))
            print(f"    {h['rerank_score']:7.2f}  was #{rank:<3} {h['source'][:38]:<38} p.{h['page']:<3} {' '.join(h['text'].split())[:70]}")
