# =====================================================================
# Step 3a: check that NVIDIA's RAG retrieval models run on this Mac.
#   embedder : nvidia/llama-nemotron-embed-1b-v2   (was llama-3.2-nv-embedqa-1b-v2,
#              the default in NVIDIA RAG v2.2.0)
#   reranker : nvidia/llama-nemotron-rerank-1b-v2  (was llama-3.2-nv-rerankqa-1b-v2)
# Usage follows the two model cards. No PDFs, no database: two test
# questions, three test passages. Each question's best passage should be
# the one on the same topic, under both models.
# Usage:  python rag/check_models.py      (inside .venv-rag; first run downloads ~5 GB)
# =====================================================================
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

EMBED = "nvidia/llama-nemotron-embed-1b-v2"
RERANK = "nvidia/llama-nemotron-rerank-1b-v2"
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

questions = [
    "What new therapies restore CFTR protein function in cystic fibrosis?",
    "How did Amazon's revenue and net income change in 2023?",
]
passages = [
    "Elexacaftor/tezacaftor/ivacaftor, a triple CFTR modulator therapy, improves "
    "chloride transport and lung function in patients with at least one F508del allele.",
    "Amazon reported net sales of $574.8 billion for 2023, up 12%, and net income of "
    "$30.4 billion compared with a net loss of $2.7 billion in 2022.",
    "Milvus is an open-source vector database built for similarity search over embeddings.",
]
print(f"device: {DEVICE}\n")

# ---- embedder: "query: " / "passage: " prefixes, mean pooling, L2 normalisation
tok = AutoTokenizer.from_pretrained(EMBED, trust_remote_code=True)
model = AutoModel.from_pretrained(EMBED, trust_remote_code=True).to(DEVICE).eval()

def embed(texts, prefix):
    batch = tok([prefix + t for t in texts], padding=True, truncation=True,
                max_length=8192, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    pooled = (hidden * mask).sum(1) / mask.sum(1)
    return F.normalize(pooled, p=2, dim=1)

q = embed(questions, "query: ")
p = embed(passages, "passage: ")
sim = (q @ p.T).cpu()
print(f"EMBEDDER  vector size {q.shape[1]}  (NVIDIA default: 2048)")
for i, question in enumerate(questions):
    scores = " ".join(f"{s:.3f}" for s in sim[i].tolist())
    print(f"  Q{i + 1} similarity to P1 P2 P3: {scores}   best = P{int(sim[i].argmax()) + 1}")
del model

# ---- reranker: "question:{q} \n \n passage:{p}", score = logit (higher = more relevant)
rtok = AutoTokenizer.from_pretrained(RERANK, trust_remote_code=True, padding_side="left")
if rtok.pad_token is None:
    rtok.pad_token = rtok.eos_token
rmodel = AutoModelForSequenceClassification.from_pretrained(RERANK, trust_remote_code=True).to(DEVICE).eval()
if rmodel.config.pad_token_id is None:
    rmodel.config.pad_token_id = rtok.eos_token_id

print("\nRERANKER")
for i, question in enumerate(questions):
    texts = [f"question:{question} \n \n passage:{d}" for d in passages]
    batch = rtok(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        scores = rmodel(**batch).logits.view(-1).float().cpu()
    print(f"  Q{i + 1} scores for P1 P2 P3: " + " ".join(f"{s:6.2f}" for s in scores.tolist())
          + f"   best = P{int(scores.argmax()) + 1}")

print("\nExpected: Q1 best = P1, Q2 best = P2 under both models.")
