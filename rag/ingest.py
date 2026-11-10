# =====================================================================
# Step 3b: ingest NVIDIA's AI-Q document collections into Milvus Lite,
# copying NVIDIA RAG v2.2.0's settings where possible.
#
#   PDFs      data/Biomedical_Dataset (43), data/Financial_Dataset (92)
#   reading   PyMuPDF page text. nv-ingest also extracts charts and runs OCR
#             on images; that is not reproduced here (a noted difference).
#   chunks    512 tokens, 150 overlap (APP_NVINGEST_CHUNKSIZE/CHUNKOVERLAP),
#             counted with the embedding model's own tokenizer
#   embedding nvidia/llama-nemotron-embed-1b-v2, "passage: " prefix,
#             mean pooling + L2 normalisation (as in check_models.py)
#   database  Milvus Lite file rag/milvus.db, one collection per dataset,
#             inner-product search (= cosine on normalised vectors).
#             FLAT index = exact search over every chunk. NVIDIA uses an
#             approximate GPU index; exact finds the true top 100. AUTOINDEX
#             is avoided: Milvus Lite then builds a faiss HNSW index, and
#             faiss's OpenMP runtime clashes with PyTorch's in one process on
#             macOS (OMP Error #15 / segfault once a collection grows large).
#
# A run rebuilds the collections it is given from scratch, unless --resume:
# then PDFs already stored (listed in rag/ingested_<collection>.txt) are
# skipped, so a crash or sleep doesn't mean starting over.
# Usage (inside .venv-rag, from ~/llm-safety/aiq_local):
#   caffeinate -i python rag/ingest.py Biomedical_Dataset            # one collection
#   caffeinate -i python rag/ingest.py Financial_Dataset --resume    # continue after a stop
#   python rag/ingest.py Biomedical_Dataset --limit 2                # quick test: 2 PDFs
# =====================================================================
import argparse, os, sys, time
import pymupdf
import torch
import torch.nn.functional as F
from pymilvus import DataType, MilvusClient
from transformers import AutoModel, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "data")
DB = os.getenv("MILVUS_DB", os.path.join(HERE, "milvus.db"))   # Dhabi: one copy per lane
EMBED = "nvidia/llama-nemotron-embed-1b-v2"
CHUNK, OVERLAP = 512, 150
DIM, BATCH = 2048, 16
# NVIDIA GPU (Dhabi) if there is one, else the Mac GPU (mps), else the CPU
DEVICE = ("cuda" if torch.cuda.is_available() else
          "mps" if torch.backends.mps.is_available() else "cpu")

ap = argparse.ArgumentParser()
ap.add_argument("collections", nargs="+", choices=["Biomedical_Dataset", "Financial_Dataset"])
ap.add_argument("--limit", type=int, help="only the first N PDFs (for a quick test)")
ap.add_argument("--resume", action="store_true", help="keep the collection and skip PDFs already stored")
args = ap.parse_args()

tok = AutoTokenizer.from_pretrained(EMBED, trust_remote_code=True)
model = AutoModel.from_pretrained(EMBED, trust_remote_code=True).to(DEVICE).eval()

def embed(texts, prefix):
    batch = tok([prefix + t for t in texts], padding=True, truncation=True,
                max_length=CHUNK + 16, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        hidden = model(**batch).last_hidden_state
    mask = batch["attention_mask"].unsqueeze(-1).to(hidden.dtype)
    return F.normalize((hidden * mask).sum(1) / mask.sum(1), p=2, dim=1).cpu().tolist()

def chunk_pdf(path):
    """Split a PDF's text into 512-token windows that overlap by 150 tokens.
    The text runs across page breaks; each chunk records the page it starts on."""
    ids, starts = [], []                      # token ids; (token offset, page number)
    with pymupdf.open(path) as doc:
        for pno, page in enumerate(doc, 1):
            text = page.get_text("text").strip()
            if text:
                starts.append((len(ids), pno))
                ids += tok(text + "\n", add_special_tokens=False)["input_ids"]
    chunks, step = [], CHUNK - OVERLAP
    for s in range(0, max(len(ids) - OVERLAP, 1), step):
        window = ids[s:s + CHUNK]
        if not window:
            break
        page = max((p for off, p in starts if off <= s), default=1)
        chunks.append((tok.decode(window).strip(), page))
    return [(t, p) for t, p in chunks if t], len(starts)

client = MilvusClient(DB)

def create_collection(name):
    schema = client.create_schema(auto_id=True)
    schema.add_field("id", DataType.INT64, is_primary=True)
    schema.add_field("vector", DataType.FLOAT_VECTOR, dim=DIM)
    schema.add_field("text", DataType.VARCHAR, max_length=65535)
    schema.add_field("source", DataType.VARCHAR, max_length=1024)
    schema.add_field("page", DataType.INT64)
    index = client.prepare_index_params()
    index.add_index(field_name="vector", index_type="FLAT", metric_type="IP")
    client.create_collection(name, schema=schema, index_params=index)

for name in args.collections:
    folder = os.path.join(DATA, name)
    pdfs = sorted(os.path.join(r, f) for r, _, fs in os.walk(folder) for f in fs if f.lower().endswith(".pdf"))
    if args.limit:
        pdfs = pdfs[:args.limit]
    if not pdfs:
        sys.exit(f"No PDFs found in {folder}")

    done_file = os.path.join(HERE, f"ingested_{name}.txt")
    resuming = args.resume and client.has_collection(name) and os.path.exists(done_file)
    done = set(open(done_file).read().split("\n")) - {""} if resuming else set()
    if not resuming:
        if client.has_collection(name):
            client.drop_collection(name)
        open(done_file, "w").close()
        create_collection(name)

    print(f"\n== {name}: {len(pdfs)} PDFs -> {DB}" + (f"  (resuming: {len(done)} already stored)" if resuming else ""))
    total, t_all = 0, time.time()
    for i, path in enumerate(pdfs, 1):
        if os.path.basename(path) in done:
            continue
        t0 = time.time()
        try:
            chunks, pages = chunk_pdf(path)
        except Exception as e:                # a damaged PDF should not stop the run
            print(f"  [{i:3d}/{len(pdfs)}] SKIPPED {os.path.basename(path)}: {e}")
            continue
        rows = []
        for b in range(0, len(chunks), BATCH):
            part = chunks[b:b + BATCH]
            for (text, page), vec in zip(part, embed([t for t, _ in part], "passage: ")):
                rows.append({"vector": vec, "text": text[:60000],
                             "source": os.path.basename(path), "page": page})
        if rows:
            client.insert(name, rows)
        with open(done_file, "a") as f:     # mark this PDF as stored, for --resume
            f.write(os.path.basename(path) + "\n")
        total += len(rows)
        print(f"  [{i:3d}/{len(pdfs)}] {os.path.basename(path)[:60]:<60} "
              f"{pages:4d} pages {len(rows):5d} chunks  {time.time() - t0:5.1f}s")
    print(f"  done: {total} chunks in {(time.time() - t_all) / 60:.1f} min")

    # ---- quick test search: the top 3 chunks for a topic question
    question = {"Biomedical_Dataset": "What therapies restore CFTR protein function?",
                "Financial_Dataset": "How did Amazon's net income change in 2023?"}[name]
    hits = client.search(name, data=embed([question], "query: "), limit=3,
                         output_fields=["source", "page", "text"], search_params={"metric_type": "IP"})[0]
    print(f"  test search: {question}")
    for h in hits:
        e = h["entity"]
        print(f"    {h['distance']:.3f}  {e['source'][:50]} p.{e['page']}: {' '.join(e['text'].split())[:110]}")

print(f"\nCollections in {os.path.basename(DB)}: " +
      ", ".join(f"{c} ({client.get_collection_stats(c)['row_count']} chunks)" for c in client.list_collections()))
