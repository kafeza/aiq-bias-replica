# Search index (Milvus Lite), identical to the Mac's

`rag/milvus.db` is the search index of NVIDIA's two document collections (Biomedical_Dataset,
Financial_Dataset), built on the Mac with `rag/ingest.py` (16 Oct 2026). It is stored here so that
every machine gets exactly the same index, and therefore exactly the same search results.

It is a compressed archive split into two parts (GitHub refuses files over 100 MB).
`dhabi/setup_envs.sbatch` joins, checks and unpacks it automatically. By hand:

```bash
cat index/milvus.db.tar.gz.part-* | tee >(sha256sum) | tar -xzf - -C rag
```

| Check | SHA-256 |
|---|---|
| the joined archive | `383c7f47dbcdcb8b710d8be22e154e9617830d4000b0846593eb7a980066b8f3` |
| the unpacked index: `(cd rag/milvus.db && find . -type f ! -name LOCK \| LC_ALL=C sort \| xargs sha256sum) \| sha256sum` | `f6cffa8c70e09f9de736f6dc2557de2980a08de731414322f0aaa8eabea6e4d6` |

Contains text passages from NVIDIA's documents (NVIDIA dataset licence): keep this repository private.
