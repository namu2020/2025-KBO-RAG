# build_index_fast.py
import os, json, hashlib, numpy as np
from typing import List, Dict, Any, Tuple

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# before
# from langchain_community.embeddings import HuggingFaceEmbeddings
# after
from langchain_huggingface import HuggingFaceEmbeddings

try:
    from langchain.schema import Document
except Exception:
    from langchain_core.documents import Document

IN_JSONL   = "./artifacts/raw_docs.jsonl"
INDEX_DIR  = "./artifacts/faiss_index"
CACHE_NPZ  = "./artifacts/emb_cache.npz"  # md5 -> vector 캐시
EMB_MODEL  = "intfloat/multilingual-e5-base" # "paraphrase-multilingual-MiniLM-L12-v2"  # "intfloat/multilingual-e5-base"  # 또는 "BAAI/bge-m3"

CHUNK_SIZE = 555
CHUNK_OVER = 55
BATCH_SIZE = 32  # 크게! (MPS/CPU에 맞춰 조절)
NORMALIZE  = True # 코사인 검색 안정화

def autodetect_device() -> str:
    # torch가 깔려있고 MPS 가능하면 'mps', 아니면 'cpu'
    try:
        import torch
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
        # CUDA 있으면 'cuda' (데스크톱 환경 대비)
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows

def to_documents(rows: List[Dict[str, Any]]) -> List[Document]:
    return [Document(page_content=r["page_content"], metadata=r["metadata"]) for r in rows]

def chunk_docs(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVER)
    return splitter.split_documents(docs)

def md5_text(s: str) -> str:
    return hashlib.md5(s.encode("utf-8")).hexdigest()

def load_cache(path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(path): return {}
    data = np.load(path, allow_pickle=True)
    keys, vecs = data["keys"].tolist(), data["vecs"]
    return {k: vecs[i] for i, k in enumerate(keys)}

def save_cache(cache: Dict[str, np.ndarray], path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    keys = np.array(list(cache.keys()), dtype=object)
    vecs = np.stack([cache[k] for k in cache.keys()], axis=0) if cache else np.zeros((0,0), dtype=np.float32)
    np.savez_compressed(path, keys=keys, vecs=vecs)

if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "true")
    # (선택) CPU일 때 스레드 늘리기
    try:
        import torch, multiprocessing
        torch.set_num_threads(max(1, multiprocessing.cpu_count() - 1))
    except Exception:
        pass

    device = autodetect_device()
    print(f"⚙️ device = {device}")

    # 1) 로드 & 청크
    raw_rows = load_jsonl(IN_JSONL)
    docs     = to_documents(raw_rows)
    chunks   = chunk_docs(docs)
    print(f"📄 chunks: {len(chunks)}")

    # 2) 캐시 준비 (chunk_md5 = md5(page_content) + 모델명)
    cache = load_cache(CACHE_NPZ)
    need_texts, need_idx = [], []
    ids = []
    for i, d in enumerate(chunks):
        h = md5_text(d.page_content + "|" + EMB_MODEL)
        ids.append(h)
        if h not in cache:
            need_texts.append(d.page_content)
            need_idx.append(i)

    # 3) 임베딩기 (MPS/CPU 자동, 대배치)
    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": device},
        encode_kwargs={"normalize_embeddings": NORMALIZE, "batch_size": BATCH_SIZE, "convert_to_numpy": True}
    )

    # 4) 미보유 분만 배치 인코딩
    if need_texts:
        print(f"🧠 encode {len(need_texts)} new chunks (batch={BATCH_SIZE}) ...")
        new_vecs = embeddings.embed_documents(need_texts)  # List[List[float]] (numpy 변환 옵션 적용됨)
        # embed_documents가 numpy를 안 주는 버전이면 np.array로 감싸기
        new_vecs = np.array(new_vecs, dtype=np.float32)
        for idx, vec in zip(need_idx, new_vecs):
            cache[ids[idx]] = vec
        save_cache(cache, CACHE_NPZ)
        print("💾 cache updated.")
    else:
        print("✅ all chunk vectors are cached.")

    # 5) 전체 벡터/메타데이터 조립
    all_vecs = np.stack([cache[h] for h in ids], axis=0).astype(np.float32)
    metadatas = [c.metadata for c in chunks]
    texts     = [c.page_content for c in chunks]

   # --- 6) 벡터스토어 생성 & 저장 (재임베딩 없이 직접 FAISS 구성) ---
    import faiss
    from langchain_community.docstore.in_memory import InMemoryDocstore
    from langchain_community.vectorstores.faiss import FAISS as LCFAISS

    dim = all_vecs.shape[1]

    # 코사인 검색: 정규화했다면 IP 사용
    index = faiss.IndexFlatIP(dim) if NORMALIZE else faiss.IndexFlatL2(dim)
    index.add(all_vecs)

    # 🔴 꼭 문자열 id로 통일!
    doc_ids = [str(i) for i in range(len(texts))]

    docstore = InMemoryDocstore({
        doc_ids[i]: Document(page_content=texts[i], metadata=metadatas[i])
        for i in range(len(texts))
    })

    vs = LCFAISS(
        embedding_function=None,   # 재임베딩 안 함
        index=index,
        docstore=docstore,
        index_to_docstore_id={i: doc_ids[i] for i in range(len(doc_ids))}
    )

    os.makedirs(INDEX_DIR, exist_ok=True)
    vs.save_local(INDEX_DIR)
    print(f"✅ FAISS saved to {INDEX_DIR} with {len(texts)} chunks (dim={dim})")
