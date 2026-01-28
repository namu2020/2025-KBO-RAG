from __future__ import annotations
import os, re, json, traceback
from pathlib import Path
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain.retrievers import EnsembleRetriever

# ----- 배경 이미지 헬퍼: 앱 맨 위에 추가 -----
import base64, os
from pathlib import Path
import streamlit as st

@st.cache_data
def _b64_img(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode()

def set_background(image_path: str | None = None,
                   image_url: str | None = None,
                   opacity: float = 0.18,   # 어둡게 오버레이
                   blur_px: int = 0):       # 사이드바 블러
    if image_path:
        ext = Path(image_path).suffix[1:] or "png"
        src = f"data:image/{ext};base64,{_b64_img(image_path)}"
    elif image_url:
        src = image_url
    else:
        return

    st.markdown(f"""
    <style>
    /* 전체 앱 배경 */
    .stApp {{
        background: linear-gradient(rgba(0,0,0,{opacity}), rgba(0,0,0,{opacity})),
                    url('{src}') no-repeat center center fixed;
        background-size: cover;
    }}
    /* 사이드바 살짝 반투명/블러 (선택) */
    section[data-testid="stSidebar"] > div:first-child {{
        background: rgba(0,0,0,0.25);
        backdrop-filter: blur({blur_px}px);
    }}
    </style>
    """, unsafe_allow_html=True)


# ====== 경로/모델 설정 ======
INDEX_DIR    = "./artifacts/faiss_index"
NAMES_PATH   = "./artifacts/player_names.json"   # build_docs.py가 생성
EMB_MODEL    = "intfloat/multilingual-e5-base"   # 또는 "BAAI/bge-m3", "paraphrase-multilingual-MiniLM-L12-v2"
OLLAMA_MODEL = "gemma3:4b"                       # 로컬에 존재하는 모델명으로

# ====== 유틸 ======
def needs_query_prefix(model_name: str) -> bool:
    name = model_name.lower()
    return ("e5" in name) or ("bge" in name)

QUERY_PREFIX = "query: " if needs_query_prefix(EMB_MODEL) else ""

STATS_KEYWORDS = r"(성적|기록|스탯|타율|출루율|장타율|ops|war|wrc\+|woba|era|fip|whip|이닝|세이브|홀드|통산|시즌|수치|정량|streak|평균|지표)"

@st.cache_data
def load_player_names() -> list[str]:
    # build_docs.py에서 만든 이름 목록 우선 사용
    if Path(NAMES_PATH).exists():
        with open(NAMES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # 백업: CSV에서 직접 로드(파일 경로 필요 시 수정)
    names = set()
    try:
        import pandas as pd
        for p in ["./2025csv/hitter_basic_all.csv", "./2025csv/pitcher_basic_all.csv"]:
            if Path(p).exists():
                df = pd.read_csv(p, encoding="utf-8", low_memory=False)
                if "선수명" in df.columns:
                    names.update(df["선수명"].dropna().astype(str).str.strip())
    except Exception:
        pass
    return sorted(names)

PLAYER_NAMES = load_player_names()

def normalize_ko(s: str) -> str:
    return re.sub(r"\s+", "", s)

def extract_name(query: str) -> str | None:
    q = normalize_ko(query)
    # 1) 완전/부분 포함 (가장 긴 이름 우선)
    cands = [n for n in PLAYER_NAMES if normalize_ko(n) in q]
    if cands:
        return max(cands, key=len)
    # 2) 퍼지 매칭(선택)
    try:
        from rapidfuzz import process, fuzz
        name, score, _ = process.extractOne(q, PLAYER_NAMES, scorer=fuzz.WRatio)
        if score >= 90:
            return name
    except Exception:
        pass
    return None

def is_stats_query(q: str) -> bool:
    return re.search(STATS_KEYWORDS, q, flags=re.IGNORECASE) is not None

# ====== 리소스 ======
# 1) 벡터스토어만 캐시 (retriever는 매 질문마다 생성)
@st.cache_resource
def get_vectorstore():
    embeddings = HuggingFaceEmbeddings(
        model_name=EMB_MODEL,
        model_kwargs={"device": "mps"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 256}
    )
    vs = FAISS.load_local(INDEX_DIR, embeddings, allow_dangerous_deserialization=True)
    return vs

def make_retrievers(vs, filter_dict: dict | None, stats_intent: bool):
    # 공통 MMR 설정
    mmr_kwargs = {"k": 8, "fetch_k": 40, "lambda_mult": 0.3}

    if stats_intent:
        # 정량 의도 -> stats만
        filt = {"type": "stats", **(filter_dict or {})}
        retriever = vs.as_retriever(search_type="mmr", search_kwargs={**mmr_kwargs, "filter": filt})
        return retriever

    # 정성(기본) -> 위키 가중 앙상블
    wiki_filt = {"type": "wiki", **(filter_dict or {})} if filter_dict else {"type": "wiki"}
    base_filt = filter_dict or {}

    wiki_ret = vs.as_retriever(search_type="mmr", search_kwargs={**mmr_kwargs, "filter": wiki_filt})
    base_ret = vs.as_retriever(search_type="mmr", search_kwargs={**mmr_kwargs, "filter": base_filt} if base_filt else mmr_kwargs)

    from langchain.retrievers import EnsembleRetriever
    return EnsembleRetriever(retrievers=[wiki_ret, base_ret], weights=[0.7, 0.3])

@st.cache_resource
def get_llm():
    llm = ChatOllama(model=OLLAMA_MODEL)
    _ = llm.invoke("ping")  # 헬스체크
    return llm

@st.cache_resource
def get_prompt():
    return ChatPromptTemplate.from_template(
        "주어진 컨텍스트만 사용해 한국어로 정확히 답하라.\n"
        "예를 들어, SSG 최정의 별명을 물으면, 정답은 마그넷정, 소년장사야. 이처럼 주어진 컨텍스트를 활용해라.\n"
        "특히 선수의 이름은 중요한 정보이니 컨텍스트에서 주어지면 꼭 활용해라.\n"
        "모르면 모른다고 말해라.\n\n<context>\n{context}\n</context>\n\n질문: {input}"
    )

# ====== 앱 ======
st.title("⚾️ 2025 KBO RAG")
st.markdown("사전 구축한 인덱스를 이용해 2025 시즌 선수 질문에 답합니다.")

st.set_page_config(page_title="KBO RAG", layout="wide")
set_background(image_path="logo2.jpg", opacity=0.22, blur_px=6)   # 또는 image_url="https://..."

if not os.path.exists(INDEX_DIR):
    st.error(f"인덱스 폴더가 없습니다: {INDEX_DIR}")
    st.stop()

# 초기화
try:
    vs = get_vectorstore()
    st.write("✅ index loaded")
    # 빠른 헬스체크(필터 없이)
    _probe = vs.as_retriever(search_type="mmr", search_kwargs={"k": 5}).get_relevant_documents(QUERY_PREFIX + "헬스체크")
    st.write(f"🔎 retriever test docs = {len(_probe)}")
    llm = get_llm()
    st.write(f"✅ chain OK")
except Exception:
    st.error("초기화 오류")
    st.code(traceback.format_exc())
    st.stop()

question = st.text_input("질문을 입력하세요:", placeholder="예) SSG 최정 선수의 별명은?")
if question:
    with st.spinner("답변을 생성하는 중입니다..."):
        name = extract_name(question)
        stats_intent = is_stats_query(question)

        route = "stats-only" if stats_intent else "wiki-biased"
        st.caption(f"라우팅: **{route}** | 추출 선수: **{name or '없음'}**")

        filter_dict = {"player": name} if name else None

        # ✅ 매 질문마다 fresh retriever 생성 (필터 누적 방지)
        retriever = make_retrievers(vs, filter_dict, stats_intent)

        prompt = get_prompt()
        doc_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, doc_chain)

        try:
            resp = rag_chain.invoke({"input": QUERY_PREFIX + question})
            st.subheader("🤖 AI 답변")
            st.write(resp.get("answer", ""))

            with st.expander("RAG Context 확인하기"):
                for i, d in enumerate(resp.get("context", []), 1):
                    st.markdown(f"**문서 #{i}**")
                    if d.metadata:
                        st.code(d.metadata, language="json")
                    preview = d.page_content if len(d.page_content) < 1200 else d.page_content[:1200] + "…"
                    st.write(preview)
                    st.markdown("---")
        except Exception:
            st.error("질의 실행 중 오류가 발생했습니다.")
            st.code(traceback.format_exc())

