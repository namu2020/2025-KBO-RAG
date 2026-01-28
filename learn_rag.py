from langchain_community.document_loaders import PyPDFLoader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter # type: ignore
from langchain_community.vectorstores import FAISS # type: ignore
from langchain_community.embeddings import HuggingFaceEmbeddings # type: ignore
from langchain_community.chat_models import ChatOllama # type: ignore
from langchain_core.prompts import ChatPromptTemplate # type: ignore
from langchain.chains.combine_documents import create_stuff_documents_chain # type: ignore
from langchain.chains import create_retrieval_chain # type: ignore

# 1. 문서 로드 (Load)
# 답변의 근거가 될 PDF 문서를 로드합니다.
loader = PyPDFLoader("./7_TransferLearning.pdf") 
docs = loader.load()

# 2. 문서 분할 (Split)
# 문서를 검색하기 좋은 크기의 여러 조각(청크)으로 나눕니다.
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# 3. 임베딩 & 벡터 스토어 생성 (Store)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2", # Huggingface 모델명 - 한국어 잘 하는 것, domain 특화 등등 모델 선택 가능
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)

# text로 검색 vs. embedding으로 검색
# text 검색: 사용자가 입력한 질문을 그대로 사용하여 검색
# embedding 검색: 질문을 임베딩하여 유사한 문서를 검색 => 보통 벡터로 저장할 때 RAG라고 부름

# 분할된 문서를 임베딩하여 FAISS 벡터 스토어에 저장합니다. (메모리 기반)
vectorstore = FAISS.from_documents(split_docs, embeddings)

# 4. 검색기(Retriever) 생성
# 벡터 스토어에서 관련 문서를 검색하는 역할을 합니다.
retriever = vectorstore.as_retriever()

# 5. LLM 로드 (Ollama 연동)
# 로컬에서 실행 중인 Ollama의 모델을 LangChain에서 사용할 수 있도록 설정합니다.
llm = ChatOllama(model="gemma3:270m") 
# 여기서 Open AI API 키 가져올 수도 있음 (이번에는 로컬에서 실행 중인 Ollama 사용)

# 6. 프롬프트 정의
# LLM에게 질문과 함께 검색된 문서를 어떻게 활용할지 지시하는 템플릿입니다.
prompt = ChatPromptTemplate.from_template("""
Answer the following question based only on the provided context.

<context>
{context}
</context>

Question: {input}
""")
# context에 RAG로 검색된 문서들이 들어감

# 7. RAG 체인 생성
# LangChain Expression Language (LCEL)을 사용하여 체인을 구성합니다.
document_chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, document_chain)

# 8. 체인 실행 및 질문
response = retrieval_chain.invoke({"input": "Adapter BERT의 저자가 누구야?"})

# 답변 출력
# 답변의 근거가 된 문서(Context) 출력
print("---------- 검색된 근거 문서 ----------\n")
for i, doc in enumerate(response["context"]):
    print(f"문서 #{i+1}:\n")
    print(doc.page_content)
    print("\n------------------------------------\n")

# 최종 답변 출력
print("---------- AI 최종 답변 ----------\n")
print(response["answer"])