# company_backend.py — throttling-safe, Bedrock + LangChain

# Prefer modern packages; fall back to old paths if needed
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_community.vectorstores import FAISS
    from langchain_aws import BedrockEmbeddings, BedrockLLM
    from langchain.chains import RetrievalQA
except ImportError:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.vectorstores import FAISS
    from langchain.embeddings import BedrockEmbeddings
    from langchain.llms.bedrock import Bedrock as BedrockLLM
    from langchain.chains import RetrievalQA

import os
import boto3
from botocore.config import Config

# -------- Config --------
PDF_URL = "https://www.upl-ltd.com/images/people/downloads/Leave-Policy-India.pdf"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"
CLAUDE_MODEL_ID    = "anthropic.claude-v2:1"
REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"

# Larger chunks = far fewer Bedrock calls → fewer throttles
CHUNK_SIZE = 800
CHUNK_OVERLAP = 80
MAX_CHUNKS = 200   # hard cap to avoid hammering the API
TOP_K = 5          # docs retrieved for the answer

def _bedrock_client():
    # Retry & timeouts help smooth temporary throughput limits
    cfg = Config(
        region_name=REGION,
        retries={"max_attempts": 10, "mode": "standard"},
        read_timeout=60,
        connect_timeout=60,
    )
    return boto3.client("bedrock-runtime", config=cfg)

def company_pdf():
    """Load the PDF, split to fewer/larger chunks, dedupe + cap, embed with Titan, return FAISS store."""
    loader = PyPDFLoader(PDF_URL)
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    # Extract text, dedupe, and cap
    seen, texts = set(), []
    for d in chunks:
        t = (d.page_content or "").strip()
        if not t:
            continue
        if t in seen:
            continue
        seen.add(t)
        texts.append(t)
        if len(texts) >= MAX_CHUNKS:
            break

    # Bedrock embeddings with retry-enabled client; creds come from env (Streamlit Secrets)
    client = _bedrock_client()
    embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID, client=client)

    # Build FAISS vector store directly (so we control what we embed)
    vector = FAISS.from_texts(texts, embeddings)
    return vector  # note: this is a VectorStore, not IndexCreator output

def company_llm():
    """Claude v2:1 via the *completion* interface (stable with v2)."""
    return BedrockLLM(
        model_id=CLAUDE_MODEL_ID,
        model_kwargs={
            "max_tokens_to_sample": 1200,
            "temperature": 0.1,
            "top_p": 0.8,
            "stop_sequences": ["\n\nHuman:"],
        },
    )

def company_rag_response(index, question: str) -> str:
    """
    Build a RetrievalQA chain over the FAISS store and ask the question.
    `index` is the FAISS VectorStore returned by company_pdf().
    """
    llm = company_llm()
    retriever = index.as_retriever(search_kwargs={"k": TOP_K})
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)
    # run() is simple and fine here
    return qa.run(question)
