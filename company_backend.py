# company_backend.py

# Prefer modern packages, fall back if older langchain is installed
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_aws import BedrockEmbeddings, BedrockLLM
    from langchain_community.vectorstores import FAISS
    from langchain.indexes import VectorstoreIndexCreator
except ImportError:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import BedrockEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.llms.bedrock import Bedrock as BedrockLLM  # completion-style

PDF_URL = "https://esdubai.com/wp-content/uploads/documents/es_employee_handbook.pdf"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"
CLAUDE_MODEL_ID = "anthropic.claude-v2:1"

def company_pdf():
    loader = PyPDFLoader(PDF_URL)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=100,
        chunk_overlap=10,
    )
    # No credentials_profile_name -> uses Streamlit Secrets env vars
    embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID)
    index_creator = VectorstoreIndexCreator(
        text_splitter=splitter,
        embedding=embeddings,
        vectorstore_cls=FAISS,
    )
    return index_creator.from_loaders([loader])

def company_llm():
    # Completion interface matches Claude v2 request shape; avoids ValidationException
    return BedrockLLM(
        model_id=CLAUDE_MODEL_ID,
        model_kwargs={
            "max_tokens_to_sample": 1200,
            "temperature": 0.1,
            "top_p": 0.8,
            "stop_sequences": ["\n\nHuman:"]
        },
    )

def company_rag_response(index, question: str) -> str:
    llm = company_llm()
    return index.query(question=question, llm=llm)
