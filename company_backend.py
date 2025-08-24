import os

try:
    # Modern package structure (langchain ≥ 0.2.x)
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_aws import BedrockEmbeddings, ChatBedrock
    from langchain_community.vectorstores import FAISS
    from langchain.indexes import VectorstoreIndexCreator
except ImportError:
    # Fallback for older versions
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import BedrockEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.llms.bedrock import Bedrock as ChatBedrock

# Configuration
PDF_URL = "https://esdubai.com/wp-content/uploads/documents/es_employee_handbook.pdf"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"
CHAT_MODEL_ID = "anthropic.claude-v2:1"

def company_pdf():
    """Load the PDF, split it into chunks, embed them with Titan, and return a FAISS index."""
    loader = PyPDFLoader(PDF_URL)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=100,
        chunk_overlap=10,
    )

    # If AWS_PROFILE is set, pass it to BedrockEmbeddings; otherwise rely on env vars.
    profile = os.getenv("AWS_PROFILE")
    embed_kwargs = {"model_id": EMBEDDING_MODEL_ID}
    if profile:
        embed_kwargs["credentials_profile_name"] = profile

    embeddings = BedrockEmbeddings(**embed_kwargs)

    index_creator = VectorstoreIndexCreator(
        text_splitter=splitter,
        embedding=embeddings,
        vectorstore_cls=FAISS,
    )
    return index_creator.from_loaders([loader])

def company_llm():
    """Return a Bedrock chat model (Claude v2:1) with preset sampling parameters."""
    profile = os.getenv("AWS_PROFILE")
    llm_kwargs = {
        "model_id": CHAT_MODEL_ID,
        "model_kwargs": {
            "max_tokens_to_sample": 5000,
            "temperature": 0.1,
            "top_p": 0.8,
        },
    }
    if profile:
        llm_kwargs["credentials_profile_name"] = profile
    return ChatBedrock(**llm_kwargs)

def company_rag_response(index, question: str) -> str:
    """Query the FAISS index with the given question and return the LLM’s answer."""
    llm = company_llm()
    return index.query(question=question, llm=llm)
