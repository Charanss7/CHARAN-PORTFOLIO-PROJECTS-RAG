import os
try:
    from langchain_community.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain_aws import BedrockEmbeddings, ChatBedrock
    from langchain_community.vectorstores import FAISS
    from langchain.indexes import VectorstoreIndexCreator
except ImportError:
    from langchain.document_loaders import PyPDFLoader
    from langchain.text_splitter import RecursiveCharacterTextSplitter
    from langchain.embeddings import BedrockEmbeddings
    from langchain.vectorstores import FAISS
    from langchain.indexes import VectorstoreIndexCreator
    from langchain.llms.bedrock import Bedrock as ChatBedrock

# Configuration for your models and PDF
PDF_URL = "https://esdubai.com/wp-content/uploads/documents/es_employee_handbook.pdf"
EMBEDDING_MODEL_ID = "amazon.titan-embed-text-v1"
CHAT_MODEL_ID = "anthropic.claude-v2:1"

def company_pdf():
    """Load the policy PDF, split it into chunks, embed them, and build a FAISS index."""
    loader = PyPDFLoader(PDF_URL)
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", " ", ""],
        chunk_size=100,
        chunk_overlap=10,
    )
    # Do NOT pass credentials_profile_name; use env vars from Streamlit Secrets
    embeddings = BedrockEmbeddings(model_id=EMBEDDING_MODEL_ID)
    index_creator = VectorstoreIndexCreator(
        text_splitter=splitter,
        embedding=embeddings,
        vectorstore_cls=FAISS,
    )
    return index_creator.from_loaders([loader])

def company_llm():
    """Return a Claude v2:1 chat model with required Bedrock parameters."""
    return ChatBedrock(
        model_id=CHAT_MODEL_ID,
        model_kwargs={
            # Anthropic models on Bedrock require this field
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens_to_sample": 5000,
            "temperature": 0.1,
            "top_p": 0.8,
        },
    )

def company_rag_response(index, question: str) -> str:
    """Query the index and return the answer from the Claude model."""
    llm = company_llm()
    return index.query(question=question, llm=llm)
