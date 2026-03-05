import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

def load_and_split_documents(pdf_path):
    # Load the PDF
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()

    # Split into chunks
    text_split = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " "]
    )
    chunks = text_split.split_documents(documents)
    return chunks

def create_vector_store(chunks, persist_directory="chroma_db"):
    # Initialize embeddings
    embedding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Create and persist vector store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory=persist_directory
    )
    return vectorstore

if __name__ == "__main__":
    # Example usage for testing
    pdf_file = "the_nestle_hr_policy_pdf_2012.pdf"
    if os.path.exists(pdf_file):
        chunks = load_and_split_documents(pdf_file)
        vs = create_vector_store(chunks)
        print(f"Vector store created with {len(chunks)} chunks.")
    else:
        print(f"Warning: {pdf_file} not found.")
