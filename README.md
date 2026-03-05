# Nestlé HR Policy RAG Assistant

An AI-powered Retrieval-Augmented Generation (RAG) assistant designed to answer questions about the 2012 Nestlé Human Resources Policy document.

## Technology Stack
- **Framework:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (all-MiniLM-L6-v2)
- **LLM:** Mistral-7B-Instruct-v0.3 (quantized via bitsandbytes)
- **Interface:** Gradio

## Repository Structure
- `app.py`: The main entry point of the application. Integrates the RAG pipeline with a Gradio web interface.
- `rag_pipeline.py`: Contains logic for document loading (PyMuPDF), text splitting (RecursiveCharacterTextSplitter), and vector store management.
- `model_config.py`: Handles the initialization of the Mistral-7B model with 4-bit quantization and sets up the RetrievalQA chain.
- `requirements.txt`: Lists all Python dependencies required to run the project.

## Installation
1. Clone the repository to your local machine.
2. Ensure you have Python 3.10+ installed.
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage
1. Place the PDF file `the_nestle_hr_policy_pdf_2012.pdf` in the root directory of the repository.
2. Run the application:
   ```bash
   python app.py
   ```
3. Once initialized, a Gradio link (local and/or public) will be provided. Open the link in your browser to interact with the assistant.

## Notes
- The model uses 4-bit quantization to run efficiently on consumer GPUs (requires approximately 6-8GB of VRAM).
- Ensure you have a Hugging Face token configured if accessing gated models.
