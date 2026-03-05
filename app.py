import os
import gradio as gr
from rag_pipeline import load_and_split_documents, create_vector_store
from model_config import get_qa_chain

# Global variable for the QA chain
qa_chain = None

def initialize_app():
    global qa_chain
    pdf_path = "the_nestle_hr_policy_pdf_2012.pdf"

    if os.path.exists(pdf_path):
        print(f"Loading and processing {pdf_path}...")
        chunks = load_and_split_documents(pdf_path)
        vectorstore = create_vector_store(chunks)

        # Initialize retriever
        retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

        # Initialize the RAG chain
        print("Initializing LLM and QA Chain (this may take a few minutes)...Value")
        qa_chain = get_qa_chain(retriever)
        print("Initialization complete.")
        return True
    else:
        print(f"Error: {pdf_path} not found.")
        return False

def predict(query):
    if qa_chain is None:
        return "Error: QA Chain not initialized. Please check if the PDF exists."

    response = qa_chain.invoke(query)
    return response['result']

if __name__ == "__main__":
    success = initialize_app()

    if success:
        # Define Gradio Interface
        demo = gr.Interface(
            fn=predict,
            inputs=gr.Textbox(lines=3, placeholder="Ask a question about the Nestlé HR Policy..."),
            outputs="text",
            title="Nestlé HR Policy RAG Assistant",
            description="An AI-powered assistant to answer questions about the 2012 Nestlé Human Resources Policy.",
            theme="default"
        )

        # Launch the app
        demo.launch(share=True)
    else:
        print("Application failed to start due to missing PDF file.")
