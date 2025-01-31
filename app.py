import os
import streamlit as st
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Get the API keys from the environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")  # Assuming you have this key as well

# Import necessary libraries from langchain
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Set LangChain API key for use in API calls
os.environ["LANGCHAIN_API_KEY"] = LANGCHAIN_API_KEY

# Streamlit app title
st.title("RAG based AI document Analyzer")

# File uploader for user to upload their document (supports .txt files)
uploaded_file = st.file_uploader("Upload a .txt document", type="txt")

if uploaded_file is not None:
    # Load the uploaded document using TextLoader
    with open("sample.txt", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = TextLoader("sample.txt")
    documents = loader.load()

    # Display the content of the first document in the uploaded file
    st.subheader("Document Content:")
    st.write(documents[0].page_content)

    # Define text splitter to split document into chunks of size 500 with overlap of 100 characters
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)

    # Splitting the documents into chunks
    chunks = text_splitter.split_documents(documents)
    st.write(f"Total chunks created: {len(chunks)}")

    # Initialize Ollama Embeddings for embedding model
    embedding_model = OllamaEmbeddings(model="nomic-embed-text")

    # Create a FAISS vector store from the document chunks
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # Save the FAISS index locally
    vector_store.save_local("faiss_index")

    # Load the FAISS index from the local storage (with dangerous deserialization allowed)
    vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

    # Initialize Groq's LLaMA model
    llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)

    # Create a retriever from the FAISS vector store
    retriever = vector_store.as_retriever()

    # Define the Q/A chain with the retriever and Groq's LLaMA model
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    # User input for query
    query = st.text_input("Ask a question about the document:", "What does the document talk about AI advancements?")

    if query:
        # Perform the Q/A query
        result = qa_chain.invoke({"query": query})

        # Display the answer from the Q/A chain
        st.subheader("Answer:")
        st.write(result["result"])

        # Show the sources from where the answer was retrieved (first 200 characters of each)
        st.subheader("Source Documents:")
        for doc in result["source_documents"]:
            st.write(f"Source: {doc.page_content[:200]}...")

else:
    st.info("Please upload a .txt document to begin the analysis.")
