# app.py
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import OpenAIEmbeddings
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant as QdrantVectorStore
from openai import OpenAI
from dotenv import load_dotenv
import tempfile
import os

# Load environment variables
load_dotenv()

# Instantiate OpenAI client
client = OpenAI()

# Initialize embedding model
embedding_model = OpenAIEmbeddings(model="text-embedding-3-large")

# UI: Title
st.title("📄 Chat with your PDF (RAG using Qdrant + OpenAI)")

# Step 1: Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

# State to track whether PDF is ready for chatting
if "ready" not in st.session_state:
    st.session_state.ready = False

if uploaded_file:
    if st.button("📚 Prepare this PDF"):
        with st.spinner("Processing the PDF..."):
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name

            # Load and chunk PDF
            loader = PyPDFLoader(tmp_path)
            pages = loader.load_and_split()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(pages)

            # Connect to Qdrant
            qdrant = QdrantVectorStore.from_documents(
                documents=chunks,
                embedding=embedding_model,
                url="http://localhost:6333",
                collection_name="learning_vectors",
                force_recreate=True
            )

            st.session_state.vector_db = qdrant
            st.session_state.ready = True
            st.success("✅ PDF is ready for chatting!")

# Step 2: Chat Interface
if st.session_state.ready:
    user_query = st.chat_input("Ask something from the PDF")

    if user_query:
        with st.spinner("Thinking..."):
            vector_db = st.session_state.vector_db

            # Similarity Search
            results = vector_db.similarity_search(user_query)

            context = "\n\n".join([
                f"Page Content: {res.page_content}\nPage Number: {res.metadata.get('page', 'Unknown')}\nFile Location: {res.metadata.get('source', 'Uploaded File')}"
                for res in results
            ])

            # Create System Prompt
            system_prompt = f"""
            You are a helpful assistant answering questions based on the provided PDF content and metadata.

            Only answer based on the following context and guide the user to the correct page.

            Context:
            {context}
            """

            # OpenAI Chat Completion
            response = client.chat.completions.create(
                model="gpt-4.1",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_query}
                ]
            )

            answer = response.choices[0].message.content
            st.chat_message("assistant").markdown(answer)
