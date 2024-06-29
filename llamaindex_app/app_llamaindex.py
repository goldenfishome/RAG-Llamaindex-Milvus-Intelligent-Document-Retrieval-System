# app with AI, separating from data cleaning; working on issue of milvus connection

import streamlit as st

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.milvus import MilvusVectorStore
import textwrap

vector_store = MilvusVectorStore(uri="http://localhost:19530", dim=1536, overwrite=False)

index = VectorStoreIndex.from_vector_store(vector_store)

# Streamlit app
st.title("Intelligent Document Retrieval and Summarization with RAG-Llamaindex")
query = st.text_input("Enter your query:")

if st.button("Submit"):
    
    query_engine = index.as_query_engine()
    response = query_engine.query(query)

    summary = textwrap.fill(str(response), 100)
    st.write("### Summary")
    st.write(summary)
