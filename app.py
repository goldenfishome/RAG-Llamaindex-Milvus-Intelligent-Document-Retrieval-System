# app with AI, separating from data cleaning; working on issue of milvus connection

import streamlit as st

import pandas as pd
import json
import numpy as np
from openai import OpenAI
from pymilvus import MilvusClient
import openai

openai_client = OpenAI()


milvus_client = MilvusClient(uri="http://localhost:19530", token="root:Milvus")

collection_name = "my_rag_collection"

def emb_text(text):
    return (
        openai_client.embeddings.create(input=text, model="text-embedding-3-small")
        .data[0]
        .embedding
    )

# Streamlit app
st.title("RAG Intelligent Document Retrieval and Summarization")
query = st.text_input("Enter your query:")

if st.button("Submit"):
    
    query_embedding = emb_text(query)

    search_res = milvus_client.search(
        collection_name=collection_name,
        data=[
            emb_text(query)
        ],  # Use the `emb_text` function to convert the question to an embedding vector
        limit=3,  # Return top 3 results
        search_params={"metric_type": "IP", "params": {}},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )

    retrieved_lines_with_distances = [
        (res["entity"]["text"], res["distance"]) for res in search_res[0]
    ]

    context = "\n".join(
        [line_with_distance[0] for line_with_distance in retrieved_lines_with_distances]
    )

    SYSTEM_PROMPT = """
    Human: You are an AI assistant. You are able to find answers to the questions from the contextual passage snippets provided.
    """
    USER_PROMPT = f"""
    Use the following pieces of information enclosed in <context> tags to provide an answer to the question enclosed in <question> tags.
    <context>
    {context}
    </context>
    <question>
    {query}
    </question>
    """

    response = openai_client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_PROMPT},
        ],
    )

    summary = response.choices[0].message.content
    st.write("### Summary")
    st.write(summary)
