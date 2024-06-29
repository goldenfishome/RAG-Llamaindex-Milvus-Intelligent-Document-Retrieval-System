# RAG-Llamaindex-Milvus-Intelligent-Document-Retrieval-System
This repository builds a Retrieval-Augmented Generation (RAG) system using LlamaIndex and Milvus, which generate new text based on the retrieved documents.

**Dataset: arxiv-metadata-oai-snapshot.json**

A publicly available dataset downloaded from Kaggle, it includes ArXiv papers, news articles, or any large corpus of text documents.

`milvus_connection.ipynb`

Preprocessing: Clean and preprocess the dataset to remove any noise and irrelevant information.
insert processed data into milvus vector database.

**load_milvus.ipynb**

Build RAG, and use LLM to get a RAG response.

**app.py**

Streamlit web interface where users can input queries and view the retrieved results via RAG model.

***llamaindex_app Folder***

Build same application but with Llamaindex Framework

**data_process.py**

Process arvix data into csv form for analysis.

**llamaindex_inserting.ipynb**

Create an index across the data, and insert data into Milvus collection.

**llamaindex_loading.ipynb**

Build RAG with llamaindex framework, and formulate the response.

**app_llamaindex.py**

Streamlit web interface where users can input queries and view the retrieved results via RAG-llamaindex model.



