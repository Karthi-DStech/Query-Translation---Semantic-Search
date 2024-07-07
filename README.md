# RAG-Query-Translation--Semantic-Search

This repository contains code for performing Query Translation techniques in RAG (Retrieval-Augmented Generation). 

Query translation techniques enhance semantic search by transforming user queries better to match their intent and context with relevant documents. These techniques collectively improve search accuracy by ensuring the query aligns more closely with the diverse ways information is presented in documents.

- **Multi Query RAG:** The intuition of the Multi Query Approach is to take the query (questions) and break it down into a few differently worded questions from different perspectives. By breaking down the query and rewriting it into different queries, it increases the likelihood of retrieving the document that the query actually wants. This Multi Query Approach may increase the reliability of retrieval.

- **Fusion RRF(Reciprocal Rank Fusion):** The RAG Fusion, takes the input query (prompt) and breaks it down into sub-questions. Now rephrase the sub-questions and each query (questions) will be processed for retrieving relevant information from the Vector Store. After retrieval, the RAG Fusion technique will rank the retrieved documents based on the Fused Scores using the **Reciprocal Rank Fusion (RRF)** Formula and build a consolidated list. In this way, the retrieved documents are filtered by rank, augmented into a prompt template and sent to an LLM including the rank for generating the response. 

- **Decomposition (Least to Most):** Decomposition is about taking the input prompt and decomposing it into a set of sub-problems and solving it sequentially. The key idea of decomposition (Least to Most) is to break down a complex problem into a series of simpler subproblems and then solve them in sequence. Solving each subproblem is facilitated by the answers to previously solved subproblems. These sub-questions are answered sequentially, from the simplest to the most complex, to build up the final answer.

**The code processes documents from a given URL, splits and indexes them, and then uses different retrieval methods to answer queries.**

## Coding Structure

`main.py`: This is the main script that orchestrates the execution of the entire process. It sets up the configuration with necessary API keys, initializes the DocumentProcessor to process the URL provided by the user, and calls the chosen retrieval method using RAGMethodCaller.

`configuration.py`: Contains the Config class, which is responsible for setting up environment variables required for Langchain and OpenAI services. This class ensures that the application has the necessary API keys configured.

`document_processor.py`: Contains the DocumentProcessor class, which handles the document processing workflow. This includes loading documents from a URL, splitting the text into manageable chunks, and indexing the documents to create a retriever for performing semantic searches.

`call_methods.py`: Defines the RAGMethodCaller class, which selects and executes the appropriate retrieval method (multi-query, fusion, or decomposition) based on the specified method name. This class acts as an interface to the different retrieval methods implemented in the project.

`models/`: This directory contains the implementation of different retrieval methods:

  - `multi_query.py`: Contains the MultiQueryRetrieval class, which implements a multi-query retrieval method. This method generates multiple versions of the query to improve retrieval     accuracy.

  - `fusion_rrf.py`: Contains the FusionRetrieval class, which implements a fusion-based retrieval method. This method combines results from multiple queries to enhance the search results.

  - `decomposition_ltm.py`: Contains the DecompositionRetrieval class, which implements a decomposition-based retrieval method. This method breaks down complex queries into simpler sub-queries to improve the retrieval process.


## Requirements

- API key from Open AI.
- API key from LangChain. 

## Configuration
Before running the code, you need to set up the configuration with your API keys.
