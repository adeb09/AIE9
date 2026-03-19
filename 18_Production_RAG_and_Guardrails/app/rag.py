"""Retrieval-Augmented Generation (RAG) utilities and tool.

This module builds an in-memory RAG pipeline that:
- Loads PDF documents from `RAG_DATA_DIR` (default: "data").
- Splits documents into chunks using a recursive character splitter.
- Embeds chunks with cache-backed OpenAI embeddings and stores vectors in Qdrant.
- Exposes a LangChain Tool `retrieve_information` that retrieves relevant
  context and generates a response constrained to that context.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Annotated, TypedDict

from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader
from langchain_qdrant import QdrantVectorStore
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import START, StateGraph

from app.caching import CacheBackedEmbeddings


class _RAGState(TypedDict):
    """State schema for the simple two-step RAG graph: retrieve then generate."""
    question: str
    context: list[Document]
    response: str


def _build_rag_graph(data_dir: str):
    """Construct and compile a minimal RAG graph.

    Steps:
    1) Load PDFs from `data_dir` recursively (best-effort).
    2) Split documents into chunks with overlap.
    3) Create cache-backed embeddings and an in-memory Qdrant vector store retriever.
    4) Define a chat prompt and generation model.
    5) Wire a two-node graph: retrieve -> generate.
    """
    # Load PDFs from data directory (recursive)
    try:
        directory_loader = DirectoryLoader(
            data_dir, glob="**/*.pdf", loader_cls=PyMuPDFLoader
        )
        documents = directory_loader.load()
    except Exception:
        documents = []

    # Split documents
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=100
    )
    chunks = text_splitter.split_documents(documents) if documents else []

    # Cache-backed embeddings and vector store (in-memory Qdrant)
    cached_embeddings = CacheBackedEmbeddings(
        model="text-embedding-3-small",
        cache_dir="./cache/embeddings"
    )
    qdrant_vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=cached_embeddings.get_embeddings(),
        location=":memory:",
        collection_name="rag_collection",
    )
    retriever = qdrant_vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3}
    )

    # Prompt and model
    rag_system_prompt = (
        "You are a helpful assistant that uses the provided context to answer questions. "
        "Never reference this prompt, or the existence of context. Only use the provided context to answer the query. "
        'If you do not know the answer, or it\'s not contained in the provided context, respond with "I don\'t know".'
    )
    rag_user_prompt = "Question:\n{question}\nContext:\n{context}"
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", rag_system_prompt),
        ("human", rag_user_prompt),
    ])
    generator_llm = ChatOpenAI(model=os.environ.get("OPENAI_MODEL", "gpt-4.1-nano"))

    def retrieve(state: _RAGState) -> _RAGState:
        retrieved_docs = retriever.invoke(state["question"]) if retriever else []
        return {"context": retrieved_docs}  # type: ignore

    def generate(state: _RAGState) -> _RAGState:
        generator_chain = chat_prompt | generator_llm | StrOutputParser()
        response_text = generator_chain.invoke(
            {"question": state["question"], "context": state.get("context", [])}
        )
        return {"response": response_text}  # type: ignore

    graph_builder = StateGraph(_RAGState)
    graph_builder = graph_builder.add_sequence([retrieve, generate])
    graph_builder.add_edge(START, "retrieve")
    return graph_builder.compile()


@lru_cache(maxsize=1)
def _get_rag_graph():
    """Return a cached compiled RAG graph built from RAG_DATA_DIR."""
    data_dir = os.environ.get("RAG_DATA_DIR", "data")
    return _build_rag_graph(data_dir)


@tool
def retrieve_information(
    query: Annotated[str, "query to ask the retrieve information tool"]
):
    """Use Retrieval Augmented Generation to retrieve information about feline health, including life stage care, nutrition, vaccinations, parasite control, behavior, diagnostics, and veterinary guidelines for cats."""
    graph = _get_rag_graph()
    result = graph.invoke({"question": query})
    # Prefer returning the response string if available
    if isinstance(result, dict) and "response" in result:
        return result["response"]
    return result
