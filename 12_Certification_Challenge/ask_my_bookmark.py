import aiohttp
import asyncio
import nest_asyncio
import os
from dotenv import load_dotenv
from typing import List, Dict, Any, Tuple, Union
import sys
import base64
from stamina import retry, retry_context
import re
from functools import partial
from textacy import preprocessing
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from qdrant_client.http.models import Distance, VectorParams
import uuid
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langgraph.graph import START, StateGraph
from langchain_core.documents import Document
from typing_extensions import List, TypedDict

import pickle


def load_cached_git_repo_data(path: str):
    # Load (unpickle) from the file
    with open(path, "rb") as f:
        starred_repo_data = pickle.load(f)
    return starred_repo_data

def strip_markdown(text: str) -> str:
    """Remove common Markdown syntax; keep inner text and emojis."""
    # Headings: ## Title -> Title
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Inline code: `code` -> code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Bold/italic: **bold** or *italic* -> bold / italic
    text = re.sub(r"\*{1,2}([^*]+)\*{1,2}", r"\1", text)
    text = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", text)
    # Link markup: [text](url) -> text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    # Collapse many newlines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def make_normalize_text_pipeline(
    *,
    url_repl: str = "",
    unicode_form: str = "NFC",
):
    """
    Build a textacy pipeline: Markdown/HTML → plain text (emojis kept).

    Args:
        url_repl: Replacement for URLs (default ""). Use "_URL_" to keep a placeholder.
        unicode_form: Unicode normalization form ("NFC", "NFD", "NFKC", "NFKD"). Default "NFC".

    Returns:
        A callable that takes a string and returns preprocessed plain text.
    """
    return preprocessing.make_pipeline(
        strip_markdown,
        preprocessing.remove.html_tags,
        preprocessing.normalize.bullet_points,
        preprocessing.normalize.quotation_marks,
        partial(preprocessing.normalize.unicode, form=unicode_form),
        preprocessing.normalize.whitespace,
    )

NAMESPACE = uuid.NAMESPACE_URL

def repo_to_uuid(repo_name: str) -> str:
    return str(uuid.uuid5(NAMESPACE, repo_name))

def normalize_docs(docs: List[Dict[str, Any]]):
    content_str = ''
    for doc in docs:
        content_str += (doc['content'] + '\n\n')
    truncated = content_str[:MAX_CHARACTERS] if len(content_str) > MAX_CHARACTERS else content_str
    normalized_text = normalize_text_pipeline(truncated)
    return normalized_text


def organize_repo_data(repo_data: Dict[Any, Any]) -> Tuple[List[str], List[Dict[str, Union[str, int]]], List[str]]:
    ids = []
    metadata = []
    docs = []
    for repo in repo_data:
        id = repo_to_uuid(repo['repo'])
        description = repo['description']
        doc_source = repo['doc_source']
        stars = repo['stars']
        url = repo['url']
        language = repo['language']
        # append to lists for DB consumption
        ids.append(id)
        metadata.append({
            'id': id,
            'repo': repo['repo'],
            'description': description,
            'language': language,
            'doc_source': doc_source,
            'stars': stars,
            'url': url,
        })
        docs.append(normalize_docs(repo['docs']))
    return ids, metadata, docs



# Default pipeline instance
normalize_text_pipeline = make_normalize_text_pipeline()