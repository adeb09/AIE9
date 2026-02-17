from langchain_community.document_loaders import TextLoader
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from ragas import EvaluationDataset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, NoiseSensitivity
from ragas import evaluate

import re
from typing import List
import time
import copy


def preprocess_text(text: str) -> List[str]:
    """borrowed from Ask Brave search for common preprocessing steps before TF-IDF"""
    # Lowercase
    text = text.lower()
    # Remove punctuation and digits
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords and short words
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    # Lemmatize
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    return tokens

# first let's load the documents for our RAG application
loader = TextLoader('data/HealthWellnessGuide.txt')
docs = loader.load()

# get documents with same chunking strategy as task 6
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=30)
split_docs = text_splitter.split_documents(docs)

# define embedding model for vector store
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# build in-memory vector store
client = QdrantClient(":memory:")
client.create_collection(
    collection_name="rag_eval_act1",
    vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
)

vector_store = QdrantVectorStore(
    client=client,
    collection_name="rag_eval_act1",
    embedding=embeddings,
)

# add Health and Wellness Guide to vector store for dense retrieval
_ = vector_store.add_documents(documents=split_docs)
dense_retriever = vector_store.as_retriever(search_kwargs={"k": 10})

# utilize BM25 Plus Retriever (Sparse Retriever)
bm25_retriever = BM25Retriever.from_documents(
    documents=split_docs,
    k=10,
    bm25_variant="plus",
    bm25_params={"delta": 0.5},
    preprocess_func=preprocess_text,
)

# create an ensemble retriever that utilizes Hybrid Approach (both Sparsen Retrieval based on BM25Plus and Dense Embedding Retrieval)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.5,0.5]
)

# define the RAG Pipeline in LangChain
RAG_PROMPT = """\
You are a helpful assistant who answers questions based on provided context. You must only use the provided context, and cannot use your own knowledge.

### Question
{question}

### Context
{context}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
llm = ChatOpenAI(model="gpt-4.1-nano")

def generate(state):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = rag_prompt.format_messages(question=state["question"], context=docs_content)
    response = llm.invoke(messages)
    return {"response" : response.content}

class State(TypedDict):
    question: str
    context: List[Document]
    response: str

def retrieve_hybrid(state):
    compressor = CohereRerank(model="rerank-v3.5")
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=ensemble_retriever, search_kwargs={"k": 5}
    )
    retrieved_docs = compression_retriever.invoke(state["question"])
    return {"context" : retrieved_docs}

# build graph
act_graph_builder = StateGraph(State).add_sequence([retrieve_hybrid, generate])
act_graph_builder.add_edge(START, "retrieve_hybrid")
act_graph = act_graph_builder.compile()

response = act_graph.invoke({"question" : "How can I improve my sleep quality?"})
print(response["response"])

# now evaluate on the same dataset and see if evaluation metrics improve or not
# dataset was defined in previously in the notebook already
act1_dataset = copy.deepcopy(dataset)

for test_row in act1_dataset:
    response = act_graph.invoke({"question" : test_row.eval_sample.user_input})
    test_row.eval_sample.response = response["response"]
    test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]
    time.sleep(10) # To try to avoid rate limiting.

act1_evaluation_dataset = EvaluationDataset.from_pandas(act1_dataset.to_pandas())

# evaluator_llm and custom_run_config were defined earlier in the notebook
act1_result = evaluate(
    dataset=act1_evaluation_dataset,
    metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
    llm=evaluator_llm,
    run_config=custom_run_config
)
act1_result