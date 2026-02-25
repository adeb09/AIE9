from typing import List, TypedDict, Dict, Callable, Any
import pickle
import time
from copy import deepcopy
import logging


from langchain.retrievers import ContextualCompressionRetriever, MultiQueryRetriever, ParentDocumentRetriever, \
    EnsembleRetriever
from langchain_community.document_loaders import TextLoader
from langchain_community.retrievers import BM25Retriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from ragas import RunConfig, evaluate, EvaluationDataset
from ragas.llms import LangchainLLMWrapper, BaseRagasLLM
from ragas.embeddings import LangchainEmbeddingsWrapper, BaseRagasEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents.base import Document
from ragas.testset import TestsetGenerator
from ragas.dataset_schema import EvaluationResult
from dotenv import load_dotenv
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution, SingleHopSpecificQuerySynthesizer, MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.testset_schema import Testset
from ragas.metrics import LLMContextRecall, Faithfulness, FactualCorrectness, ResponseRelevancy, ContextEntityRecall, \
    NoiseSensitivity
from langgraph.graph import START, StateGraph
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from qdrant_client.http.models import Distance, VectorParams
from langchain_core.retrievers import RetrieverLike, BaseRetriever
from langchain_cohere import CohereRerank
from langchain_core.language_models import BaseLanguageModel
from langchain.storage import InMemoryStore
from langchain_experimental.text_splitter import SemanticChunker
import pandas as pd


logger = logging.getLogger(__name__)

# load environment variables
load_dotenv()

HEALTH_AND_WELLNESS_GUIDE_PATH = "data/HealthWellnessGuide.txt"
GOLDEN_DATA_SET_FILE = "golden_data_set.pkl"
TEST_SET_SIZE = 15
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 400
CHILD_CHUNK_OVERLAP = 50
SLEEP_SECONDS = 1 # to avoid openAI rate-limiting
RAG_PROMPT = """
You are a helpful and kind assistant. Use the context provided below to answer the question.

If you do not know the answer, or are unsure, say you don't know.

### Question
{question}

### Context
{context}
"""

RAG_LLM = ChatOpenAI(model="gpt-4.1-mini")
RAG_EMBEDDING_MODEL = OpenAIEmbeddings(model="text-embedding-3-small")

def create_knowledge_graph(documents: List[Document], llm: BaseRagasLLM, embedding_model: BaseRagasEmbeddings,
                           cache_file: str = 'kg.json') -> KnowledgeGraph:
    """
    This function creates a Knowledge Graph (kg) of the documents supplied to it (this kg is utilized to create test queries downstream)
    Args:
        documents: the list of documents to create a knowledge graph of
        llm: the LLM to utilize to create transformations to our kg
        embedding_model: the embedding model to use with the LLM
        cache_file: file_path/name for the cached KG file (can reload this instead of redoing computation each time on same list of documents)

    Returns: KnowledgeGraph Object
    """
    # create knowledge graph
    try:
        kg = KnowledgeGraph.load(cache_file)
    except FileNotFoundError:
        kg = KnowledgeGraph()

        # add each document as a node in the KG
        for doc in documents:
            kg.nodes.append(
                Node(type=NodeType.DOCUMENT,
                     properties={"page_content": doc.page_content, "document_metadata": doc.metadata}))

        # apply transformations to KG (summarization, extracting headlines and themes..)
        transforms = default_transforms(documents=documents, llm=llm, embedding_model=embedding_model)
        apply_transforms(kg, transforms)
        kg.save(cache_file)
    return kg


def create_golden_data_set(documents: List[Document], llm: str, embedding_model: str,
                           use_cache: bool = True) -> Testset:
    # create knowledge graph from corpus (utilized for generating queries via LLM generator)
    # kg = create_knowledge_graph(documents, llm, embedding_model, cache_file='kg.json')
    try:
        test_queries = pickle.load(open(GOLDEN_DATA_SET_FILE, 'rb'))
    except FileNotFoundError:
        llm_wrapper = LangchainLLMWrapper(ChatOpenAI(model=llm))
        embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model=embedding_model))
        generator = TestsetGenerator(llm=llm_wrapper, embedding_model=embeddings)
        test_queries = generator.generate_with_langchain_docs(documents, testset_size=TEST_SET_SIZE)
        with open(GOLDEN_DATA_SET_FILE, 'wb') as f:
            pickle.dump(test_queries, f)


    # distribution of queries we want
    # single_specific, multi_abstract, multi_specific = query_type_ratios
    # query_distribution = [
    #     (SingleHopSpecificQuerySynthesizer(llm=llm), single_specific),
    #     (MultiHopAbstractQuerySynthesizer(llm=llm), multi_abstract),
    #     (MultiHopSpecificQuerySynthesizer(llm=llm), multi_specific),
    # ]

    return test_queries


class State(TypedDict):
    question: str
    context: List[Document]
    response: str

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = RAG_PROMPT.format(question=state["question"], context=docs_content)
    response = RAG_LLM.invoke(messages)
    return {"response": response.content}

def get_naive_retriever(vector_store: QdrantVectorStore, k: int = 10, **kwargs):
    return vector_store.as_retriever(search_kwargs={"k": k})

def get_bm25_retriever(docs: List[Document], k: int = 10, **kwargs):
    return BM25Retriever.from_documents(docs, k=k)

def get_cohere_reranker(base_retriever: RetrieverLike, cohere_model: str = 'rerank-v3.5', **kwargs):
    compressor = CohereRerank(model=cohere_model)
    return ContextualCompressionRetriever(base_compressor=compressor, base_retriever=base_retriever)

def get_multiquery_retriever(retriever: BaseRetriever, llm: BaseLanguageModel, **kwargs):
    return MultiQueryRetriever.from_llm(retriever=retriever, llm=llm)

def get_parent_document_retriever(parent_chunk_size: int, parent_chunk_overlap: int, child_chunk_size: int,
                                  child_chunk_overlap: int, qdrant_client: QdrantClient, collection_name: str,
                                  documents: List[Document], openai_embedding_model: str = 'text-embedding-3-small',
                                  embedding_size: int = 1536, is_semantic: bool = False, **kwargs):
    qdrant_client.create_collection(
        collection_name=collection_name,
        vector_config=models.VectorConfig(size=embedding_size,
        distance=models.DistanceType.COSINE))
    parent_document_vector_store = QdrantVectorStore(
        collection_name=collection_name,
        embedding=OpenAIEmbeddings(model=openai_embedding_model),
        client=qdrant_client)

    parent_splitter = None if is_semantic else RecursiveCharacterTextSplitter(chunk_size=parent_chunk_size, overlap=parent_chunk_overlap)
    # create an InMemoryStore for the parent document retriever
    in_memory_store = InMemoryStore()
    parent_document_retriever = ParentDocumentRetriever(
        vectorstore=parent_document_vector_store,
        docstore=in_memory_store,
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=child_chunk_size, overlap=child_chunk_overlap),
        parent_splitter=parent_splitter)
    parent_document_retriever.add_documents(documents)
    return parent_document_retriever

def load_sources(file_paths: List[str]):
    raw_documents = []

    # load all sources for RAG
    for file_path in file_paths:
        loader = TextLoader(file_path)
        raw_documents.extend(loader.load())
    return raw_documents

def get_ensemble_retriever(retrievers: List[RetrieverLike], weighting: List[float] = None, **kwargs):
    if weighting is None:
        weighting = [1.0 / len(retrievers)] * len(retrievers)
    else:
        assert (len(weighting) == len(retrievers)), "Length of weighting must be equal to length of retrievers!"
    return EnsembleRetriever(retrievers=retrievers, weights=weighting)

def set_up_vector_db(docs: List[Document], db_url: str = None, embedding_size: int = 1536, is_semantic: bool = False):
    # create vector store and add documents
    if db_url is None:
        db_url = ":memory:"

    if is_semantic:
        vector_store = QdrantVectorStore.from_documents(
            docs,
            RAG_EMBEDDING_MODEL,
            location=db_url,
            collection_name="wellness_semantic"
        )
        return None, vector_store
    else:
        client = QdrantClient(db_url)
        client.create_collection(
            collection_name="wellness",
            vectors_config=VectorParams(size=embedding_size, distance=Distance.COSINE)
        )
        vector_store = QdrantVectorStore(
            client=client,
            collection_name="wellness",
            embedding=RAG_EMBEDDING_MODEL
        )
        _ = vector_store.add_documents(docs)
        return client, vector_store

def retriever_factory(retriever: RetrieverLike):
    def retrieve_node(state: State):
        retrieved_docs = retriever.invoke(state["question"])
        return {"context": retrieved_docs}
    return retrieve_node

def build_rag_graph(retriever: Callable[[State], Dict[str, Any]]):
    rag_graph_builder = StateGraph(State).add_sequence([retriever, generate])
    rag_graph_builder.add_edge(START, "retrieve_node")  # node name from retriever_factory's __name__
    rag_graph = rag_graph_builder.compile()
    return rag_graph

def invoke_test_queries(rag_graph: StateGraph, dataset: Testset):
    # invoke each test query from the Testset object
    for test_row in dataset:
        response = rag_graph.invoke({"question": test_row.eval_sample.user_input})
        test_row.eval_sample.response = response["response"]
        test_row.eval_sample.retrieved_contexts = [context.page_content for context in response["context"]]
        time.sleep(SLEEP_SECONDS)  # to avoid rate limits
    return dataset

def gather_rag_statistics(data_set: Testset, llm: LangchainLLMWrapper):
    custom_run_config = RunConfig(timeout=60)
    results: EvaluationResult = evaluate(
        dataset=EvaluationDataset.from_pandas(data_set.to_pandas()),
        metrics=[LLMContextRecall(), Faithfulness(), FactualCorrectness(), ResponseRelevancy(), ContextEntityRecall(), NoiseSensitivity()],
        llm=llm,
        run_config=custom_run_config,
    )
    return results

def run_experiment_loop(golden_data_set: Testset, retriever_mapping: Dict[str, Callable], retriever_params: dict,
                        evaluator_llm: LangchainLLMWrapper, chunk_strategy: str) -> List[Dict]:
    rag_metrics = []
    for retriever_str, retriever_func in retriever_mapping.items():
        logger.warning(f"Beginning experiment loop: retriever: {retriever_str}, retriever_func: {retriever_func}, "
                       f"chunk_strategy: {chunk_strategy}")
        retriever_runnable: RetrieverLike = retriever_func(**retriever_params)

        # add retrievers (objects) to utilize for ensemble retriever at the end (mutating retriever_params dic)
        if retriever_str == "retriever":
            retriever_params.get("retrievers").append(retriever_runnable)

        retriever_node = retriever_factory(retriever=retriever_runnable)
        rag_graph = build_rag_graph(retriever=retriever_node)
        tested_data_set = invoke_test_queries(rag_graph, deepcopy(golden_data_set))
        rag_stats = gather_rag_statistics(tested_data_set, evaluator_llm)
        # EvaluationResult.scores is a list of per-row dicts; aggregate to mean per metric for this run
        scores_dict = pd.DataFrame(rag_stats.scores).mean(numeric_only=True, skipna=True).to_dict()
        rag_metrics.append({'retriever': retriever_str, 'chunking_strategy': f'{chunk_strategy}', **scores_dict})
        logger.warning(f"Done with experiment loop: retriever: {retriever_str}, retriever_func: {retriever_func.__name__}, "
                       f"chunk_strategy: {chunk_strategy}")
    return rag_metrics

def run_experiment(document_sources: List[str]):
    raw_corpus = load_sources(document_sources)

    # get golden data set for evaluation of retrieval methods
    golden_data_set: Testset = create_golden_data_set(raw_corpus, "gpt-4.1-nano", "text-embedding-3-small")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    wellness_docs = text_splitter.split_documents(raw_corpus)
    vector_db_client, vector_store = set_up_vector_db(wellness_docs)

    naive_retriever = get_naive_retriever(vector_store=vector_store)
    non_semantic_params = {
        "vector_store": vector_store,                       # get_naive_retriever
        # "docs": wellness_docs,                              # get_bm25_retriever
        # "base_retriever": naive_retriever,                  # get_cohere_reranker
        "retriever": naive_retriever,                       # get_multiquery_retriever
        "llm": ChatOpenAI(model="gpt-4.1-nano"),            # get_multiquery_retriever
        # "parent_chunk_size": PARENT_CHUNK_SIZE,             # get_parent_document_retriever
        # "parent_chunk_overlap": PARENT_CHUNK_OVERLAP,       # get_parent_document_retriever
        # "child_chunk_size": CHILD_CHUNK_SIZE,               # get_parent_document_retriever
        # "child_chunk_overlap": CHILD_CHUNK_OVERLAP,         # get_parent_document_retriever
        # "qdrant_client": vector_db_client,                  # get_parent_document_retriever
        # "collection_name": "wellness_guide",                # get_parent_document_retriever
        # "documents": raw_corpus,                            # get_parent_document_retriever
        "retrievers": [],                                   # get_ensemble_retriever
    }

    retriever_mapping = {
        "naive": get_naive_retriever,
        # "bm25": get_bm25_retriever,
        # "reranker": get_cohere_reranker,
        "multiquery": get_multiquery_retriever,
        # "parent_document": get_parent_document_retriever,
        # "ensemble": get_ensemble_retriever,
    }

    evaluator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))

    non_semantic_rag_metrics = run_experiment_loop(
        golden_data_set, retriever_mapping, non_semantic_params, evaluator_llm, "non_semantic_chunking")

    # semantic chunking portion of experiment
    semantic_chunker = SemanticChunker(RAG_EMBEDDING_MODEL)
    semantic_wellness_docs = semantic_chunker.split_documents(wellness_docs)
    _, semantic_vector_store = set_up_vector_db(semantic_wellness_docs, is_semantic=True)

    naive_semantic_retriever = get_naive_retriever(semantic_vector_store)
    semantic_params = {
        "vector_store": semantic_vector_store,              # get_naive_retriever
        # "docs": semantic_wellness_docs,                     # get_bm25_retriever
        # "base_retriever": naive_semantic_retriever,         # get_cohere_reranker
        # "retriever": naive_semantic_retriever,              # get_multiquery_retriever
        # "llm": ChatOpenAI(model="gpt-4.1-nano"),            # get_multiquery_retriever
        # "parent_chunk_size": PARENT_CHUNK_SIZE,             # get_parent_document_retriever
        # "parent_chunk_overlap": PARENT_CHUNK_OVERLAP,       # get_parent_document_retriever
        # "child_chunk_size": CHILD_CHUNK_SIZE,               # get_parent_document_retriever
        # "child_chunk_overlap": CHILD_CHUNK_OVERLAP,         # get_parent_document_retriever
        # "qdrant_client": vector_db_client,                  # get_parent_document_retriever
        # "collection_name": "semantic_wellness_guide",       # get_parent_document_retriever
        # "documents": semantic_wellness_docs,                # get_parent_document_retriever
        "retrievers": [],                                   # get_ensemble_retriever
    }

    semantic_mapping = {
        "naive": get_naive_retriever,
        # "bm25": get_bm25_retriever,
        # "reranker": get_cohere_reranker,
        # "multiquery": get_multiquery_retriever,
        # "parent_document": get_parent_document_retriever,
        # "ensemble": get_ensemble_retriever,
    }

    semantic_rag_metrics = run_experiment_loop(
        golden_data_set, semantic_mapping, semantic_params, evaluator_llm, "semantic_chunking")

    # turn metrics into a dataframe for easier analysis and plotting
    metrics_df = pd.DataFrame(non_semantic_rag_metrics + semantic_rag_metrics)
    return metrics_df



if __name__ == "__main__":
    metrics_df = run_experiment(document_sources=[f"{HEALTH_AND_WELLNESS_GUIDE_PATH}"])