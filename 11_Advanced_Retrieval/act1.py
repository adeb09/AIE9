from langchain_community.document_loaders import TextLoader
from ragas.llms import LangchainLLMWrapper, BaseRagasLLM
from ragas.embeddings import LangchainEmbeddingsWrapper, BaseRagasEmbeddings
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents.base import Document
from ragas.testset import TestsetGenerator
from dotenv import load_dotenv
from ragas.testset.graph import KnowledgeGraph
from ragas.testset.graph import Node, NodeType
from ragas.testset.transforms import default_transforms, apply_transforms
from ragas.testset.synthesizers import default_query_distribution, SingleHopSpecificQuerySynthesizer, MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer
from ragas.testset.synthesizers.testset_schema import Testset

from typing import List, Tuple
import pickle


HEALTH_AND_WELLNESS_GUIDE_PATH = "data/HealthAndWellnessGuide.txt"
GOLDEN_DATA_SET_FILE = "golden_data_set.pkl"
TEST_SET_SIZE = 20


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


def create_golden_data_set(documents: List[Document], llm: BaseRagasLLM, embedding_model: BaseRagasEmbeddings,
                           use_cache: bool = True) -> Testset:
    # create knowledge graph from corpus (utilized for generating queries via LLM generator)
    # kg = create_knowledge_graph(documents, llm, embedding_model, cache_file='kg.json')
    try:
        test_queries = pickle.load(open(GOLDEN_DATA_SET_FILE, 'rb'))
    except FileNotFoundError:
        generator = TestsetGenerator(llm=llm, embedding_model=embedding_model)
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


def run():
    load_dotenv()

    # load wellness guide and get documents
    loader = TextLoader("data/HealthWellnessGuide.txt")
    docs = loader.load()

    # LLM and embedding models to utilize for KG transformations and synthetic data generation (synthetic test queries)
    llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-nano"))
    embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings())

    golden_data_set: Testset = create_golden_data_set(docs, llm, embeddings)
    print(type(golden_data_set))
    print(len(golden_data_set))
    print(golden_data_set.to_pandas())
    print(dir(golden_data_set))



if __name__ == "__main__":
    run()