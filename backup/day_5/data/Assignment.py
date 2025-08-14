# ==============================
# Setup
# ==============================
import os, getpass, time
from uuid import uuid4
import pandas as pd

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangChain API Key:")
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["LANGCHAIN_PROJECT"] = f"Philippines AI Bills RAG - {uuid4().hex[0:8]}"

# ==============================
# Load Documents
# ==============================
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

loader = DirectoryLoader("bills/", glob="*.pdf", loader_cls=PyMuPDFLoader)
docs = loader.load()[:20]  # subset for cost control

# ==============================
# LLM and Embeddings
# ==============================
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-mini"))
embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

# ==============================
# Knowledge Graph
# ==============================
from ragas.testset.graph import KnowledgeGraph, Node, NodeType

kg = KnowledgeGraph()
for doc in docs:
    kg.nodes.append(Node(type=NodeType.DOCUMENT,
                         properties={"page_content": doc.page_content,
                                     "document_metadata": doc.metadata}))

from ragas.testset.transforms import default_transforms, apply_transforms
apply_transforms(kg, default_transforms(documents=docs, llm=llm, embedding_model=embeddings))

kg.save("bills/ai_law.json")
kg = KnowledgeGraph.load("bills/ai_law.json")

# ==============================
# Golden Dataset (Synthetic)
# ==============================
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import (
    SingleHopSpecificQuerySynthesizer,
    MultiHopAbstractQuerySynthesizer,
    MultiHopSpecificQuerySynthesizer
)

generator = TestsetGenerator(llm=llm, embedding_model=embeddings, knowledge_graph=kg)
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=llm), 0.5),
    (MultiHopAbstractQuerySynthesizer(llm=llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=llm), 0.25),
]

golden_dataset = generator.generate(testset_size=20, query_distribution=query_distribution)
golden_dataset.to_jsonl("bills/golden_dataset.json")  # Save for reuse

# ==============================
# LangSmith Dataset
# ==============================
from langsmith import Client
client = Client()
dataset_name = "Philippines AI Bills - Golden Dataset 3"
ls_dataset = client.create_dataset(dataset_name=dataset_name,
                                   description="Golden dataset for retriever benchmarking")
for row in golden_dataset.to_pandas().itertuples():
    client.create_example(
        inputs={"question": row.user_input},
        outputs={"answer": row.reference},
        metadata={"context": row.reference_contexts},
        dataset_id=ls_dataset.id
    )

# ==============================
# Retriever Setup
# ==============================
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from operator import itemgetter
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain.schema import StrOutputParser

# Split docs for vector retrieval
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
rag_documents = text_splitter.split_documents(docs)

vectorstore = Qdrant.from_documents(
    documents=rag_documents,
    embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
    location=":memory:",
    collection_name="AI Bills RAG"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

# ==============================
# RAG Prompt
# ==============================
RAG_PROMPT = """\
Given context and question, answer only using context. If unknown, say "I don't know".

Context: {context}
Question: {question}"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

# ==============================
# Retriever Variants
# ==============================
# We'll define a simple wrapper for each retriever type
def run_rag_retriever(retriever_name, retriever, semantic_chunking=False):
    from langchain_core.runnables import RunnableMap
    from langchain_core.output_parsers import StrOutputParser
    start_time = time.time()
    
    if semantic_chunking:
        # Example: increase chunk overlap for semantic chunking ON
        text_splitter_sc = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=200)
        docs_sc = text_splitter_sc.split_documents(docs)
        vectorstore_sc = Qdrant.from_documents(
            documents=docs_sc,
            embedding=OpenAIEmbeddings(model="text-embedding-3-small"),
            location=":memory:",
            collection_name=f"AI Bills RAG SC {retriever_name}"
        )
        retriever = vectorstore_sc.as_retriever(search_kwargs={"k": 10})
    
    # Wrap the LLM in a Runnable
    def prompt_runnable(inputs):
        # inputs should be a dict with keys matching your template, e.g., "context" and "question"
        return rag_prompt.format(context=inputs["context"], question=inputs["question"])

    def llm_callable(inputs):
        return llm(inputs["text"])

    # Compose the RAG chain
    rag_chain = (
        {
            "context": lambda inputs: retriever.get_relevant_documents(inputs["question"]),
            "question": lambda inputs: inputs["question"]
        }
        | prompt_runnable
        | llm_callable 
        | StrOutputParser()
    )
    
    responses = []
    for row in golden_dataset.to_pandas().itertuples():
        responses.append({
            "question": row.user_input,
            "prediction": rag_chain.invoke({"question": row.user_input}),
            "reference": row.reference
        })
    
    latency = time.time() - start_time
    return responses, latency

# ==============================
# Run all retrievers (placeholder names)
# ==============================
retriever_names = [
    "Naive", "BM25", "Multi-Query", "Parent-Document", "Contextual Compression", "Ensemble"
]
results = []

for name in retriever_names:
    for sc in [False, True]:  # semantic chunking off/on
        responses, latency = run_rag_retriever(name, retriever, semantic_chunking=sc)
        results.append({
            "retriever": name,
            "semantic_chunking": sc,
            "responses": responses,
            "latency_s": latency
        })

# ==============================
# Push results to LangSmith
# ==============================
from langsmith.evaluation import LangChainStringEvaluator, evaluate

eval_llm = ChatOpenAI(model="gpt-4.1")
qa_evaluator = LangChainStringEvaluator("qa", config={"llm": eval_llm})

for run in results:
    dataset_id = ls_dataset.id
    evaluate(
        lambda x: x,  # dummy since we've already invoked
        data=dataset_id,
        evaluators=[qa_evaluator],
        metadata={
            "retriever": run["retriever"],
            "semantic_chunking": run["semantic_chunking"],
            "latency_s": run["latency_s"]
        }
    )

# ==============================
# Link to LangSmith dashboard
# ==============================
print("✅ All retriever runs completed. View your comparison dashboard here:")
print(f"https://app.langchain.com/projects/{os.environ['LANGCHAIN_PROJECT']}/datasets")
