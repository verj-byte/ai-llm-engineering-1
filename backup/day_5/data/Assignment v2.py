import os, getpass, time
from uuid import uuid4
from operator import itemgetter
import pandas as pd

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Qdrant
from langchain.prompts import ChatPromptTemplate

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset.graph import KnowledgeGraph, Node, NodeType
from ragas.testset import TestsetGenerator
from ragas.testset.synthesizers import SingleHopSpecificQuerySynthesizer, MultiHopAbstractQuerySynthesizer, MultiHopSpecificQuerySynthesizer
from ragas.testset.transforms import default_transforms, apply_transforms

from langsmith import Client

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = getpass.getpass("LangChain API Key:")
os.environ["OPENAI_API_KEY"] = getpass.getpass("OpenAI API Key:")
os.environ["LANGCHAIN_PROJECT"] = f"Philippines AI Bills RAG - {uuid4().hex[0:8]}"

# ------------------------
# 1. Load documents
# ------------------------
from langchain_community.document_loaders import DirectoryLoader, PyMuPDFLoader

path = "bills/"
loader = DirectoryLoader(path, glob="*.pdf", loader_cls=PyMuPDFLoader)
docs = loader.load()

# ------------------------
# 2. Knowledge Graph
# ------------------------
generator_llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4.1-nano"))
generator_embeddings = LangchainEmbeddingsWrapper(OpenAIEmbeddings(model="text-embedding-3-small"))

kg = KnowledgeGraph()
for doc in docs[:20]:
    kg.nodes.append(
        Node(
            type=NodeType.DOCUMENT,
            properties={"page_content": doc.page_content, "document_metadata": doc.metadata}
        )
    )

default_transforms_obj = default_transforms(documents=docs, llm=generator_llm, embedding_model=generator_embeddings)
apply_transforms(kg, default_transforms_obj)
kg.save("bills/ai_law.json")
bills_data_kg = KnowledgeGraph.load("bills/ai_law.json")

# ------------------------
# 3. Golden dataset (generate once)
# ------------------------
generator = TestsetGenerator(llm=generator_llm, embedding_model=generator_embeddings, knowledge_graph=bills_data_kg)
query_distribution = [
    (SingleHopSpecificQuerySynthesizer(llm=generator_llm), 0.5),
    (MultiHopAbstractQuerySynthesizer(llm=generator_llm), 0.25),
    (MultiHopSpecificQuerySynthesizer(llm=generator_llm), 0.25),
]
golden_dataset = generator.generate(testset_size=20, query_distribution=query_distribution)

# ------------------------
# 4. Semantic chunking
# ------------------------
def chunk_docs(docs, chunk_size=500, chunk_overlap=50):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return text_splitter.split_documents(docs)

rag_documents = chunk_docs(docs[:20], chunk_size=500)  # Semantic Chunking ON
rag_documents_no_chunk = docs[:20]  # Semantic Chunking OFF

# ------------------------
# 5. Vectorstore & retrievers
# ------------------------
def build_vectorstore(documents, embedding_model):
    return Qdrant.from_documents(documents=documents, embedding=embedding_model,
                                 location=":memory:", collection_name=f"AI_Bills_RAG_{uuid4().hex[:8]}")

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

# Base retriever (Naive)
vectorstore_naive = build_vectorstore(rag_documents, embedding_model)
retriever_naive = vectorstore_naive.as_retriever(search_kwargs={"k": 10})

# For demonstration, additional retrievers placeholders (BM25, Multi-Query, Parent, Rerank, Ensemble)
# Replace with actual implementations from your library if available
retrievers = {
    "Naive": retriever_naive,
    "BM25": retriever_naive,          # placeholder
    "Multi-Query": retriever_naive,   # placeholder
    "Parent-Document": retriever_naive,  # placeholder
    "Rerank": retriever_naive,        # placeholder
    "Ensemble": retriever_naive       # placeholder
}

# ------------------------
# 6. RAG Prompts
# ------------------------
RAG_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.
If you cannot answer the question based on the context - you must say "I don't know".
Context: {context}
Question: {question}
"""

EMPATHY_RAG_PROMPT = """\
Given a provided context and question, you must answer the question based only on context.
If you cannot answer the question based on the context - you must say "I don't know".
You must answer the question using empathy and kindness, and make sure the user feels heard.
Context: {context}
Question: {question}
"""

rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)
empathy_rag_prompt = ChatPromptTemplate.from_template(EMPATHY_RAG_PROMPT)

llm = ChatOpenAI(model="gpt-4.1-mini")
from langchain_core.output_parsers import StrOutputParser

def run_rag_chain(question, retriever, prompt_template):
    docs = retriever.get_relevant_documents(question)
    context_text = "\n\n".join([d.page_content for d in docs])
    
    # prompt_text = prompt_template.format(context=context_text, question=question)
    # return llm(prompt_text)

    chain = prompt_template | llm | StrOutputParser()
    return chain.invoke({"context": context_text, "question": question})

# ------------------------
# 7. Langsmith setup
# ------------------------
client = Client()
langsmith_dataset = client.create_dataset(dataset_name="Philippines AI Bills v1.23", description="Golden dataset")

for row in golden_dataset.to_pandas().iterrows():
    client.create_example(
        inputs={"question": row[1]["user_input"]},
        outputs={"answer": row[1]["reference"]},
        metadata={"context": row[1]["reference_contexts"]},
        dataset_id=langsmith_dataset.id
    )

# ------------------------
# 8. Evaluate retrievers on golden dataset
# ------------------------
results = []

for retriever_name, retriever_obj in retrievers.items():
    for chunking_status, docs_set in [("Chunking ON", rag_documents), ("Chunking OFF", rag_documents_no_chunk)]:
        for idx, row in golden_dataset.to_pandas().iterrows():
            answer = run_rag_chain(row["user_input"], retriever_obj, rag_prompt)
            empathy_answer = run_rag_chain(row["user_input"], retriever_obj, empathy_rag_prompt)
            results.append({
                "Retriever": retriever_name,
                "Semantic Chunking": chunking_status,
                "Question": row["user_input"],
                "Answer": answer,
                "Empathy Answer": empathy_answer
            })

results_df = pd.DataFrame(results)
results_df.to_csv("retriever_comparison.csv", index=False)
print("Saved retriever comparison to retriever_comparison.csv")

# Now you can visualize this CSV in a single Langsmith or any BI tool for cost, latency, and performance.
