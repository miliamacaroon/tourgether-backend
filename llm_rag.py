
import os
import pickle
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from openai import OpenAI
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, BaseMessage
from langchain_openai import ChatOpenAI
from typing import Annotated, Sequence, TypedDict, List
from dotenv import load_dotenv

# ===============================
# ENV + OPENAI SETUP
# ===============================
load_dotenv()
openai_key = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=openai_key)
agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=openai_key)

# ===============================
# STATE DEFINITION
# ===============================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], None]
    query: str
    documents: List[Document]

# ===============================
# RETRIEVAL UTILS
# ===============================
def embed_query(text):
    response = client.embeddings.create(model="text-embedding-3-large", input=text)
    return np.array(response.data[0].embedding, dtype="float32")

def normalize(scores_dict):
    if not scores_dict: return {}
    values = np.array(list(scores_dict.values()))
    if values.max() == values.min(): return {k: 1.0 for k in scores_dict.keys()}
    return {k: (v - values.min()) / (values.max() - values.min()) for k, v in scores_dict.items()}

def load_faiss_bm25(path_embeddings, path_index):
    with open(path_embeddings, "rb") as f:
        df = pickle.load(f)
    index = faiss.read_index(path_index)
    corpus = df['FAISS_TEXT'].str.lower().str.split().tolist()
    bm25 = BM25Okapi(corpus)
    return df, index, bm25

# Paths
BASE_PATH = "faiss_embeddings_region/"
attraction_df, attraction_index, attraction_bm25 = load_faiss_bm25(BASE_PATH + "attraction_embeddings_region.pkl", BASE_PATH + "faiss_attraction_region_cosine.index")
restaurant_df, restaurant_index, restaurant_bm25 = load_faiss_bm25(BASE_PATH + "restaurant_embeddings_region.pkl", BASE_PATH + "faiss_restaurant_region_cosine.index")

def hybrid_retrieval(query, domain="attraction", top_k=5):
    df, index, bm25 = (restaurant_df, restaurant_index, restaurant_bm25) if domain=="restaurant" else (attraction_df, attraction_index, attraction_bm25)

    # 1. FAISS Search
    vec = embed_query(query).reshape(1, -1)
    faiss.normalize_L2(vec)
    f_scores, f_indices = index.search(vec, 10)

    # Create the f_map that was missing
    f_map = normalize({int(idx): float(sc) for idx, sc in zip(f_indices[0], f_scores[0])})

    # 2. BM25 Search
    tokenized_query = query.lower().split()
    b_scores = bm25.get_scores(tokenized_query)
    b_indices = np.argsort(b_scores)[::-1][:10]

    # Create the b_map that was missing
    b_map = normalize({int(idx): float(b_scores[idx]) for idx in b_indices})

    # 3. Hybrid Ranking (Now f_map and b_map exist!)
    combined = {idx: 0.6 * f_map.get(idx, 0) + 0.4 * b_map.get(idx, 0) for idx in set(f_map) | set(b_map)}
    ranked = sorted(combined.items(), key=lambda x: x[1], reverse=True)[:top_k]

    docs = []
    for idx, _ in ranked:
        row = df.iloc[idx]
        docs.append(Document(
            page_content=row["FAISS_TEXT"],
            metadata={"PICTURE": row.get("PICTURE", ""), "NAME": row.get("NAME", "")}
        ))
    return docs

# ===============================
# GRAPH NODES
# ===============================
def retrieve_node(state: AgentState):
    query = state["query"]
    # Get attractions (with pictures) and restaurants
    attr_docs = hybrid_retrieval(query, "attraction", top_k=5)
    rest_docs = hybrid_retrieval(query, "restaurant", top_k=3)
    return {"documents": attr_docs + rest_docs}

def generate_node(state: AgentState):
    prompt = PromptTemplate.from_template(
        "You are a professional travel planner. Create a narrative itinerary for {query}.\n"
        "Context:\n{context}\nOutput the itinerary with clear Day sections using ### Day X."
    )
    context = "\n".join([d.page_content for d in state["documents"]])
    chain = prompt | agent_llm
    response = chain.invoke({"query": state["query"], "context": context})
    # We return both the message AND the documents so app.py can see the metadata
    return {"messages": [response], "documents": state["documents"]}

# ===============================
# BUILD GRAPH
# ===============================
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
graph = workflow.compile()
