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
from huggingface_hub import hf_hub_download

# ===============================
# LOAD ENVIRONMENT VARIABLES
# ===============================
load_dotenv()  # optional locally; Railway injects automatically

OPENAI_KEY = os.getenv("OPENAI_API_KEY")
TAVILY_KEY = os.getenv("TAVILY_API_KEY")  # if needed
HF_TOKEN = os.getenv("HF_TOKEN")

# ===============================
# OPENAI SETUP
# ===============================
client = OpenAI(api_key=OPENAI_KEY)
agent_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.4, api_key=OPENAI_KEY)

# ===============================
# PATH HELPERS (download files from HF if not present locally)
# ===============================
def download_hf_file(repo_id: str, filename: str, local_dir="downloads") -> str:
    os.makedirs(local_dir, exist_ok=True)
    local_path = os.path.join(local_dir, os.path.basename(filename))
    if not os.path.exists(local_path):
        print(f"📥 Downloading {filename} from Hugging Face...")
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            token=HF_TOKEN,
            cache_dir=local_dir
        )
        print(f"✅ Downloaded {filename} to {local_path}")
    return local_path

# ===============================
# FAISS + BM25 LOAD
# ===============================
def load_faiss_bm25(repo_id: str, embeddings_file: str, index_file: str):
    path_embeddings = download_hf_file(repo_id, embeddings_file)
    path_index = download_hf_file(repo_id, index_file)
    
    with open(path_embeddings, "rb") as f:
        df = pickle.load(f)
    index = faiss.read_index(path_index)
    corpus = df['FAISS_TEXT'].str.lower().str.split().tolist()
    bm25 = BM25Okapi(corpus)
    return df, index, bm25

# ===============================
# DOWNLOAD EMBEDDINGS
# ===============================
REPO_ID = "intxnk01/tourgether-models"
attraction_df, attraction_index, attraction_bm25 = load_faiss_bm25(
    REPO_ID,
    "faiss_embeddings_region/attraction_embeddings_region.pkl",
    "faiss_embeddings_region/faiss_attraction_region_cosine.index"
)
restaurant_df, restaurant_index, restaurant_bm25 = load_faiss_bm25(
    REPO_ID,
    "faiss_embeddings_region/restaurant_embeddings_region.pkl",
    "faiss_embeddings_region/faiss_restaurant_region_cosine.index"
)

# ===============================
# HYBRID RETRIEVAL
# ===============================
def embed_query(text):
    response = client.embeddings.create(model="text-embedding-3-large", input=text)
    return np.array(response.data[0].embedding, dtype="float32")

def normalize(scores_dict):
    if not scores_dict: return {}
    values = np.array(list(scores_dict.values()))
    if values.max() == values.min(): return {k: 1.0 for k in scores_dict.keys()}
    return {k: (v - values.min()) / (values.max() - values.min()) for k, v in scores_dict.items()}

def hybrid_retrieval(query, domain="attraction", top_k=5):
    df, index, bm25 = (restaurant_df, restaurant_index, restaurant_bm25) if domain=="restaurant" else (attraction_df, attraction_index, attraction_bm25)
    
    # FAISS
    vec = embed_query(query).reshape(1, -1)
    faiss.normalize_L2(vec)
    f_scores, f_indices = index.search(vec, 10)
    f_map = normalize({int(idx): float(sc) for idx, sc in zip(f_indices[0], f_scores[0])})
    
    # BM25
    tokenized_query = query.lower().split()
    b_scores = bm25.get_scores(tokenized_query)
    b_indices = np.argsort(b_scores)[::-1][:10]
    b_map = normalize({int(idx): float(b_scores[idx]) for idx in b_indices})
    
    # Hybrid
    combined = {idx: 0.6*f_map.get(idx,0)+0.4*b_map.get(idx,0) for idx in set(f_map)|set(b_map)}
    ranked = sorted(combined.items(), key=lambda x:x[1], reverse=True)[:top_k]
    
    docs = []
    for idx, _ in ranked:
        row = df.iloc[idx]
        docs.append(Document(
            page_content=row["FAISS_TEXT"],
            metadata={"PICTURE": row.get("PICTURE",""), "NAME": row.get("NAME","")}
        ))
    return docs

# ===============================
# GRAPH NODES
# ===============================
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], None]
    query: str
    documents: List[Document]

def retrieve_node(state: AgentState):
    query = state["query"]
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
