from langgraph.graph import StateGraph, START, END
from langchain.tools import tool
from langchain.chat_models import init_chat_model, BaseChatModel
from langchain.messages import SystemMessage
import os
import pymupdf
import re
from qdrant_client import QdrantClient, models
from typing import Iterable, Optional
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel, Field


BATCH = 512
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION = "papers"
DB_URL = "http://localhost:6333"
DATA_DIRECTORY = "doclake"

embedding_model = SentenceTransformer(MODEL_NAME)
client = QdrantClient(url=DB_URL)
llm_model: BaseChatModel = init_chat_model("openai:gpt-5-nano-2025-08-07")


def stable_paper_id(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip()


def extract_pdf_text(pdf_path: str) -> str:
    doc = pymupdf.open(pdf_path)
    parts: list[str] = []
    for page in doc:
        t = page.get_text("text")
        if isinstance(t, str) and t.strip():
            parts.append(t)
    return "\n\n".join(parts)


def extract_txt_text(txt_path: str) -> str:
    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def chunks(text: str, chunk_size: int = 1500, overlap: int = 200) -> Iterable[str]:
    text = normalize_text(text)
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(text):
        yield text[i : i + chunk_size]
        i += step


def reset_collection():
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)

    dim = embedding_model.get_sentence_embedding_dimension()
    if dim: 
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )


def setup() -> None:
    dim = embedding_model.get_sentence_embedding_dimension()
    assert dim is not None, "Embedding dimension is unknown (model didn't report it)."

    reset_collection() 
    existing = {c.name for c in client.get_collections().collections}
    if COLLECTION not in existing:
        client.create_collection(
            collection_name=COLLECTION,
            vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
        )

    next_id = 0
    for name in os.listdir(DATA_DIRECTORY):
        lower_name = name.lower()
        if not (lower_name.endswith(".pdf") or lower_name.endswith(".txt")):
            continue

        pdf_path = os.path.join(DATA_DIRECTORY, name)
        paper_id = stable_paper_id(pdf_path)
        if lower_name.endswith(".pdf"):
            full_text = extract_pdf_text(pdf_path)
        else:
            full_text = extract_txt_text(pdf_path)
        if not full_text:
            continue

        ch = list(chunks(full_text))
        vecs = embedding_model.encode(ch, normalize_embeddings=True)

        batch: list[models.PointStruct] = []
        for chunk_idx, (chunk_text, vec) in enumerate(zip(ch, vecs), start=1):
            batch.append(
                models.PointStruct(
                    id=next_id,
                    vector=vec.tolist(),
                    payload={
                        "paper_id": paper_id,
                        "chunk_id": chunk_idx,
                        "text": chunk_text,
                        "model": MODEL_NAME,
                        "source_file": name,
                    },
                )
            )

            next_id += 1

            if len(batch) >= BATCH:
                client.upsert(collection_name=COLLECTION, points=batch)
                batch.clear()

        if batch:
            client.upsert(collection_name=COLLECTION, points=batch)


@tool
def retrieve_docs(query: str) -> list[str]:
    """Query the paper database for relevant chunks based on a user query."""
    q_vec = embedding_model.encode(query, normalize_embeddings=True).tolist()
    response = client.query_points(
        collection_name=COLLECTION,
        query=q_vec,
        with_payload=True,
        with_vectors=False,
        limit=4,
    )
    return [p.payload.get("text", "") for p in response.points if p.payload is not None]


class RAGState(BaseModel):
    question: str
    use_retrieval: Optional[bool] = None
    contexts: Optional[list[str]] = None
    answer: Optional[str] = None 
    

class RouteDecision(BaseModel) : 
    use_retrieval : bool = Field(...,description="Whether external knowledge/search would improve answer.")
    

router_model = llm_model.with_structured_output(RouteDecision)

RouterSys = SystemMessage(content="You are a Router for a RAG System, " \
    "Return use_retrieval=true if the user asks about facts that need citation from documents "
    "or you can explain better with context from a paper: i.e. a modern framework"
    "Return false for pure reasoning, opinions, brainstorming, or questions answerable from the conversation alone.")

def retrieve_node(state: RAGState) -> dict:
    contexts = retrieve_docs.invoke(state.question)
    return {"contexts": contexts}


def decision_node(state: RAGState) -> dict:
    
    #decision = router_model.invoke([RouterSys, ("human", state.question)])
    #return {"use_retrieval" : decision.use_retrieval}

    return {"use_retrieval" : True} 


def answer_final(state:RAGState) -> dict : 

    if state.contexts : 
        ctx = "\n\n".join(state.contexts)
        resp = llm_model.invoke([
            ("system", f"Use the following context to answer:\n\n{ctx}"),
            ("user", state.question),
        ])
    else: 
        resp = llm_model.invoke([
            ("system" , "answer the user query"),
            ("user" , state.question)
        ])

    # print(resp.content)
    return {"answer" : resp.content}

builder = StateGraph(state_schema=RAGState)
builder.add_node("retrieve", retrieve_node)
builder.add_node("decision", decision_node)
builder.add_node("answer", answer_final)
builder.add_edge(START, "decision")
builder.add_conditional_edges("decision", lambda s: "retrieve" if s.use_retrieval else "answer")
builder.add_edge("retrieve", "answer")
builder.add_edge("answer", END)
graph = builder.compile()


def __run__() -> None:
    if os.getenv("OPENAI_API_KEY") is None:
        print("Environment variable OPENAI_API_KEY is not set")
        return

    setup()
    while True:
        user_prompt = input(">").strip()
        if user_prompt.lower() in {"exit", "quit"}:
            break
        if not user_prompt:
            print("Prompt cannot be empty.")
            continue

        result = graph.invoke({"question": user_prompt})
        # print(result.answer if result.answer != None else "No answer generated.")
        # print(result)
        print(result.get("answer", "No answer generated."))
        # 


if __name__ == "__main__":
    __run__()
