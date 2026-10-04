import os
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from pydantic import BaseModel, Field

load_dotenv()

ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://hoda-khar.github.io/Portfolio/").split(",") if o.strip()]
PERSONA = os.getenv("PERSONA", "third").lower()
from fastapi import FastAPI

app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "ok", "message": "Le chatbot est en ligne"}

PERSONA_TEXT = (
    "Parle de Hoda à la troisième personne (« Hoda a réalisé… »). Tu es un assistant virtuel."
    if PERSONA == "third"
    else "Exprime-toi à la première personne, comme si tu étais Hoda (« j'ai réalisé… »)."
)

SYSTEM_PROMPT = f"""Tu es l'assistant du portfolio de Hoda Kharbouche.
RÈGLES STRICTES :
1. Réponds UNIQUEMENT à partir du CONTEXTE ci-dessous.
2. Si l'information n'y est pas, dis : « Je n'ai pas cette information dans le portfolio. »
3. N'invente jamais de chiffre, de date ou de projet.
4. {PERSONA_TEXT}"""

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.1)
embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

def create_vectorstore():
    data_dir = Path("data")
    md_files = sorted(data_dir.glob("*.md"))
    if not md_files:
        return None

    headers = [("#", "h1"), ("##", "h2"), ("###", "h3")]
    header_splitter = MarkdownHeaderTextSplitter(headers, strip_headers=False)
    char_splitter = RecursiveCharacterTextSplitter(chunk_size=900, chunk_overlap=120)

    chunks = []
    for path in md_files:
        text = path.read_text(encoding="utf-8")
        for section in header_splitter.split_text(text):
            for piece in char_splitter.split_documents([section]):
                titles = " > ".join(piece.metadata[k] for k in ("h1", "h2", "h3") if k in piece.metadata)
                piece.page_content = f"[{path.stem} | {titles}]\n{piece.page_content}"
                piece.metadata["source"] = path.name
                chunks.append(piece)

    return Chroma.from_documents(documents=chunks, embedding=embeddings)

vectorstore = create_vectorstore()
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 5}) if vectorstore else None

app = FastAPI(title="Chatbot Portfolio Hoda - Cloud Gemini")
app.add_middleware(CORSMiddleware, allow_origins=ORIGINS, allow_methods=["*"], allow_headers=["*"])

class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=1500)

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=500)
    history: list[Turn] = Field(default_factory=list, max_length=8)

class ChatResponse(BaseModel):
    answer: str
    sources: list[str]

_hits = defaultdict(deque)

def rate_limit(request: Request, limit: int = 20, window: int = 60):
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    while _hits[ip] and now - _hits[ip][0] > window:
        _hits[ip].popleft()
    if len(_hits[ip]) >= limit:
        raise HTTPException(429, "Trop de requêtes, réessayez dans une minute.")
    _hits[ip].append(now)

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    rate_limit(request)
    if not retriever:
        raise HTTPException(500, "Base de données vide.")

    docs = await retriever.ainvoke(req.message)
    context = "\n---\n".join([d.page_content for d in docs])

    messages = [SystemMessage(content=f"{SYSTEM_PROMPT}\n\nCONTEXTE :\n{context}")]
    for t in req.history[-6:]:
        messages.append(HumanMessage(t.content) if t.role == "user" else AIMessage(t.content))
    messages.append(HumanMessage(req.message))

    try:
        response = await llm.ainvoke(messages)
    except Exception as exc:
        raise HTTPException(503, f"Gemini est indisponible : {exc}")

    sources = sorted(list(set(d.metadata.get("source", "?") for d in docs)))
    return ChatResponse(answer=response.content.strip(), sources=sources)