import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import build_or_load_index, make_chain, configure_gemini
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Load .env (Gemini key)
load_dotenv()

# Init FastAPI
app = FastAPI(title="Kishan RAG API")

# Allow frontend requests (GitHub Pages, localhost, etc.)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to ["https://your-github-pages-url"] for stricter
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load index + chain once
INDEX_DIR = "rag_index"
llm = configure_gemini()
store = FAISS.load_local(
    INDEX_DIR,
    HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
    allow_dangerous_deserialization=True,
)
retriever = store.as_retriever(search_kwargs={"k": 6})
qa_chain = make_chain(llm, retriever)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = qa_chain.invoke({"query": req.message})
        
        # if the result is a dict, get the "result" key
        if isinstance(result, dict):
            answer = result.get("result", str(result))
        else:
            answer = str(result)

        return ChatResponse(response=answer)
    except Exception as e:
        return ChatResponse(response=f"[Error] {str(e)}")


@app.get("/")
async def root():
    return {"status": "Kishan RAG API running"}
