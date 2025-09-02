import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from rag import build_or_load_index, make_chain, configure_gemini
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

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
    GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    allow_dangerous_deserialization=True,
)
retriever = store.as_retriever(search_kwargs={"k": 6})
qa_chain = make_chain(llm, retriever)

# Request/Response models
class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

import traceback

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    try:
        result = qa_chain.invoke({"query": req.message})

        if isinstance(result, dict):
            answer = (
                result.get("result")
                or result.get("answer")
                or str(result)
            )
        else:
            answer = str(result)

        if not answer.strip():
            answer = "⚠️ Sorry, I couldn’t find relevant info in my portfolio/resume."

        return ChatResponse(response=answer)

    except Exception as e:
        traceback.print_exc()  # 👈 will show full error in Render logs
        return ChatResponse(response=f"[Error] {str(e)}")




@app.get("/")
async def root():
    return {"status": "Kishan RAG API running"}

