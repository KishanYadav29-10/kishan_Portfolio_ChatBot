import os
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from dotenv import load_dotenv
import google.generativeai as genai

# --- LangChain / RAG bits ---
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# --- Parsers ---
from bs4 import BeautifulSoup
from pypdf import PdfReader


# -----------------------------
# Utils: load env + configure LLM
# -----------------------------
def configure_gemini():
    load_dotenv()
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Missing GOOGLE_API_KEY. Put it in a .env file or export it in your shell."
        )
    genai.configure(api_key=api_key)
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)
    return llm


# -----------------------------
# Parsers for PDF and HTML
# -----------------------------
def read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    texts = []
    for page in reader.pages:
        try:
            texts.append(page.extract_text() or "")
        except Exception:
            pass
    return "\n".join(texts).strip()


def read_html_text(html_path: Path) -> Tuple[str, List[dict]]:
    """
    Returns: (plain_text, project_cards)
    project_cards: list of dicts with keys: title, description, href (if present)
    """
    html = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(html, "html.parser")

    # Full visible text (good baseline for RAG)
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    plain_text = " ".join(soup.stripped_strings)

    # Extract project cards from your portfolio structure
    project_cards = []
    # common class in your markup: .apihu-port-single-service
    for card in soup.select(".apihu-port-single-service"):
        title_tag = card.select_one(".apihu-port-single-service-title")
        desc_tag = card.select_one(".apihu-port-single-service-text")
        link_tag = card.select_one(".apihu-port-single-service-btn")

        title = title_tag.get_text(strip=True) if title_tag else ""
        desc = desc_tag.get_text(" ", strip=True) if desc_tag else ""
        href = link_tag.get("href") if link_tag else None

        if title or desc:
            project_cards.append(
                {
                    "title": title,
                    "description": desc,
                    "href": href,
                }
            )

    return plain_text, project_cards


# -----------------------------
# Chunking
# -----------------------------
def chunk_text(text: str, chunk_size: int = 900, chunk_overlap: int = 150) -> List[str]:
    """
    Simple, fast character-level chunker. Keeps overlap for better retrieval.
    """
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == n:
            break
        start = end - chunk_overlap
        if start < 0:
            start = 0
    return chunks


# -----------------------------
# Build corpus from inputs
# -----------------------------
def collect_documents(input_paths: List[Path]) -> List[Document]:
    """
    Walks given paths. Reads: .pdf, .html, .htm, .txt
    Returns a list of LangChain Documents with metadata.
    """
    docs: List[Document] = []
    all_projects = []  # to enrich answers for "Demonstrates your AI projects" use-case

    for p in input_paths:
        if p.is_dir():
            file_list = sorted([fp for fp in p.rglob("*") if fp.is_file()])
        else:
            file_list = [p]

        for fp in tqdm(file_list, desc=f"Indexing {p}"):
            ext = fp.suffix.lower()
            try:
                if ext == ".pdf":
                    raw = read_pdf_text(fp)
                    for i, chunk in enumerate(chunk_text(raw)):
                        docs.append(
                            Document(
                                page_content=chunk,
                                metadata={"source": str(fp), "type": "resume/pdf", "chunk": i},
                            )
                        )
                elif ext in [".html", ".htm"]:
                    raw, projects = read_html_text(fp)
                    all_projects.extend(projects)
                    for i, chunk in enumerate(chunk_text(raw)):
                        docs.append(
                            Document(
                                page_content=chunk,
                                metadata={"source": str(fp), "type": "portfolio/html", "chunk": i},
                            )
                        )
                    # Add structured project cards as separate, dense docs
                    for proj in projects:
                        blob = f"Project: {proj.get('title','')}\nAbout: {proj.get('description','')}\nLink: {proj.get('href','')}"
                        docs.append(
                            Document(
                                page_content=blob,
                                metadata={
                                    "source": str(fp),
                                    "type": "portfolio/project_card",
                                    "title": proj.get("title", ""),
                                },
                            )
                        )
                elif ext == ".txt":
                    raw = fp.read_text(encoding="utf-8", errors="ignore")
                    for i, chunk in enumerate(chunk_text(raw)):
                        docs.append(
                            Document(
                                page_content=chunk,
                                metadata={"source": str(fp), "type": "notes/txt", "chunk": i},
                            )
                        )
                else:
                    # skip other file types quietly
                    continue
            except Exception as e:
                print(f"[WARN] Failed to parse {fp}: {e}")

    # If we collected explicit projects, add a summary doc to help retrieval
    if all_projects:
        lines = []
        for proj in all_projects:
            lines.append(
                f"- {proj.get('title','')} :: {proj.get('description','')} :: {proj.get('href','')}"
            )
        summary = "Kishan's Projects (from portfolio):\n" + "\n".join(lines)
        for i, chunk in enumerate(chunk_text(summary, chunk_size=1200, chunk_overlap=200)):
            docs.append(
                Document(
                    page_content=chunk,
                    metadata={"source": "portfolio_summary", "type": "portfolio/summary", "chunk": i},
                )
            )

    return docs


# -----------------------------
# Vector store helpers
# -----------------------------
def build_or_load_index(index_dir: Path, inputs: List[Path]) -> FAISS:
    embed = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if index_dir.exists() and (index_dir / "index.faiss").exists():
        print(f"[OK] Loading existing index from {index_dir}")
        return FAISS.load_local(str(index_dir), embed, allow_dangerous_deserialization=True)

    print("[*] Building FAISS index ...")
    docs = collect_documents(inputs)
    if not docs:
        raise RuntimeError("No documents collected. Check your input paths.")
    store = FAISS.from_documents(docs, embed)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    print(f"[OK] Saved index to {index_dir}")
    return store


# -----------------------------
# Prompt
# -----------------------------
SYSTEM_STYLE = """You are Kishan Yadav's portfolio assistant.
You ONLY answer questions about Kishan: his resume, skills, certifications, and projects.
If the question is outside that scope, say you’re limited to Kishan-related topics.

Tone: friendly, concise, helpful. Prefer bullet points and concrete details.

When citing projects, include the project name and (if present) the GitHub link from context.
If the answer isn’t in the context, say: “I don’t have that detail in my portfolio/resume yet.”"""

RAG_TEMPLATE = """{system}

Use the following context (resume + portfolio chunks) to answer the user:

<context>
{context}
</context>

User: {question}

Guidelines:
- If projects are referenced, name them exactly as in context.
- When relevant, mention tools/stack (e.g., Python, Streamlit, LangChain, HuggingFace, OpenCV, MobileNetV2, Stable Diffusion, Mistral-7B, Ollama).
- If context mentions dates or metrics (e.g., Jan–Jun 2025, 90%+ accuracy), include them.
- If unsure or not found in context, say so briefly.

Answer:"""


def make_chain(llm: ChatGoogleGenerativeAI, retriever) -> RetrievalQA:
    prompt = PromptTemplate(
        template=RAG_TEMPLATE,
        input_variables=["context", "question"],
        partial_variables={"system": SYSTEM_STYLE},
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=False,
    )
    return chain


# -----------------------------
# CLI
# -----------------------------
def cmd_index(args):
    index_dir = Path(args.index_dir)
    inputs = [Path(p) for p in args.inputs]
    if args.reset and index_dir.exists():
        shutil.rmtree(index_dir)
        print(f"[OK] Cleared existing index at {index_dir}")
    _ = build_or_load_index(index_dir, inputs)


def cmd_chat(args):
    # Load LLM
    llm = configure_gemini()

    # Load / build index
    index_dir = Path(args.index_dir)
    inputs = [Path(p) for p in args.inputs] if args.inputs else []
    store = build_or_load_index(index_dir, inputs) if inputs else FAISS.load_local(
        str(index_dir),
        HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2"),
        allow_dangerous_deserialization=True,
    )

    retriever = store.as_retriever(search_kwargs={"k": 6})
    chain = make_chain(llm, retriever)

    print("\nKishan’s RAG Chatbot (Gemini) — ask about resume, skills, certifications, or projects.")
    print("Type 'exit' to quit.\n")
    while True:
        q = input("You: ").strip()
        if not q:
            continue
        if q.lower() in {"exit", "quit"}:
            print("Bye!")
            break
        try:
            ans = chain.run(q)
        except Exception as e:
            ans = f"[Error] {e}"
        print(f"\nBot: {ans}\n")


def build_arg_parser():
    p = argparse.ArgumentParser(description="Kishan's local RAG chatbot (Gemini).")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_index = sub.add_parser("index", help="Build (or rebuild) FAISS index from inputs")
    p_index.add_argument("--index-dir", default="rag_index", help="Directory for FAISS index")
    p_index.add_argument("--reset", action="store_true", help="Delete existing index before building")
    p_index.add_argument(
        "inputs",
        nargs="+",
        help="Paths to resume/portfolio files or folders (pdf, html, txt). Example: ./assets ./portfolio.html",
    )
    p_index.set_defaults(func=cmd_index)

    p_chat = sub.add_parser("chat", help="Start local CLI chat")
    p_chat.add_argument("--index-dir", default="rag_index", help="Directory where FAISS index is stored")
    p_chat.add_argument(
        "--inputs",
        nargs="*",
        help="(Optional) if provided, will build/rebuild index first from these paths.",
    )
    p_chat.set_defaults(func=cmd_chat)

    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
