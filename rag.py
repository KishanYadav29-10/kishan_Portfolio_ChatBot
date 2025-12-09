import os
import argparse
import shutil
from pathlib import Path
from typing import List, Tuple
from tqdm import tqdm

from dotenv import load_dotenv

load_dotenv(dotenv_path="G:\\Data science and Ai\\Portfolio\\KIshan Personal Assitant\\.env")


# LangChain modern imports
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Parsers
from bs4 import BeautifulSoup
from pypdf import PdfReader


# -----------------------------
# Configure OpenAI
# -----------------------------
def configure_openai():
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing OPENAI_API_KEY")

    return ChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.3
    )


# -----------------------------
# File parsers
# -----------------------------
def read_pdf_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        try:
            pages.append(page.extract_text() or "")
        except:
            pass
    return "\n".join(pages)


def read_html_text(html_path: Path) -> Tuple[str, List[dict]]:
    raw = html_path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = " ".join(soup.stripped_strings)

    projects = []
    for card in soup.select(".apihu-port-single-service"):
        title = card.select_one(".apihu-port-single-service-title")
        desc = card.select_one(".apihu-port-single-service-text")
        link = card.select_one(".apihu-port-single-service-btn")

        projects.append({
            "title": title.get_text(strip=True) if title else "",
            "description": desc.get_text(strip=True) if desc else "",
            "href": link.get("href") if link else None
        })

    return text, projects


# -----------------------------
# Chunk text
# -----------------------------
def chunk_text(text: str, chunk_size=900, overlap=150):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# -----------------------------
# Build documents
# -----------------------------
def collect_documents(paths: list[Path]):
    docs = []
    all_projects = []

    for p in paths:
        files = list(p.rglob("*")) if p.is_dir() else [p]

        for fp in files:
            ext = fp.suffix.lower()

            if ext == ".pdf":
                text = read_pdf_text(fp)
                for i, chunk in enumerate(chunk_text(text)):
                    docs.append(Document(page_content=chunk, metadata={"source": str(fp)}))

            elif ext in [".html", ".htm"]:
                text, projects = read_html_text(fp)
                all_projects.extend(projects)

                for i, chunk in enumerate(chunk_text(text)):
                    docs.append(Document(page_content=chunk, metadata={"source": str(fp)}))

                for proj in projects:
                    blob = f"Project: {proj['title']}\n{proj['description']}\n{proj['href']}"
                    docs.append(Document(page_content=blob, metadata={"source": str(fp)}))

            elif ext == ".txt":
                raw = fp.read_text(encoding="utf-8", errors="ignore")
                for i, chunk in enumerate(chunk_text(raw)):
                    docs.append(Document(page_content=chunk, metadata={"source": str(fp)}))

    return docs


# -----------------------------
# Build FAISS
# -----------------------------
def build_or_load_index(index_dir: Path, inputs: list[Path]):
    embed = OpenAIEmbeddings(model="text-embedding-3-large")

    if (index_dir / "index.faiss").exists():
        print("[OK] Loading index")
        return FAISS.load_local(str(index_dir), embed, allow_dangerous_deserialization=True)

    print("[*] Building FAISS index...")

    docs = collect_documents(inputs)
    store = FAISS.from_documents(docs, embed)

    index_dir.mkdir(exist_ok=True)
    store.save_local(str(index_dir))

    print("[OK] Index saved")
    return store


# -----------------------------
# Build RAG chain (Modern LCEL)
# -----------------------------
def make_chain(llm, retriever):
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are Kishan Yadav’s portfolio assistant.\n"
         "Use the following context to answer the user's question:\n\n"
         "{context}\n\n"
         "Guidelines:\n"
         "- Only answer using Kishan’s resume, skills, certifications, and projects.\n"
         "- If the answer is not in the context, say: 'I don’t have that detail in my portfolio yet.'"),
        ("human", "{question}")
    ])

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return chain



# -----------------------------
# CLI Commands
# -----------------------------
def cmd_index(args):
    paths = [Path(x) for x in args.inputs]
    index_dir = Path(args.index_dir)

    if args.reset and index_dir.exists():
        shutil.rmtree(index_dir)

    build_or_load_index(index_dir, paths)


def cmd_chat(args):
    llm = configure_openai()

    index_dir = Path(args.index_dir)
    paths = [Path(x) for x in args.inputs] if args.inputs else []

    store = build_or_load_index(index_dir, paths) if paths else FAISS.load_local(
        str(index_dir),
        OpenAIEmbeddings(model="text-embedding-3-large"),
        allow_dangerous_deserialization=True
    )

    retriever = store.as_retriever(search_kwargs={"k": 6})
    chain = make_chain(llm, retriever)

    print("\nKishan RAG Assistant (OpenAI)")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in ("exit", "quit"):
            break

        out = chain.invoke(q)
        print("\nBot:", out.content)


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("index")
    p1.add_argument("inputs", nargs="+")
    p1.add_argument("--index-dir", default="rag_index")
    p1.add_argument("--reset", action="store_true")
    p1.set_defaults(func=cmd_index)

    p2 = sub.add_parser("chat")
    p2.add_argument("--inputs", nargs="*")
    p2.add_argument("--index-dir", default="rag_index")
    p2.set_defaults(func=cmd_chat)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
