import os
import tempfile
import datetime
import pickle
import re
import json
import sqlite3  # built-in
import time
import numpy as np
import pandas as pd
from PyPDF2 import PdfReader
from PIL import Image, UnidentifiedImageError
import faiss
import streamlit as st
from streamlit.components.v1 import html
from transformers import CLIPProcessor, CLIPModel
import torch
from sentence_transformers import SentenceTransformer
from html import escape
from langchain.docstore.document import Document
from rank_bm25 import BM25Okapi
from langchain.agents import Tool, initialize_agent
from langchain.llms.base import LLM
import google.generativeai as genai
from typing import Optional, List, Any

# =========================
# DATABASE SETUP (SQLite)
# =========================
DB_PATH = "agent_chatbot.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
    CREATE TABLE IF NOT EXISTS chats (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT,
        content TEXT,
        time TEXT,
        sources TEXT,
        images TEXT
    )""")
    c.execute("""
    CREATE TABLE IF NOT EXISTS documents (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        filename TEXT,
        filetype TEXT,
        uploaded_at TEXT
    )""")
    conn.commit()
    conn.close()

def insert_chat(role, content, sources=None, images=None):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO chats (role, content, time, sources, images) VALUES (?, ?, ?, ?, ?)",
            (role, content, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(sources), str(images))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB insert_chat error:", e)

def insert_document(filename, filetype):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute(
            "INSERT INTO documents (filename, filetype, uploaded_at) VALUES (?, ?, ?)",
            (filename, filetype, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print("DB insert_document error:", e)

def get_recent_chats(limit=50):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, role, content, time FROM chats ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    # return reversed so older first in the list
    return rows[::-1]

def get_all_chats_df(limit=None):
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT id, role, content, time, sources, images FROM chats ORDER BY id DESC"
    if limit:
        query = f"SELECT id, role, content, time, sources, images FROM chats ORDER BY id DESC LIMIT {int(limit)}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

# initialize DB
init_db()

# =========================
# Device + APIs + models
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Gemini configuration (expects gemini API key in st.secrets["GEMINI_API_KEY"])
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    # show error only inside streamlit app; printing also helps debugging
    try:
        st.error(f"Secret loading failed: {e}")
    except Exception:
        print("Secret loading failed:", e)

# default model names in your original file kept
chat_model = genai.GenerativeModel("models/gemini-2.5-pro")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Gemini wrapper for LangChain LLM usage (kept from your file)
class GeminiLLM(LLM):
    chat_model: Optional[Any] = None
    def __init__(self, model_name="models/gemini-2.5-pro", **kwargs):
        super().__init__(**kwargs)
        self.chat_model = genai.GenerativeModel(model_name)
    @property
    def _llm_type(self) -> str:
        return "gemini"
    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self.chat_model.generate_content(prompt)
        try:
            if hasattr(response, "candidates") and response.candidates:
                return response.candidates[0].get("content") or response.candidates[0].get("text", "")
            elif isinstance(response, str):
                return response
        except Exception:
            return str(response)

llm_agent = GeminiLLM()

# =========================
# FAISS / CLIP / embedding paths
# =========================
TEXT_FAISS_PATH = "faiss_store/text_index.pkl"
IMAGE_FAISS_PATH = "faiss_store/image_index.pkl"
DISTILL_FAISS_PATH = "faiss_store/distill_index.pkl"
os.makedirs("faiss_store", exist_ok=True)

# =========================
# File extraction & embedding helpers (preserved)
# =========================
def extract_text_from_pdfs(pdf_files):
    docs = []
    for pdf_file in pdf_files:
        fname = getattr(pdf_file, "name", None) or "uploaded.pdf"
        try:
            insert_document(fname, "PDF")
        except Exception:
            pass
        try:
            pdf_reader = PdfReader(pdf_file)
        except Exception as e:
            # fallback to temporary write
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
                    tmpf.write(pdf_file.read())
                    tmpf.flush()
                    pdf_reader = PdfReader(tmpf.name)
                    fname = tmpf.name
            except Exception as e2:
                st.error(f"Failed to read PDF {fname}: {e2}")
                continue
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": fname, "page": i + 1}))
    return docs

def extract_text_from_excel(excel_file):
    docs = []
    try:
        insert_document(excel_file.name, "Excel")
    except Exception:
        pass
    try:
        xls = pd.ExcelFile(excel_file)
        for sheet_name in xls.sheet_names:
            try:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
            except Exception:
                continue
            for i, row in df.iterrows():
                row_text = " | ".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notnull(row[col])])
                if row_text.strip():
                    docs.append(Document(page_content=row_text, metadata={"source": excel_file.name, "sheet": sheet_name, "row": i+1}))
    except Exception as e:
        st.error(f"Failed to process Excel file {getattr(excel_file,'name','excel')}: {e}")
    return docs

def extract_text_from_csv(csv_file):
    docs = []
    try:
        insert_document(getattr(csv_file, "name", "csv"), "CSV")
    except Exception:
        pass
    try:
        df = pd.read_csv(csv_file)
        for i, row in df.iterrows():
            row_text = " | ".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notnull(row[col])])
            if row_text.strip():
                docs.append(Document(page_content=row_text, metadata={"source": csv_file.name, "row": i+1}))
    except Exception as e:
        st.error(f"Error reading CSV file {getattr(csv_file,'name','csv')}: {e}")
    return docs

def embed_text(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)

# CLIP setup
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def embed_image(image_file):
    try:
        insert_document(getattr(image_file, "name", "image.png"), "Image")
    except Exception:
        pass
    if hasattr(image_file, "read"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            tmp.write(image_file.read())
            tmp.flush()
            path = tmp.name
    else:
        path = image_file
    try:
        image = Image.open(path).convert("RGB")
        inputs = clip_processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = clip_model.get_image_features(**inputs).squeeze().cpu().numpy()
        return path, emb
    except UnidentifiedImageError:
        st.error(f"Skipped invalid image file: {getattr(image_file,'name', path)}")
        return None, None
    finally:
        if hasattr(image_file, "read") and os.path.exists(path):
            os.remove(path)

# ImageRetriever class (preserved)
class ImageRetriever:
    def __init__(self):
        self.paths = []
        self.embeddings = None
    def add_images(self, new_paths, new_embeddings):
        valid_paths = []
        valid_embs = []
        for p, e in zip(new_paths, new_embeddings):
            if p is not None and e is not None:
                valid_paths.append(p)
                valid_embs.append(e)
        if not valid_paths or not valid_embs:
            return
        self.paths.extend(valid_paths)
        arr = np.array(valid_embs).astype("float32")
        if self.embeddings is None:
            dim = arr.shape[1] if arr.ndim == 2 else arr.shape[0]
            self.embeddings = faiss.IndexFlatL2(dim)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        self.embeddings.add(arr)
        with open(IMAGE_FAISS_PATH + ".tmp", "wb") as f:
            pickle.dump({"paths": self.paths, "embeddings": self.embeddings}, f)
        os.replace(IMAGE_FAISS_PATH + ".tmp", IMAGE_FAISS_PATH)
    def get_relevant_images(self, query_text, top_k=5):
        if not self.paths or self.embeddings is None or getattr(self.embeddings, "ntotal", 0) == 0:
            return []
        inputs = clip_processor(text=query_text, return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            query_emb = clip_model.get_text_features(**inputs).cpu().numpy()
        if query_emb.ndim == 1:
            query_emb = query_emb.reshape(1, -1)
        if query_emb.shape[1] != self.embeddings.d:
            print(f"Skipped image search: dimension mismatch {query_emb.shape[1]} vs {self.embeddings.d}")
            return []
        k = min(top_k, self.embeddings.ntotal)
        if k <= 0:
            return []
        D, I = self.embeddings.search(query_emb.astype("float32"), k)
        indices = I[0].tolist()
        return [self.paths[i] for i in indices]

image_retriever = ImageRetriever()
if os.path.exists(IMAGE_FAISS_PATH):
    try:
        with open(IMAGE_FAISS_PATH, "rb") as f:
            data = pickle.load(f)
            image_retriever.paths = data.get("paths", [])
            image_retriever.embeddings = data.get("embeddings", None)
    except Exception:
        image_retriever = ImageRetriever()

# =========================
# Text FAISS + BM25 + Distill memory (preserved)
# =========================
text_index = None
text_docs = []
bm25 = None
distill_index = None
distill_docs = []

if os.path.exists(TEXT_FAISS_PATH):
    try:
        with open(TEXT_FAISS_PATH, "rb") as f:
            data = pickle.load(f)
            text_index = data.get("index")
            text_docs = data.get("docs", [])
            if text_docs:
                bm25 = BM25Okapi([d.page_content.split() for d in text_docs])
    except Exception:
        text_index, text_docs, bm25 = None, [], None

if os.path.exists(DISTILL_FAISS_PATH):
    try:
        with open(DISTILL_FAISS_PATH, "rb") as f:
            data = pickle.load(f)
            distill_index = data.get("index")
            distill_docs = data.get("docs", [])
    except Exception:
        distill_index, distill_docs = None, []

def save_text_faiss(index, docs, path=TEXT_FAISS_PATH):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"index": index, "docs": docs}, f)
    os.replace(tmp, path)

# =========================
# RAG agent (preserved) with same tools & retrieval
# =========================
def rag_chat_stream_agentic(query, use_images=True):
    def text_search_tool(query_text):
        global text_index, text_docs, bm25
        docs = []
        sources = []
        top_k = min(5, text_index.ntotal if text_index else 0)
        if text_index and top_k > 0:
            query_emb = embed_text([query_text]).astype("float32")
            D, I = text_index.search(query_emb, top_k)
            docs.extend([text_docs[i] for i in I[0]])
            sources.extend(docs)
        if bm25:
            tokenized_query = query_text.split()
            bm25_scores = bm25.get_scores(tokenized_query)
            top_bm25 = np.argsort(bm25_scores)[::-1][:top_k]
            docs.extend([text_docs[i] for i in top_bm25 if text_docs[i] not in docs])
        return docs, sources

    def image_search_tool(query_text):
        images = image_retriever.get_relevant_images(query_text, top_k=3) if use_images else []
        return images

    def distillation_search_tool(query_text):
        global distill_index, distill_docs
        docs = []
        if distill_index and getattr(distill_index, "ntotal", 0) > 0:
            query_emb = embed_text([query_text]).astype("float32")
            top_k = min(3, distill_index.ntotal)
            D, I = distill_index.search(query_emb, top_k)
            docs.extend([distill_docs[i] for i in I[0]])
        return docs

    tools = [
        Tool(
            name="Text Search",
            func=lambda q: text_search_tool(q)[0],
            description="Search text documents for relevant information."
        ),
        Tool(
            name="Image Search",
            func=image_search_tool,
            description="Search images for relevant visual information."
        ),
        Tool(
            name="Distillation Search",
            func=distillation_search_tool,
            description="Search distilled conversation history for relevant context."
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm_agent,
        agent_type="zero-shot-react-description",
        verbose=False
    )

    docs, sources = text_search_tool(query)
    images = image_search_tool(query) if use_images else []
    distill_docs_local = distillation_search_tool(query)

    context = ""
    if docs:
        context += "Text Documents:\n" + "\n".join([d.page_content for d in docs])
    if images:
        context += "\nRelevant Images:\n" + ", ".join([os.path.basename(p) for p in images])
    if distill_docs_local:
        context += "\nDistilled Conversation History:\n" + "\n".join([d.page_content for d in distill_docs_local])

    prompt = f"""
    You are Agent TEASER, a Retrieval-Augmented Generation assistant.
    Use the provided context to answer the query concisely and accurately.
    If no relevant context is found, respond based on your knowledge.
    Format the answer clearly, using bullet points or numbered lists where appropriate.
    Query: {query}
    Context: {context}
    """
    
    response = chat_model.generate_content(prompt, stream=True)
    return response, images, sources

# =========================
# Streamlit UI: session state and helpers
# =========================
if "uploaded_file_list" not in st.session_state:
    st.session_state["uploaded_file_list"] = []
if "messages" not in st.session_state:
    st.session_state.messages = []
if "working_memory" not in st.session_state:
    st.session_state.working_memory = []
if "transactional_memory" not in st.session_state:
    st.session_state.transactional_memory = []
if "distillation_memory" not in st.session_state:
    st.session_state.distillation_memory = []
if "show_db_viewer" not in st.session_state:
    st.session_state.show_db_viewer = False
if "db_view_limit" not in st.session_state:
    st.session_state.db_view_limit = 200

def format_answer(answer: str) -> str:
    answer = re.sub(r"(\d+)\.\s+", r"\n\1. ", answer)
    answer = re.sub(r"[-•]\s+", r"\n- ", answer)
    return answer.strip()

# =========================
# Streamlit layout setup
# =========================
st.set_page_config(page_title="TEASER Agent", layout="wide", page_icon="🤖")

# Load CSS and JS from static/ directory
with open("static/styles.css", "r") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
html('<script src="static/script.js"></script>', height=0)

# Load header from templates/ directory
with open("templates/header.html", "r") as f:
    st.markdown(f.read(), unsafe_allow_html=True)

# File uploader (handled in Python to maintain functionality)
uploaded_files = st.file_uploader(
    "Upload",
    type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls", "csv"],
    accept_multiple_files=True,
    key="uploader_main"
)

# =========================
# Sidebar: Chats + New Chat + DB button + DB history list
# =========================
with st.sidebar:
    st.markdown("<h3 style='color: white;'>💬 Chats</h3>", unsafe_allow_html=True)
    search_query = st.text_input("🔍 Search DB history", value="", placeholder="Search DB messages...", key="db_search")
    if st.button("➕ New Chat"):
        # clear in-memory session messages (but we keep DB)
        st.session_state.messages = []
        st.experimental_rerun()

    # Database viewer button (requirement 4)
    if st.button("🗄️ Database"):
        st.session_state.show_db_viewer = not st.session_state.show_db_viewer

    st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)

    # Display recent DB history (all chats from DB)
    st.markdown("**Recent database history**")
    db_rows = get_recent_chats(limit=200)  # returns (id, role, content, time)
    # filter by search_query if provided
    if search_query:
        db_rows = [r for r in db_rows if search_query.lower() in (r[2] or "").lower() or search_query.lower() in (r[1] or "").lower()]

    # Show entries with compact buttons to load into session
    for row in reversed(db_rows):  # newest first in sidebar
        rid, role, content, ts = row
        display_text = (content[:80] + "...") if len(content) > 80 else content
        if role == "user":
            label = f"👤 {ts} • {escape(display_text)}"
        else:
            label = f"🤖 {ts} • {escape(display_text)}"
        # Each chat entry button appends that DB message into the current session view
        if st.button(label, key=f"dbrow_{rid}"):
            # load this DB row into session view (append)
            st.session_state.messages.append({"role": role, "content": content, "time": ts})
            # also scroll to bottom in the main area by triggering rerun and letting rendered JS scroll
            st.experimental_rerun()

    st.markdown("<hr style='border:1px solid #333;'>", unsafe_allow_html=True)

    # DB viewer controls
    st.markdown("**DB Viewer**")
    st.session_state.db_view_limit = st.number_input("Rows to show", min_value=10, max_value=2000, value=200, step=10)
    if st.session_state.show_db_viewer:
        try:
            df = get_all_chats_df(limit=st.session_state.db_view_limit)
            # show a compact table
            st.dataframe(df, use_container_width=True)
            # allow export to CSV
            csv = df.to_csv(index=False)
            st.download_button("Download DB CSV", data=csv, file_name="agent_chat_history.csv")
        except Exception as e:
            st.error(f"Failed to load DB viewer: {e}")

# =========================
# Main: Chat container
# =========================
st.markdown("<div class='chat-container' id='chat_container'>", unsafe_allow_html=True)

# Render previously stored in-session messages (keeps stacking)
for i, msg in enumerate(st.session_state.messages):
    ts = msg.get("time", "")
    if msg["role"] == "user":
        safe_text = escape(msg['content'])
        st.markdown(f"""
        <div class='chat-row user-row' style='justify-content:flex-end;'>
          <div>
            <div class='chat-bubble user-bubble'>{safe_text}</div>
            <div class='timestamp'>{ts}</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # assistant message
        msg_id = f"assistant_render_{i}"
        safe_answer = base64_encode = None
        # render final assistant content immediately if already present
        safe_answer = escape(msg['content'])
        st.markdown(f"""
        <div class='chat-row assistant-row' style='justify-content:flex-start;'>
          <div class='chat-avatar'>🤖</div>
          <div>
            <div id='{msg_id}' class='chat-bubble assistant-bubble'>{safe_answer}</div>
            <div class='timestamp'>{ts}</div>
            <div class='copy-btn' onclick="copyToClipboard('{msg_id}')">📋 Copy</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        # show images if present
        if msg.get("images"):
            for img in msg["images"]:
                try:
                    st.image(img, caption=os.path.basename(img), use_container_width=True)
                except Exception:
                    pass

# target marker to scroll-to
st.markdown("<div id='chat_scroll_target'></div>", unsafe_allow_html=True)

# Force client scroll to bottom after rendering current messages
st.experimental_rerun_trigger = None
st.markdown("<script>setTimeout(scrollChatToBottom,50);</script>", unsafe_allow_html=True)

if uploaded_files:
    new_text_docs = []
    new_image_paths = []
    new_image_embeddings = []
    total_files = len(uploaded_files)
    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, file in enumerate(uploaded_files):
        status_text.text(f"Processing file {idx + 1}/{total_files}: {file.name}")
        print(f"DEBUG: Processing file {idx + 1}/{total_files}: {file.name}")
        if file.name in st.session_state["uploaded_file_list"]:
            print(f"DEBUG: File {file.name} already processed, skipping")
            continue
        st.session_state["uploaded_file_list"].append(file.name)

        try:
            if file.type == "application/pdf" or file.name.lower().endswith(".pdf"):
                new_text_docs.extend(extract_text_from_pdfs([file]))
                print(f"DEBUG: Processed PDF file {file.name}")
            elif file.name.lower().endswith((".xlsx", ".xls")):
                new_text_docs.extend(extract_text_from_excel(file))
                print(f"DEBUG: Processed Excel file {file.name}")
            elif file.name.lower().endswith(".csv"):
                new_text_docs.extend(extract_text_from_csv(file))
                print(f"DEBUG: Processed CSV file {file.name}")
            else:
                path, emb = embed_image(file)
                if path and emb is not None:
                    new_image_paths.append(path)
                    new_image_embeddings.append(emb)
                    print(f"DEBUG: Processed image file {file.name}")
        except Exception as e:
            st.error(f"Failed to process file {file.name}: {e}")
            print(f"DEBUG: Error processing file {file.name}: {e}")
            continue

        progress_bar.progress((idx + 1) / total_files)

    if new_text_docs:
        try:
            with st.spinner("Embedding text documents..."):
                new_text_embeddings = embed_text([d.page_content for d in new_text_docs]).astype("float32")
                if text_index is None:
                    dim = new_text_embeddings.shape[1]
                    text_index = faiss.IndexFlatL2(dim)
                # ensure shape matches expectation
                if new_text_embeddings.ndim == 1:
                    new_text_embeddings = new_text_embeddings.reshape(1, -1)
                text_index.add(new_text_embeddings)
                text_docs.extend(new_text_docs)
                bm25 = BM25Okapi([d.page_content.split() for d in text_docs])
                save_text_faiss(text_index, text_docs)
        except Exception as e:
            st.error(f"Error updating text index: {e}")
            print(f"DEBUG: Error updating text index: {e}")

    if new_image_paths:
        try:
            with st.spinner("Embedding images..."):
                image_retriever.add_images(new_image_paths, new_image_embeddings)
        except Exception as e:
            st.error(f"Error updating image index: {e}")
            print(f"DEBUG: Error updating image index: {e}")

    progress_bar.empty()
    status_text.empty()
    st.success(f"Processed {len(uploaded_files)} files and updated the vector database!")

# =========================
# Chat input + streaming behavior + DB logging
# =========================
user_query = st.chat_input("Ask your query here...", key="main_chat_input")
if user_query:
    # 1) Immediately insert user message into session_state and DB
    current_time = datetime.datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_query, "time": current_time})
    st.session_state.working_memory.append({"role": "user", "content": user_query, "time": current_time})
    st.session_state.transactional_memory.append({"role": "user", "content": user_query, "time": current_time})
    try:
        insert_chat("user", user_query)
    except Exception:
        pass

    # Render the user bubble (append below previous messages)
    st.markdown(f"""
    <div class='chat-row user-row' style='justify-content:flex-end;'>
      <div>
        <div class='chat-bubble user-bubble'>{escape(user_query)}</div>
        <div class='timestamp'>{current_time}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Show machine "thinking" placeholder
    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(f"""
    <div class='chat-row' style='justify-content:flex-start;'>
      <div class='chat-avatar'>🤖</div>
      <div>
        <div id='thinking_bubble' class='chat-bubble assistant-bubble'>
          <span class='loading-dots'><span>.</span><span>.</span><span>.</span></span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)
    # flush scroll
    st.markdown("<script>setTimeout(scrollChatToBottom,50);</script>", unsafe_allow_html=True)

    # Special "list files" handling
    if "list files" in user_query.lower():
        file_sources = set()
        for d in text_docs:
            src = d.metadata.get("source")
            if src:
                file_sources.add(src)
        file_sources.update([os.path.basename(p) for p in image_retriever.paths if p])
        files_list_text = "\n".join(sorted(file_sources)) if file_sources else "No files uploaded yet."
        formatted_answer = f"📄 Uploaded Files:\n{files_list_text}"
        sources = []
        images = []
        # log assistant to DB
        insert_chat("assistant", formatted_answer, sources, images)
        # append to session_state
        st.session_state.messages.append({"role":"assistant","content":formatted_answer,"images":images,"sources":sources,"time":current_time})

        # render assistant final
        msg_id = f"assistant_{len(st.session_state.messages)-1}"
        st.markdown(f"""
        <div class='chat-row assistant-row' style='justify-content:flex-start;'>
          <div class='chat-avatar'>🤖</div>
          <div>
            <div id='{msg_id}' class='chat-bubble assistant-bubble'>{escape(formatted_answer)}</div>
            <div class='timestamp'>{current_time}</div>
            <div class='copy-btn' onclick="copyToClipboard('{msg_id}')">📋 Copy</div>
          </div>
        </div>
        """, unsafe_allow_html=True)
        thinking_placeholder.empty()
        st.markdown("<script>setTimeout(scrollChatToBottom,50);</script>", unsafe_allow_html=True)
    else:
        # Normal RAG + streaming path
        response, images, sources = rag_chat_stream_agentic(user_query, use_images=True)
        # We'll stream chunks progressively into a placeholder element
        assistant_placeholder = st.empty()
        assistant_id = f"assistant_stream_{len(st.session_state.messages)}"
        # initialize placeholder with avatar + empty bubble
        assistant_placeholder.markdown(f"""
            <div class='chat-row assistant-row' style='justify-content:flex-start;'>
              <div class='chat-avatar'>🤖</div>
              <div>
                <div id='{assistant_id}' class='chat-bubble assistant-bubble'></div>
                <div class='timestamp'>{current_time}</div>
                <div class='copy-btn' onclick="copyToClipboard('{assistant_id}')">📋 Copy</div>
              </div>
            </div>
        """, unsafe_allow_html=True)
        # stream and update progressively
        collected_text = ""
        try:
            for chunk in response:
                # chunk may be object or dict; try several access patterns
                text_piece = None
                if hasattr(chunk, "text") and chunk.text:
                    text_piece = chunk.text
                elif isinstance(chunk, dict) and "text" in chunk:
                    text_piece = chunk.get("text")
                elif hasattr(chunk, "candidates") and getattr(chunk, "candidates"):
                    # join candidate pieces if present
                    try:
                        cand = chunk.candidates[0]
                        text_piece = cand.get("content") or cand.get("text") or ""
                    except Exception:
                        text_piece = None
                if not text_piece:
                    continue
                # append and update placeholder
                collected_text += text_piece
                # small throttle to improve UX but not block too long
                # update UI with escaped text to avoid HTML injection
                assistant_placeholder.markdown(f"""
                    <div class='chat-row assistant-row' style='justify-content:flex-start;'>
                      <div class='chat-avatar'>🤖</div>
                      <div>
                        <div id='{assistant_id}' class='chat-bubble assistant-bubble'>{escape(collected_text)}</div>
                        <div class='timestamp'>{current_time}</div>
                        <div class='copy-btn' onclick="copyToClipboard('{assistant_id}')">📋 Copy</div>
                      </div>
                    </div>
                """, unsafe_allow_html=True)
                # scroll little by little
                st.markdown("<script>setTimeout(scrollChatToBottom,50);</script>", unsafe_allow_html=True)
                # tiny sleep to make streaming readable - keep very short
                time.sleep(0.03)
        except Exception as e:
            print("Streaming error:", e)
            # attempt to fall back to non-streaming
            try:
                final_resp = chat_model.generate_content(f"Query: {user_query}", stream=False)
                collected_text += final_resp if isinstance(final_resp, str) else str(final_resp)
            except Exception:
                pass

        # finalize with source
        formatted_answer = format_answer(collected_text)
        source_info = ""
        if sources and sources[0]:
            if hasattr(sources[0], "metadata"):
                if "page" in sources[0].metadata:
                    source_info = f"\nSource: {sources[0].metadata['source']}, Page: {sources[0].metadata['page']}"
                elif "row" in sources[0].metadata:
                    sheet = sources[0].metadata.get("sheet", "")
                    row_info = f", Sheet: {sheet}" if sheet else ""
                    source_info = f"\nSource: {sources[0].metadata['source']}, Row: {sources[0].metadata['row']}{row_info}"
        elif images and images[0]:
            desc_response = chat_model.generate_content(["Describe this image briefly:", Image.open(images[0])])
            source_info = f"\nSource: {os.path.basename(images[0])}, Description: {desc_response.text}"
        formatted_answer += source_info

        assistant_placeholder.markdown(f"""
            <div class='chat-row assistant-row' style='justify-content:flex-start;'>
              <div class='chat-avatar'>🤖</div>
              <div>
                <div id='{assistant_id}' class='chat-bubble assistant-bubble'>{escape(formatted_answer)}</div>
                <div class='timestamp'>{current_time}</div>
                <div class='copy-btn' onclick="copyToClipboard('{assistant_id}')">📋 Copy</div>
              </div>
            </div>
        """, unsafe_allow_html=True)

        thinking_placeholder.empty()

        # show images retrieved inline
        if images:
            for img in images:
                try:
                    st.image(img, caption=os.path.basename(img), use_container_width=True)
                except Exception:
                    pass

        # log assistant and save to session and DB
        st.session_state.messages.append({"role":"assistant","content":formatted_answer,"images":images,"sources":sources,"time":current_time})
        st.session_state.working_memory.append({"role":"assistant","content":formatted_answer,"time":current_time})
        st.session_state.transactional_memory.append({"role":"assistant","content":formatted_answer,"time":current_time})
        try:
            insert_chat("assistant", formatted_answer, sources, images)
        except Exception:
            pass

        # update distillation memory as before
        if st.session_state.transactional_memory:
            distill_text = "\n".join([m["content"] for m in st.session_state.transactional_memory])
            try:
                distill_emb = embed_text([distill_text]).astype("float32")
                if distill_index is None:
                    dim = distill_emb.shape[1]
                    distill_index = faiss.IndexFlatL2(dim)
                if distill_emb.ndim == 1:
                    distill_emb = distill_emb.reshape(1, -1)
                distill_index.add(distill_emb)
                distill_docs.append(Document(page_content=distill_text, metadata={"source":"distillation","time":current_time}))
                with open(DISTILL_FAISS_PATH + ".tmp","wb") as f:
                    pickle.dump({"index":distill_index,"docs":distill_docs}, f)
                os.replace(DISTILL_FAISS_PATH + ".tmp", DISTILL_FAISS_PATH)
                st.session_state.distillation_memory.append({"role":"distilled","content":distill_text,"time":current_time})
            except Exception as e:
                st.error(f"Error updating distillation memory: {e}")
                print(f"DEBUG: Error updating distillation memory: {e}")

        # final scroll
        st.markdown("<script>setTimeout(scrollChatToBottom,50);</script>", unsafe_allow_html=True)

# close chat container
st.markdown("</div>", unsafe_allow_html=True)