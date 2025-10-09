# AGENT CHATBOT (with Database + FAISS + Gemini + Multi-modal RAG)
import os
import tempfile
import datetime
import pickle
import re
import json
import sqlite3  # ✅ added for database
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

# -------------------- DATABASE SETUP --------------------
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
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO chats (role, content, time, sources, images) VALUES (?, ?, ?, ?, ?)",
        (role, content, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), str(sources), str(images))
    )
    conn.commit()
    conn.close()

def insert_document(filename, filetype):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO documents (filename, filetype, uploaded_at) VALUES (?, ?, ?)",
        (filename, filetype, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    )
    conn.commit()
    conn.close()

def get_recent_chats(limit=10):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT role, content, time FROM chats ORDER BY id DESC LIMIT ?", (limit,))
    rows = c.fetchall()
    conn.close()
    return rows[::-1]

# Initialize DB on startup
init_db()
# --------------------------------------------------------

# Configure device for GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Configure Gemini API
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception as e:
    st.error(f"Secret loading failed: {e}")

chat_model = genai.GenerativeModel("models/gemini-2.5-pro")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device=device)

# Custom LLM wrapper for Gemini
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

# Paths for FAISS indices and memory stores
TEXT_FAISS_PATH = "faiss_store/text_index.pkl"
IMAGE_FAISS_PATH = "faiss_store/image_index.pkl"
DISTILL_FAISS_PATH = "faiss_store/distill_index.pkl"
os.makedirs("faiss_store", exist_ok=True)

# Extract text from PDF files
def extract_text_from_pdfs(pdf_files):
    docs = []
    for pdf_file in pdf_files:
        fname = getattr(pdf_file, "name", None) or "uploaded.pdf"
        insert_document(fname, "PDF")  # ✅ store in DB
        try:
            pdf_reader = PdfReader(pdf_file)
        except Exception as e:
            st.error(f"Failed to read PDF {fname}: {e}")
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpf:
                tmpf.write(pdf_file.read())
                tmpf.flush()
                pdf_reader = PdfReader(tmpf.name)
                fname = tmpf.name
        for i, page in enumerate(pdf_reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(Document(page_content=text, metadata={"source": fname, "page": i + 1}))
    return docs

# Extract text from Excel files
def extract_text_from_excel(excel_file):
    docs = []
    insert_document(excel_file.name, "Excel")  # ✅ store in DB
    try:
        xls = pd.ExcelFile(excel_file)
        for sheet_name in xls.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            for i, row in df.iterrows():
                row_text = " | ".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notnull(row[col])])
                if row_text.strip():
                    docs.append(Document(page_content=row_text, metadata={"source": excel_file.name, "sheet": sheet_name, "row": i+1}))
    except Exception as e:
        st.error(f"Failed to process Excel file {excel_file.name}: {e}")
    return docs

# Extract text from CSV files
def extract_text_from_csv(csv_file):
    docs = []
    insert_document(csv_file.name, "CSV")  # ✅ store in DB
    try:
        df = pd.read_csv(csv_file)
        for i, row in df.iterrows():
            row_text = " | ".join([f"{col}: {str(row[col])}" for col in df.columns if pd.notnull(row[col])])
            if row_text.strip():
                docs.append(Document(page_content=row_text, metadata={"source": csv_file.name, "row": i+1}))
    except Exception as e:
        st.error(f"Error reading CSV file {csv_file.name}: {e}")
    return docs

# Generate text embeddings
def embed_text(texts):
    return embedding_model.encode(texts, convert_to_numpy=True)

# Initialize CLIP model for image embeddings
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Embed images using CLIP
def embed_image(image_file):
    insert_document(getattr(image_file, "name", "image.png"), "Image")  # ✅ store in DB
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

# Image retriever using FAISS for similarity search
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
        if not self.paths or self.embeddings is None or self.embeddings.ntotal == 0:
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

# Initialize text FAISS index, BM25, and distillation memory
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

# Save FAISS index to disk
def save_text_faiss(index, docs, path=TEXT_FAISS_PATH):
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump({"index": index, "docs": docs}, f)
    os.replace(tmp, path)

# RAG agent with streaming response and multi-modal search
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
        print(f"DEBUG: Distillation search - distill_index: {distill_index}, distill_docs length: {len(distill_docs)}")
        if distill_index and distill_index.ntotal > 0:
            query_emb = embed_text([query_text]).astype("float32")
            top_k = min(3, distill_index.ntotal)
            D, I = distill_index.search(query_emb, top_k)
            docs.extend([distill_docs[i] for i in I[0]])
            print(f"DEBUG: Retrieved {len(docs)} distilled documents")
        else:
            print("DEBUG: No distillation index or empty, skipping search")
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

# Initialize Streamlit session state
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

# Format assistant responses for display
def format_answer(answer: str) -> str:
    answer = re.sub(r"(\d+)\.\s+", r"\n\1. ", answer)
    answer = re.sub(r"[-•]\s+", r"\n- ", answer)
    return answer.strip()

# Configure Streamlit UI
st.set_page_config(page_title="TEASER Agent", layout="wide", page_icon="🤖")
st.markdown("<h2 style='text-align:center;'> 🤖 Meet TEASER</h2>", unsafe_allow_html=True)
st.markdown("<div class='chat-container'>", unsafe_allow_html=True)

st.markdown("""
<style>
.chat-container { max-width:900px; margin:auto; overflow-y:auto; max-height:75vh; padding-bottom:100px; }
.chat-row { display:flex; align-items:flex-start; margin:6px 0; }
.chat-avatar { font-size:28px; margin:6px; }
.chat-bubble { padding:10px 15px; border-radius:18px; max-width:75%; word-wrap:break-word; font-size:15px; line-height:1.5; }
.user-bubble { max-width:400px; margin-left:auto; text-align:right; background-color:#333; color:#fff; }
.assistant-bubble { margin-right:auto; text-align:left; background-color:transparent; color:#fff; white-space:pre-wrap; }
.timestamp { font-size:11px; color:#aaa; margin:2px 12px; text-align:right; }
.sidebar .block-container { background:#1e1e1e; color:white; }
.stFileUploader { max-width: 50% !important; margin:0 auto 20px auto; }
.copy-btn { font-size:12px; color:#bbb; cursor:pointer; margin-top:4px; }
.copy-btn:hover { color: #fff; text-decoration: underline; }
.loading-dots span { animation: dots 1.4s infinite; display: inline-block; }
.loading-dots span:nth-child(2) { animation-delay: 0.2s; }
.loading-dots span:nth-child(3) { animation-delay: 0.4s; }
@keyframes dots { 0%, 20% { opacity: 0; } 50% { opacity: 1; } 100% { opacity: 0; } }
</style>
""", unsafe_allow_html=True)

# Inject JavaScript for clipboard functionality with fallback
html("""
<script>
function copyToClipboard(elementId) {
    try {
        const element = document.getElementById(elementId);
        if (!element) {
            console.error("Element with ID " + elementId + " not found");
            alert("Error: Could not find text to copy");
            return;
        }
        const text = element.innerText;
        if (navigator.clipboard) {
            navigator.clipboard.writeText(text).then(
                () => alert("Copied to clipboard!"),
                (err) => {
                    console.error("Clipboard write failed: ", err);
                    fallbackCopy(text);
                }
            );
        } else {
            fallbackCopy(text);
        }
    } catch (err) {
        console.error("Error in copyToClipboard: ", err);
        alert("Error copying text: " + err);
    }
    function fallbackCopy(text) {
        const textArea = document.createElement("textarea");
        textArea.value = text;
        document.body.appendChild(textArea);
        textArea.select();
        try {
            document.execCommand('copy');
            alert("Copied to clipboard (fallback method)!");
        } catch (err) {
            alert("Failed to copy: " + err);
        }
        document.body.removeChild(textArea);
    }
}
</script>
""", height=0)

# File upload section for PDFs, Excel, CSV, and images
uploaded_files = st.file_uploader(
    "Upload",
    type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls", "csv"],
    accept_multiple_files=True,
)

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
            if file.type == "application/pdf":
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
                print(f"DEBUG: Embedding {len(new_text_docs)} text documents")
                new_text_embeddings = embed_text([d.page_content for d in new_text_docs]).astype("float32")
                if text_index is None:
                    dim = new_text_embeddings.shape[1]
                    text_index = faiss.IndexFlatL2(dim)
                text_index.add(new_text_embeddings)
                text_docs.extend(new_text_docs)
                bm25 = BM25Okapi([d.page_content.split() for d in text_docs])
                save_text_faiss(text_index, text_docs)
                print(f"DEBUG: Text index updated with {len(new_text_docs)} documents")
        except Exception as e:
            st.error(f"Error updating text index: {e}")
            print(f"DEBUG: Error updating text index: {e}")

    if new_image_paths:
        try:
            with st.spinner("Embedding images..."):
                print(f"DEBUG: Embedding {len(new_image_paths)} images")
                image_retriever.add_images(new_image_paths, new_image_embeddings)
                print(f"DEBUG: Image index updated with {len(new_image_paths)} images")
        except Exception as e:
            st.error(f"Error updating image index: {e}")
            print(f"DEBUG: Error updating image index: {e}")

    progress_bar.empty()
    status_text.empty()
    st.success(f"Processed {len(uploaded_files)} files and updated the vector database!")

# Handle chat input and "list files" query
user_query = st.chat_input("Ask your query here...")
if user_query:
    current_time = datetime.datetime.now().strftime("%H:%M")
    st.session_state.messages.append({"role": "user", "content": user_query, "time": current_time})
    st.session_state.working_memory.append({"role": "user", "content": user_query, "time": current_time})
    st.session_state.transactional_memory.append({"role": "user", "content": user_query, "time": current_time})

    st.markdown(f"""
    <div class='chat-row' style='justify-content:flex-end;'>
      <div>
        <div class='chat-bubble user-bubble'>{escape(user_query)}</div>
        <div class='timestamp'>{current_time}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    thinking_placeholder = st.empty()
    thinking_placeholder.markdown(f"""
    <div class='chat-row' style='justify-content:flex-start;'>
      <div class='chat-avatar'>🤖</div>
      <div>
        <div class='chat-bubble assistant-bubble'>
          <span class='loading-dots'><span>.</span><span>.</span><span>.</span></span>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

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
    else:
        print(f"DEBUG: Processing query: {user_query}")
        response, images, sources = rag_chat_stream_agentic(user_query, use_images=True)
        buffer_text = []
        live_placeholder = st.empty()
        for chunk in response:
            text_piece = getattr(chunk, "text", None)
            if not text_piece and hasattr(chunk, "candidates") and chunk.candidates:
                text_piece = "".join([c.get("content","") or c.get("text","") for c in chunk.candidates])
            if text_piece:
                buffer_text.append(text_piece)
                current_stream = "".join(buffer_text)
                live_placeholder.markdown(f"""
                    <div class='chat-row' style='justify-content:flex-start;'>
                      <div class='chat-avatar'>🤖</div>
                      <div class='chat-bubble assistant-bubble'>{escape(current_stream)}</div>
                    </div>""", unsafe_allow_html=True)
        formatted_answer = format_answer("".join(buffer_text))
        live_placeholder.empty()

    thinking_placeholder.empty()

    source_lines = []
    if sources:
        for d in sources:
            src = d.metadata.get("source", "uploaded.pdf")
            page = d.metadata.get("page", d.metadata.get("row", "?"))
            source_lines.append(f"Source: {os.path.basename(src)} : {page}")
    if images:
        for p in images:
            source_lines.append(f"Image: {os.path.basename(p)}")
    if source_lines:
        formatted_answer += "\n\n" + "\n".join(source_lines)

    st.session_state.messages.append({"role":"assistant","content":formatted_answer,"images":images,"sources":sources,"time":current_time})
    st.session_state.working_memory.append({"role":"assistant","content":formatted_answer,"time":current_time})
    st.session_state.transactional_memory.append({"role":"assistant","content":formatted_answer,"time":current_time})

    if st.session_state.transactional_memory:
        distill_text = "\n".join([m["content"] for m in st.session_state.transactional_memory])
        try:
            distill_emb = embed_text([distill_text]).astype("float32")
            if distill_index is None:
                dim = distill_emb.shape[1]
                distill_index = faiss.IndexFlatL2(dim)
            distill_index.add(distill_emb)
            distill_docs.append(Document(page_content=distill_text, metadata={"source":"distillation","time":current_time}))
            with open(DISTILL_FAISS_PATH + ".tmp","wb") as f:
                pickle.dump({"index":distill_index,"docs":distill_docs}, f)
            os.replace(DISTILL_FAISS_PATH + ".tmp", DISTILL_FAISS_PATH)
            st.session_state.distillation_memory.append({"role":"distilled","content":distill_text,"time":current_time})
        except Exception as e:
            st.error(f"Error updating distillation memory: {e}")
            print(f"DEBUG: Error updating distillation memory: {e}")

    msg_id = f"assistant_{len(st.session_state.messages)-1}"
    st.markdown(f"""
    <div class='chat-row' style='justify-content:flex-start;'>
      <div class='chat-avatar'>🤖</div>
      <div>
        <div id='{msg_id}' class='chat-bubble assistant-bubble'>{formatted_answer}</div>
        <div class='timestamp'>{current_time}</div>
        <div class='copy-btn' onclick='copyToClipboard("{msg_id}")'>📋 Copy</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    if images:
        for img in images:
            try:
                st.image(img, caption="Relevant Image", use_container_width=True)
            except Exception:
                pass

st.markdown("</div>", unsafe_allow_html=True)
