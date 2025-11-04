# 🧠 Agent-Teaser  
### *An Autonomous Multi-Modal RAG Chatbot with FAISS, Gemini AI & SQLite Database*  

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10+-blue?logo=python" />
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" />
  <img src="https://img.shields.io/badge/Database-SQLite-lightgrey?logo=sqlite" />
  <img src="https://img.shields.io/badge/Retrieval-FAISS-green?logo=facebook" />
  <img src="https://img.shields.io/badge/LLM-Gemini%202.5%20Pro-yellow?logo=google" />
  <img src="https://img.shields.io/badge/License-MIT-lightgrey" />
</p>

> 🚀 **Agent-Teaser** is an AI-powered assistant that understands and retrieves information from **PDFs**, **Excel**, **CSV**, and **Images**, powered by **FAISS vector search**, **Gemini 2.5 Pro**, and a persistent **SQLite** database.

---

## ✨ Key Features  

✅ **Multi-Modal RAG Support** — Handle both text 📄 and image 🖼️ data.  
✅ **FAISS-Based Vector Retrieval** — Fast and scalable semantic search.  
✅ **Gemini 2.5 Pro Integration** — State-of-the-art generative AI reasoning.  
✅ **SQLite Chat & File History** — Every user query and response is logged.  
✅ **Persistent Chat Memory** — Maintains short-term and distilled memory.  
✅ **Streaming Responses** — Smooth real-time typing effect.  
✅ **Chat History Sidebar** — Timestamped session logs per user.  
✅ **File Upload Support** — PDFs, Excel, CSV, and Images supported.  
✅ **Modern Streamlit UI** — Intuitive, responsive chat interface.  

---

## 🧩 Tech Stack  

| Layer | Technology |
|--------|-------------|
| 🖥️ Frontend | Streamlit |
| ⚙️ Backend | Python, LangChain, Gemini API |
| 🗄️ Database | SQLite |
| 🔍 Vector Search | FAISS |
| 🧠 Embeddings | SentenceTransformer & CLIP |
| 💬 Language Model | Google Gemini 2.5 Pro |

---

## 🗂️ Project Structure  

```bash
Agent-Teaser/
│
├── multi_modal_rag.py          # Main app logic (retrieval + chat)
├── requirements.txt             # Dependencies list
├── .streamlit/
│   └── secrets.toml             # API keys and config
├── faiss_store/                 # Vector index storage
├── agent_chatbot.db             # SQLite chat database
└── README.md                    # Project documentation
