🧠 Agent-Teaser
An Autonomous Multi-Modal RAG Chatbot with FAISS, Gemini AI & SQLite Database

Agent-Teaser is an AI-powered assistant capable of understanding and retrieving information from PDFs, Excel, CSV, and Images, augmented with FAISS-based vector search and Gemini Generative AI.
It also features an interactive chat UI with history persistence, streaming responses, and database logging for all user-bot interactions.

🚀 Features

✅ Multi-modal RAG support (Text 📄 + Image 🖼️)

✅ FAISS vector search for retrieval

✅ Gemini 2.5 Pro model integration

✅ SQLite database for chat & file history

✅ Persistent chat memory (working + distillation memory)

✅ Smooth streaming (typing) responses

✅ Chat history sidebar with user timestamps

✅ File upload support for PDFs, Excel, CSV, and images

✅ Modern Streamlit chat interface


🧩 Tech Stack

Frontend: Streamlit

Backend: Python (LangChain, Gemini API)

Database: SQLite

Vector Search: FAISS

Embeddings: SentenceTransformer & CLIP

Language Model: Google Gemini 2.5 Pro



🗂️ Project Structure

Agent-Teaser/
│
├── multi_modal_rag.py          
├── requirements.txt             
├── .streamlit/
│   └── secrets.toml             
├── faiss_store/                 
├── agent_chatbot.db             
└── README.md                   



💾 Database Schema

| id | role | content | time | sources | images |
| -- | ---- | ------- | ---- | ------- | ------ |



💬 Example Queries

“Summarize the content of the uploaded PDF.”
“Find all rows in the Excel sheet related to renewable energy.”
“Show me the most relevant image for this topic.”
“List all uploaded files.”

🧠 Future Enhancements

🔐 Authentication system for user-based chat logs
🧰 Admin dashboard to manage history and documents
☁️ Cloud database integration (PostgreSQL or Firebase)
💡 Smart agent chaining (TEASER multi-agent architecture)

🧑‍💻 Author

Venkat Vatshal

📧 venkatvatshal@gmail.com
