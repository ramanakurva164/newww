import streamlit as st
from rag_utils import load_vectorstore, retrieve, chunk_text, get_embedder
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
from PyPDF2 import PdfReader

# --- Streamlit Page Configuration and UI Styling ---
st.set_page_config(page_title="RAG Chatbot", page_icon="📘", layout="wide")

st.markdown("""
    <style>
    .stApp { background-color: #1a1a1a; color: #f5f5f5; font-family: 'Segoe UI', sans-serif; }
    .st-emotion-cache-1ky2r8c, .st-emotion-cache-163f9l9 { background-color: #1a1a1a; }
    .stTextInput > div > div > input { background-color: #2b2b2b; border: 2px solid #f97316; border-radius: 10px; padding: 10px; font-size: 16px; color: #f5f5f5; }
    .stButton>button { background-color: #f97316; color: #1a1a1a; border-radius: 12px; padding: 12px 24px; font-weight: 600; font-size: 15px; box-shadow: 0px 4px 10px rgba(249, 115, 22, 0.3); transition: 0.2s ease-in-out; }
    .stButton>button:hover { background-color: #ea580c; transform: scale(1.05); }
    .chat-bubble-user { background: #1f2937; border-left: 4px solid #f97316; padding: 12px; border-radius: 12px; margin: 8px 0; font-size: 15px; line-height: 1.5; color: #f5f5f5; box-shadow: 0px 2px 6px rgba(0,0,0,0.5); }
    .chat-bubble-bot { background: #2b2b2b; border-left: 4px solid #3b82f6; padding: 12px; border-radius: 12px; margin: 8px 0; font-size: 15px; line-height: 1.5; color: #f5f5f5; box-shadow: 0px 2px 6px rgba(0,0,0,0.5); }
    .st-emotion-cache-1nj0h6l { border-bottom: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Load LLM ---
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

qa_pipeline = load_model()

# --- Main App Logic ---
st.title("📘 RAG-Based Document Question Answering Chatbot")

# --- Sidebar for functionality ---
with st.sidebar:
    st.header("📂 Document Management")
    uploaded_files = st.file_uploader(
        "Upload new documents (PDF only)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        texts = []
        for uploaded_file in uploaded_files:
            reader = PdfReader(uploaded_file)
            for page in reader.pages:
                txt = page.extract_text()
                if txt and txt.strip():
                    texts.append(txt.strip())

        with st.spinner("Indexing uploaded documents..."):
            chunks = chunk_text(texts)
            embedder = get_embedder()
            embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
            active_index = faiss.IndexFlatL2(embeddings.shape[1])
            active_index.add(embeddings)
            st.session_state.active_index = active_index
            st.session_state.active_chunks = chunks
        st.success("✅ Uploaded and indexed documents!")

    st.header("⚙️ Settings")
    
    col1_sb, col2_sb = st.columns(2)
    with col1_sb:
        if st.button("Rebuild Index"):
            from rag_utils import build_vectorstore
            with st.spinner("Rebuilding from ./docs folder..."):
                active_index, active_chunks = build_vectorstore()
                st.session_state.active_index = active_index
                st.session_state.active_chunks = active_chunks
            st.success("✅ Rebuilt vectorstore.")
    
    with col2_sb:
        if st.button("New Chat"):
            st.session_state.messages = []
            st.experimental_rerun()

# --- Load or Initialize Vectorstore ---
if "active_index" not in st.session_state or "active_chunks" not in st.session_state:
    with st.spinner("Loading prebuilt vectorstore..."):
        active_index, active_chunks = load_vectorstore()
        st.session_state.active_index = active_index
        st.session_state.active_chunks = active_chunks

# --- Main Chat Interface ---
# Display previous messages
for msg in st.session_state.messages:
    if msg["role"] == "user":
        st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {msg['content']}</div>", unsafe_allow_html=True)
    elif msg["role"] == "assistant":
        st.markdown(f"<div class='chat-bubble-bot'><b>Bot:</b> {msg['content']}</div>", unsafe_allow_html=True)

# Handle user input
prompt = st.chat_input("Ask a question from your documents...")
if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {prompt}</div>", unsafe_allow_html=True)

    with st.spinner("Generating answer..."):
        # Retrieve context from the active vectorstore
        context, indices = retrieve(prompt, st.session_state.active_index, st.session_state.active_chunks)
        combined_context = " ".join(context)
        
        # Build prompt for LLM
        llm_prompt = f"Context: {combined_context}\n\nQuestion: {prompt}\nAnswer:"
        response = qa_pipeline(llm_prompt, max_new_tokens=200)[0]["generated_text"]

        # Format answer with citations
        full_response = response
        
        # Add bot message and context to chat history
        st.session_state.messages.append({"role": "assistant", "content": full_response, "context": context})

        # Display bot response
        st.markdown(f"<div class='chat-bubble-bot'><b>Bot:</b> {full_response}</div>", unsafe_allow_html=True)

        # Display retrieved context in an expandable section
        with st.expander("📖 Retrieved Context"):
            st.write(context)
