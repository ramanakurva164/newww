import streamlit as st
from rag_utils import load_vectorstore, retrieve, chunk_text, get_embedder
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
from PyPDF2 import PdfReader

st.set_page_config(page_title="RAG Chatbot", page_icon="📘", layout="wide")

# UI Custom Styling (Blue-White Theme)
st.markdown("""
    <style>
    /* App background */
    .stApp { 
        background-color: #1a1a1a; 
        color: #f5f5f5;
        font-family: 'Segoe UI', sans-serif;
    }

    /* Input box */
    .stTextInput > div > div > input {
        background-color: #2b2b2b; 
        border: 2px solid #f97316;  /* orange accent */
        border-radius: 10px; 
        padding: 10px;
        font-size: 16px;
        color: #f5f5f5;
    }

    /* Buttons */
    .stButton>button {
        background-color: #f97316;  /* orange */
        color: #1a1a1a; 
        border-radius: 12px;
        padding: 12px 24px; 
        font-weight: 600;
        font-size: 15px;
        box-shadow: 0px 4px 10px rgba(249, 115, 22, 0.3);
        transition: 0.2s ease-in-out;
    }

    .stButton>button:hover { 
        background-color: #ea580c;  /* darker orange */
        transform: scale(1.05);
    }

    /* Chat bubble */
    .chat-bubble {
        background: #2b2b2b; 
        border: 2px solid #f97316;
        padding: 12px; 
        border-radius: 12px; 
        margin: 8px 0;
        font-size: 15px;
        line-height: 1.5;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.5);
        color: #f5f5f5;
    }
</style>
""", unsafe_allow_html=True)

st.title("📘 RAG-Based Document Question Answering Chatbot")

# ---- File Upload Section ----
uploaded_files = st.file_uploader(
    "📂 Upload your documents (PDF only)", 
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

    # Chunk and embed
    chunks = chunk_text(texts)
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    active_index, active_chunks = index, chunks
    st.success("✅ Uploaded and indexed documents!")

else:
    # Fall back to preloaded vectorstore
    with st.spinner("Loading prebuilt vectorstore..."):
        active_index, active_chunks = load_vectorstore()


# ---- Load LLM ----
@st.cache_resource
def load_model():
    model_name = "google/flan-t5-base"  # CPU-friendly, no HF token required
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
    return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

qa_pipeline = load_model()


# ---- Query UI ----
query = st.text_input("🔎 Ask a question from your documents:")

col1, col2 = st.columns(2)
with col1:
    ask = st.button("Get Answer")
with col2:
    rebuild = st.button("Rebuild Index from ./docs folder")

if rebuild:
    from rag_utils import build_vectorstore
    with st.spinner("Rebuilding vectorstore from PDFs in ./docs ..."):
        active_index, active_chunks = build_vectorstore()
    st.success("✅ Rebuilt vectorstore.")


# ---- Answer Generation ----
if ask and query.strip():
    context = retrieve(query, active_index, active_chunks)
    combined_context = " ".join(context)

    prompt = f"Context: {combined_context}\n\nQuestion: {query}\nAnswer:"
    response = qa_pipeline(prompt, max_new_tokens=200)[0]["generated_text"]

    st.markdown(f"<div class='chat-bubble'><b>Answer:</b> {response}</div>", unsafe_allow_html=True)

    with st.expander("📖 Retrieved Context"):
        st.write(context)