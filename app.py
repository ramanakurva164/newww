import streamlit as st
from rag_utils import load_vectorstore, retrieve, chunk_text, get_embedder, build_vectorstore
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
from PyPDF2 import PdfReader
from supabase import create_client, Client
import json

# --- Supabase Client and Authentication Functions ---
try:
    supabase: Client = create_client(st.secrets["SUPABASE_URL"], st.secrets["SUPABASE_KEY"])
except KeyError as e:
    st.error(f"Missing Streamlit secret: {e}. Please ensure SUPABASE_URL and SUPABASE_KEY are in .streamlit/secrets.toml")
    st.stop()

def sign_in(email, password):
    """Signs in a user with email and password."""
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if response.user:
            st.session_state["user"] = response.user
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Login failed. Please check your credentials.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def sign_up(email, password):
    """Signs up a new user with email and password."""
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            st.success("Account created successfully! Please check your email for a confirmation link.")
        else:
            st.error("Sign-up failed.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

def logout():
    """Logs out the user."""
    try:
        supabase.auth.sign_out()
        st.session_state["logged_in"] = False
        st.session_state["user"] = None
        st.rerun()
    except Exception as e:
        st.error(f"Logout failed: {e}")

# --- Streamlit Page Configuration and UI Styling ---
st.set_page_config(page_title="RAG Chatbot", page_icon="📘", layout="wide")

st.markdown("""
    <style>
    /* Main container and background */
    .stApp { background-color: #1a1a1a; color: #f5f5f5; font-family: 'Segoe UI', sans-serif; }
    /* Fix for some streamlit components */
    .st-emotion-cache-1ky2r8c, .st-emotion-cache-163f9l9 { background-color: #1a1a1a; }
    /* Input box style */
    .stTextInput > div > div > input { background-color: #2b2b2b; border: 2px solid #f97316; border-radius: 10px; padding: 10px; font-size: 16px; color: #f5f5f5; }
    /* Button styles */
    .stButton>button { background-color: #f97316; color: #1a1a1a; border-radius: 12px; padding: 12px 24px; font-weight: 600; font-size: 15px; box-shadow: 0px 4px 10px rgba(249, 115, 22, 0.3); transition: 0.2s ease-in-out; }
    .stButton>button:hover { background-color: #ea580c; transform: scale(1.05); }
    /* User and bot chat bubbles */
    .chat-bubble-user { background: #1f2937; border-left: 4px solid #f97316; padding: 12px; border-radius: 12px; margin: 8px 0; font-size: 15px; line-height: 1.5; color: #f5f5f5; box-shadow: 0px 2px 6px rgba(0,0,0,0.5); }
    .chat-bubble-bot { background: #2b2b2b; border-left: 4px solid #3b82f6; padding: 12px; border-radius: 12px; margin: 8px 0; font-size: 15px; line-height: 1.5; color: #f5f5f5; box-shadow: 0px 2px 6px rgba(0,0,0,0.5); }
    /* Sidebar header */
    .st-emotion-cache-1nj0h6l { border-bottom: 1px solid #333; }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State for Authentication ---
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "user" not in st.session_state:
    st.session_state["user"] = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Conditional Rendering based on Login Status ---
if not st.session_state["logged_in"]:
    # Display the login/signup form
    st.title("Login to your RAG Chatbot")
    st.write("Or create an account below.")
    email = st.text_input("Email")
    password = st.text_input("Password", type="password")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Login"):
            if email and password:
                sign_in(email, password)
            else:
                st.warning("Please enter both email and password.")
    with col2:
        if st.button("Sign Up"):
            if email and password:
                sign_up(email, password)
            else:
                st.warning("Please enter both email and password.")
else:
    # --- Load LLM ---
    @st.cache_resource
    def load_model():
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    qa_pipeline = load_model()

    # --- Main App Logic for Authenticated Users ---
    st.title("📘 Chat with your documents like never before.")
    
    # --- Sidebar for functionality and profile ---
    with st.sidebar:
        st.header("👤 My Account")
        st.write(f"Logged in as: **{st.session_state.user.email}**")
        
        # Display user profile data from Supabase
        user_id = st.session_state.user.id
        try:
            response = supabase.from("profiles").select("*").eq("id", user_id).single().execute()
            if response.data:
                profile_data = response.data
                st.write("---")
                st.subheader("Profile Details")
                st.json(json.dumps(profile_data, indent=2))
        except Exception as e:
            st.error(f"Could not fetch profile data: {e}")

        # Logout button
        if st.button("Logout"):
            logout()
            
        st.write("---")
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
            
            # We process and use the uploaded files
            with st.spinner("Indexing uploaded documents..."):
                chunks = chunk_text(texts)
                embedder = get_embedder()
                embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
                active_index = faiss.IndexFlatL2(embeddings.shape[1])
                active_index.add(embeddings)
                st.session_state.active_index = active_index
                st.session_state.active_chunks = chunks
            st.success("✅ Uploaded and indexed documents!")

        if st.button("New Chat"):
            st.session_state.messages = []
            st.rerun()
    
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
            # Display context if available
            if "context" in msg:
                with st.expander("📖 Retrieved Context"):
                    for ctx_chunk in msg["context"]:
                        st.write(ctx_chunk)
    
    # Handle user input
    prompt = st.chat_input("Ask a question from your documents...")
    if prompt:
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='chat-bubble-user'><b>You:</b> {prompt}</div>", unsafe_allow_html=True)

        with st.spinner("Generating answer..."):
            # Retrieve context from the active vectorstore
            context = retrieve(prompt, st.session_state.active_index, st.session_state.active_chunks)
            combined_context = " ".join(context)
            
            # Build prompt for LLM
            llm_prompt = f"Context: {combined_context}\n\nQuestion: {prompt}\nAnswer:"
            response = qa_pipeline(llm_prompt, max_new_tokens=200)[0]["generated_text"]

            # Add bot message and context to chat history
            st.session_state.messages.append({"role": "assistant", "content": response, "context": context})

            # Display bot response
            st.markdown(f"<div class='chat-bubble-bot'><b>Bot:</b> {response}</div>", unsafe_allow_html=True)
            
            # Display retrieved context in an expandable section
            if context:
                with st.expander("📖 Retrieved Context"):
                    for ctx_chunk in context:
                        st.write(ctx_chunk)
