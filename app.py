import streamlit as st
from rag_utils import load_vectorstore, retrieve, chunk_text, get_embedder
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import faiss
from PyPDF2 import PdfReader
from supabase import create_client, Client, AuthApiError

# --- Supabase Client Initialization ---
# Initialize the Supabase client. Using st.secrets to securely store credentials.
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)
except KeyError as e:
    st.error(f"Missing Streamlit secret: {e}. Ensure SUPABASE_URL and SUPABASE_KEY are in .streamlit/secrets.toml")
    st.stop()

# --- Authentication Functions ---
def sign_in(email, password):
    """Signs in a user with email and password using Supabase."""
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        st.session_state["user"] = response.user
        st.session_state["logged_in"] = True
        st.rerun()
    except AuthApiError as e:
        st.error(f"Login failed: {e.message}")
    except Exception as e:
        st.error(f"An unexpected error occurred during login: {e}")

def sign_up(email, password):
    """Signs up a new user with email and password using Supabase."""
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            st.success("Account created! Please check your email for a confirmation link.")
        else:
            st.error("Sign-up failed. The user may already exist or the password is too weak.")
    except AuthApiError as e:
        st.error(f"Sign-up failed: {e.message}")
    except Exception as e:
        st.error(f"An unexpected error occurred during sign-up: {e}")

def logout():
    """Logs out the current user."""
    try:
        supabase.auth.sign_out()
        # Clear all session state variables upon logout
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    except Exception as e:
        st.error(f"Logout failed: {e}")

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="RAG Chatbot", page_icon="📘", layout="wide")

# --- UI Styling ---
st.markdown("""
    <style>
    /* General styling */
    .stApp {
        background-color: #1a1a1a;
        color: #f5f5f5;
        font-family: 'Segoe UI', sans-serif;
    }
    /* Input box style */
    .stTextInput > div > div > input {
        background-color: #2b2b2b;
        border: 2px solid #f97316;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        color: #f5f5f5;
    }
    .stTextInput > div > div > div[data-baseweb="button"] {
       background-color: #2b2b2b;
        border: 2px solid #f97316;
        border-radius: 10px;
        padding: 10px;
        font-size: 16px;
        color: #f5f5f5;
            }
    /* Button styles */
    .stButton>button {
        background-color: #f97316;
        color: #1a1a1a;
        border-radius: 12px;
        padding: 12px 24px;
        font-weight: 600;
        font-size: 15px;
        border: none;
        box-shadow: 0px 4px 10px rgba(249, 115, 22, 0.3);
        transition: 0.2s ease-in-out;
    }
    .stButton>button:hover {
        background-color: #ea580c;
        transform: scale(1.05);
    }
    /* Chat bubble styles */
    .chat-bubble {
        padding: 12px;
        border-radius: 12px;
        margin: 8px 0;
        font-size: 15px;
        line-height: 1.5;
        color: #f5f5f5;
        box-shadow: 0px 2px 6px rgba(0,0,0,0.5);
        max-width: 80%;
        word-wrap: break-word;
    }
    .chat-bubble-user {
        background: #1f2937;
        border-left: 4px solid #f97316;
        align-self: flex-end;
        margin-left: auto;
    }
    .chat-bubble-bot {
        background: #2b2b2b;
        border-left: 4px solid #3b82f6;
        align-self: flex-start;
    }
    </style>
""", unsafe_allow_html=True)

# --- Initialize Session State ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Authentication Gate ---
# --- Authentication Gate ---
if not st.session_state.logged_in:
    st.markdown(
        """
        <div style="display: flex; justify-content: center; align-items: center;">
            <img src="https://res.cloudinary.com/dxu4rrvdh/image/upload/e_background_removal/e_dropshadow:azimuth_220;elevation_60;spread_20/f_png/v1755923220/robot-head-speech-bubble-red-600nw-2483741073_x4phdj.webp" 
                 width="220" style="border-radius:50%;">
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.title("Login to your RAG Chatbot")
    st.write("Enter your credentials or create a new account.")

    with st.form("login_form"):
        email = st.text_input("Email")
        password = st.text_input("Password", type="password")
        
        login_button = st.form_submit_button("Login")
        signup_button = st.form_submit_button("Sign Up")

        if login_button:
            if email and password:
                sign_in(email, password)
            else:
                st.warning("Please enter both email and password.")
        
        if signup_button:
            if email and password:
                sign_up(email, password)
            else:
                st.warning("Please enter both email and password.")

else:
    # --- Load LLM (Cached for Performance) ---
    @st.cache_resource
    def load_model():
        """Loads the Flan-T5 model and tokenizer."""
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
        return pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    qa_pipeline = load_model()

    # --- Main Application UI ---
    st.title("📘 Chat with your documents like never before.")

    # --- Sidebar for User Profile and Document Management ---
    with st.sidebar:
        st.header("👤 My Account")
        if st.session_state.user:
            st.write(f"Logged in as: **{st.session_state.user.email}**")
        
        if st.button("Logout"):
            logout()
        
        st.write("---")
        st.header("📂 Document Management")
        
        uploaded_files = st.file_uploader(
            "Upload new PDFs to chat with them for this session.",
            type=["pdf"],
            accept_multiple_files=True
        )

        if uploaded_files:
            with st.spinner("Processing and indexing uploaded documents..."):
                texts = []
                for uploaded_file in uploaded_files:
                    try:
                        reader = PdfReader(uploaded_file)
                        for page in reader.pages:
                            txt = page.extract_text()
                            if txt and txt.strip():
                                texts.append(txt.strip())
                    except Exception as e:
                        st.error(f"Error reading {uploaded_file.name}: {e}")
                
                if texts:
                    chunks = chunk_text(texts)
                    embedder = get_embedder()
                    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=False)
                    
                    # Create and store a new FAISS index for the uploaded files
                    active_index = faiss.IndexFlatL2(embeddings.shape[1])
                    active_index.add(embeddings)
                    
                    st.session_state.active_index = active_index
                    st.session_state.active_chunks = chunks
                    st.success("✅ Documents indexed successfully!")

        if st.button("New Chat"):
            st.session_state.messages = []
            st.rerun()

    # --- Load or Initialize Vectorstore ---
    # Load a prebuilt vectorstore if no custom documents have been uploaded in this session
    if "active_index" not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            try:
                active_index, active_chunks = load_vectorstore()
                st.session_state.active_index = active_index
                st.session_state.active_chunks = active_chunks
            except Exception as e:
                st.error(f"Could not load the prebuilt vectorstore: {e}")
                st.stop()

    # --- Main Chat Interface ---
    # Display previous messages
    for msg in st.session_state.messages:
        role_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
        st.markdown(f"<div class='chat-bubble {role_class}'><b>{'You' if msg['role'] == 'user' else 'Bot'}:</b> {msg['content']}</div>", unsafe_allow_html=True)
        if msg["role"] == "assistant" and "context" in msg:
            with st.expander("📖 View Retrieved Context"):
                for ctx_chunk in msg["context"]:
                    st.write(f"- {ctx_chunk}")

    # Handle user input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.markdown(f"<div class='chat-bubble chat-bubble-user'><b>You:</b> {prompt}</div>", unsafe_allow_html=True)

        with st.spinner("Thinking..."):
            try:
                # Retrieve context from the active vectorstore
                context = retrieve(prompt, st.session_state.active_index, st.session_state.active_chunks)
                combined_context = " ".join(context)
                
                # Build prompt for LLM
                llm_prompt = f"Based on the following context, answer the question.\n\nContext: {combined_context}\n\nQuestion: {prompt}\n\nAnswer:"
                response = qa_pipeline(llm_prompt, max_new_tokens=256)[0]["generated_text"]

                # Add bot message and context to chat history
                bot_message = {"role": "assistant", "content": response, "context": context}
                st.session_state.messages.append(bot_message)
                
                # Display bot response and context
                st.markdown(f"<div class='chat-bubble chat-bubble-bot'><b>Bot:</b> {response}</div>", unsafe_allow_html=True)
                if context:
                    with st.expander("📖 View Retrieved Context"):
                        for ctx_chunk in context:
                            st.write(f"- {ctx_chunk}")

            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
