import streamlit as st
import io
import torch
import faiss
import tempfile
import numpy as np
import soundfile as sf
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline as hf_pipeline
from PyPDF2 import PdfReader
from supabase import create_client, Client, AuthApiError
from TTS.api import TTS
import transformers

from rag_utils import load_vectorstore, retrieve, chunk_text, get_embedder

# ---------------- Supabase Client Initialization ----------------
try:
    supabase_url = st.secrets["SUPABASE_URL"]
    supabase_key = st.secrets["SUPABASE_KEY"]
    supabase: Client = create_client(supabase_url, supabase_key)
except KeyError as e:
    st.error(f"Missing Streamlit secret: {e}. Ensure SUPABASE_URL and SUPABASE_KEY are in .streamlit/secrets.toml")
    st.stop()

# ---------------- Authentication Functions ----------------
def sign_in(email, password):
    try:
        response = supabase.auth.sign_in_with_password({"email": email, "password": password})
        if hasattr(response, "user") and response.user:
            st.session_state["user"] = response.user
            st.session_state["logged_in"] = True
            st.rerun()
        else:
            st.error("Login failed. Did you confirm your email?")
    except AuthApiError as e:
        st.error(f"Login failed: {e.message}")
    except Exception as e:
        st.error(f"Unexpected error during login: {e}")

def sign_up(email, password):
    try:
        response = supabase.auth.sign_up({"email": email, "password": password})
        if response.user:
            st.success("Account created! Please check your email for a confirmation link.")
        else:
            st.error("Sign-up failed. User may already exist or password is too weak.")
    except AuthApiError as e:
        st.error(f"Sign-up failed: {e.message}")
    except Exception as e:
        st.error(f"Unexpected error during sign-up: {e}")

def logout():
    try:
        supabase.auth.sign_out()
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()
    except Exception as e:
        st.error(f"Logout failed: {e}")

# ---------------- Hugging Face & Coqui TTS ----------------
@st.cache_resource
def load_tts(model_name: str):
    """Load a TTS model safely: HF pipeline or Coqui XTTS."""
    if model_name == "coqui/XTTS-v2":
        return TTS("tts_models/multilingual/multi-dataset/xtts_v2")
    else:
        device = 0 if torch.cuda.is_available() else -1
        return hf_pipeline("text-to-speech", model=model_name, device=device)

def generate_audio(text: str, tts_pipeline, model_name: str):
    """Generate speech and save as WAV."""
    try:
        if model_name == "coqui/XTTS-v2":
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                tts_pipeline.tts_to_file(text=text, speaker="en", language="en", file_path=tmp.name)
                return tmp.name
        else:
            out = tts_pipeline(text)
            audio = out["audio"]
            sr = int(out.get("sampling_rate", 16000))

            if hasattr(audio, "cpu"):  # torch tensor
                audio = audio.detach().cpu().numpy()
            audio = np.asarray(audio).squeeze().astype(np.float32)
            audio = np.clip(audio, -1.0, 1.0)

            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
                sf.write(tmp.name, audio, sr, format="WAV", subtype="PCM_16")
                return tmp.name
    except Exception as e:
        st.error(f"TTS generation failed: {e}")
        return None

# ---------------- Streamlit Page Config ----------------
st.set_page_config(page_title="RAG Chatbot", page_icon="📘", layout="wide")

# ---------------- UI Styling ----------------
st.markdown("""
    <style>
    .stApp { background-color: #1a1a1a; color: #f5f5f5; font-family: 'Segoe UI', sans-serif; }
    .stTextInput > div > div > input { background-color: #2b2b2b; border: 2px solid #f97316; border-radius: 10px; padding: 10px; font-size: 16px; color: #f5f5f5; }
    .stButton>button { background-color: #f97316; color: #1a1a1a; border-radius: 12px; padding: 12px 24px; font-weight: 600; font-size: 15px; border: none; box-shadow: 0px 4px 10px rgba(249, 115, 22, 0.3); transition: 0.2s ease-in-out; }
    .stButton>button:hover { background-color: #ea580c; transform: scale(1.05); }
    .chat-container { display: flex; margin: 6px 0; width: 100%; }
    .chat-bubble { display: inline-block; padding: 12px 16px; border-radius: 16px; font-size: 15px; line-height: 1.4; color: #f5f5f5; box-shadow: 0px 2px 6px rgba(0,0,0,0.5); max-width: 70%; word-wrap: break-word; white-space: pre-wrap; }
    .chat-bubble-user { background: #1f2937; border: 2px solid #f97316; border-bottom-right-radius: 4px; margin-left: auto; text-align: right; }
    .chat-bubble-bot { background: #2b2b2b; border: 2px solid #3b82f6; border-bottom-left-radius: 4px; margin-right: auto; text-align: left; }
    </style>
""", unsafe_allow_html=True)

# ---------------- Session State ----------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user" not in st.session_state:
    st.session_state.user = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------- Authentication Gate ----------------
if not st.session_state.logged_in:
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
    # ---------------- Load LLM ----------------
    @st.cache_resource
    def load_model():
        model_name = "google/flan-t5-base"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float32)
        return transformers.pipeline("text2text-generation", model=model, tokenizer=tokenizer)

    qa_pipeline = load_model()

    # ---------------- Main Application ----------------
    st.title("📘 Chat with your documents like never before.")

    # ---------------- Sidebar ----------------
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
                    
                    active_index = faiss.IndexFlatL2(embeddings.shape[1])
                    active_index.add(embeddings)
                    
                    st.session_state.active_index = active_index
                    st.session_state.active_chunks = chunks
                    st.success("✅ Documents indexed successfully!")

        if st.button("New Chat"):
            st.session_state.messages = []
            st.rerun()

        st.write("---")
        st.header("🎙️ Voice Settings")
        tts_model_choice = st.selectbox(
            "Choose a TTS model",
            [
                "facebook/mms-tts-eng",
                "espnet/kan-bayashi_ljspeech_vits",
                "espnet/kan-bayashi_ljspeech_fastspeech2",
                "coqui/XTTS-v2"
            ],
            index=0
        )
        tts_pipeline = load_tts(tts_model_choice)

        play_audio = st.checkbox("🔊 Enable Audio Response", value=True)

    if "active_index" not in st.session_state:
        with st.spinner("Loading knowledge base..."):
            try:
                active_index, active_chunks = load_vectorstore()
                st.session_state.active_index = active_index
                st.session_state.active_chunks = active_chunks
            except Exception as e:
                st.error(f"Could not load the prebuilt vectorstore: {e}")
                st.stop()

    # ---------------- Display Previous Messages ----------------
    for msg in st.session_state.messages:
        role_class = "chat-bubble-user" if msg["role"] == "user" else "chat-bubble-bot"
        role_label = "You" if msg["role"] == "user" else "Bot"
        st.markdown(
            f"""
            <div class="chat-container">
                <div class="chat-bubble {role_class}">
                    <b>{role_label}:</b> {msg['content']}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        if msg["role"] == "assistant" and "context" in msg:
            with st.expander("📖 View Retrieved Context"):
                for ctx_chunk in msg["context"]:
                    st.write(f"- {ctx_chunk}")

    # ---------------- Handle New Chat Input ----------------
    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.messages.append({"role": "user", "content": prompt})

        st.markdown(
            f"""
            <div class="chat-container">
                <div class="chat-bubble chat-bubble-user">
                    <b>You:</b> {prompt}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

        with st.spinner("Thinking..."):
            try:
                context = retrieve(prompt, st.session_state.active_index, st.session_state.active_chunks)
                combined_context = " ".join(context)
                
                llm_prompt = f"Based on the following context, answer the question.\n\nContext: {combined_context}\n\nQuestion: {prompt}\n\nAnswer:"
                response = qa_pipeline(llm_prompt, max_new_tokens=256)[0]["generated_text"]

                bot_message = {"role": "assistant", "content": response, "context": context}
                st.session_state.messages.append(bot_message)

                st.markdown(
                    f"""
                    <div class="chat-container">
                        <div class="chat-bubble chat-bubble-bot">
                            <b>Bot:</b> {response}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if context:
                    with st.expander("📖 View Retrieved Context"):
                        for ctx_chunk in context:
                            st.write(f"- {ctx_chunk}")

                if play_audio:
                    audio_path = generate_audio(response, tts_pipeline, tts_model_choice)
                    if audio_path:
                        st.audio(audio_path, format="audio/wav")

            except Exception as e:
                st.error(f"An error occurred while generating the response: {e}")
