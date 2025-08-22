import os
import faiss
import pickle
from typing import List, Tuple
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader

INDEX_DIR = "vectorstore"
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
CHUNKS_PATH = os.path.join(INDEX_DIR, "chunks.pkl")
DOCS_PATH = "docs"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Cache the embedder to avoid reloading repeatedly
_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder

def load_pdfs() -> List[str]:
    """Load PDFs from DOCS_PATH and extract text page by page."""
    texts: List[str] = []
    if not os.path.isdir(DOCS_PATH):
        return texts
    for filename in os.listdir(DOCS_PATH):
        if filename.lower().endswith(".pdf"):
            reader = PdfReader(os.path.join(DOCS_PATH, filename))
            for page in reader.pages:
                txt = page.extract_text()
                if txt and txt.strip():
                    texts.append(txt.strip())
    return texts

def chunk_text(texts: List[str], chunk_size_words: int = 300, overlap_words: int = 50) -> List[str]:
    """Split texts into overlapping word chunks for better retrieval."""
    chunks: List[str] = []
    for text in texts:
        words = text.split()
        n = len(words)
        i = 0
        while i < n:
            chunk = " ".join(words[i:i+chunk_size_words])
            if chunk:
                chunks.append(chunk)
            i += max(1, chunk_size_words - overlap_words)
    return chunks

def build_vectorstore() -> Tuple[faiss.IndexFlatL2, List[str]]:
    """Build FAISS index from PDFs and persist to disk."""
    os.makedirs(INDEX_DIR, exist_ok=True)
    texts = load_pdfs()
    if not texts:
        # Create an empty index to keep app functional
        dim = 384  # all-MiniLM-L6-v2 dimension
        index = faiss.IndexFlatL2(dim)
        chunks: List[str] = []
        faiss.write_index(index, INDEX_PATH)
        with open(CHUNKS_PATH, "wb") as f:
            pickle.dump(chunks, f)
        return index, chunks

    chunks = chunk_text(texts)
    embedder = get_embedder()
    embeddings = embedder.encode(chunks, convert_to_numpy=True, show_progress_bar=True)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, INDEX_PATH)
    with open(CHUNKS_PATH, "wb") as f:
        pickle.dump(chunks, f)

    return index, chunks

def load_vectorstore() -> Tuple[faiss.IndexFlatL2, List[str]]:
    """Load the FAISS index and chunks or build them if missing."""
    if os.path.exists(INDEX_PATH) and os.path.exists(CHUNKS_PATH):
        index = faiss.read_index(INDEX_PATH)
        with open(CHUNKS_PATH, "rb") as f:
            chunks = pickle.load(f)
        return index, chunks
    return build_vectorstore()

def retrieve(query: str, index: faiss.IndexFlatL2, chunks: List[str], k: int = 3) -> List[str]:
    """Retrieve top-k relevant chunks for a query."""
    if not chunks or index.ntotal == 0:
        return []
    embedder = get_embedder()
    q_emb = embedder.encode([query], convert_to_numpy=True)
    D, I = index.search(q_emb, k)
    
    retrieved_chunks = []
    for i in I[0]:
        if 0 <= i < len(chunks):
            retrieved_chunks.append(chunks[i])
            
    return retrieved_chunks
