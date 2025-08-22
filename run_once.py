from rag_utils import build_vectorstore

if __name__ == "__main__":
    print("🚀 Building vectorstore from PDFs in ./docs ...")
    build_vectorstore()
    print("✅ Done! You can now run: streamlit run app.py")