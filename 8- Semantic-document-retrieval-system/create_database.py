from dotenv import load_dotenv
load_dotenv()  # reads OPENAI_API_KEY from .env

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
import shutil

CHROMA_PATH = "chroma"
DATA_PATH = "data/books"  # put .md files here (subfolders OK)

def main():
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    if not documents:
        print(f"No documents found under {DATA_PATH}. Ensure you have *.md files.")
        return
    chunks = split_text(documents)
    save_to_chroma(chunks)

def load_documents():
    # recursive + markdown
    loader = DirectoryLoader(
        DATA_PATH,
        glob="**/*.md",
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=100,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")
    if chunks:
        print("Sample chunk:", chunks[0].page_content[:200], "...")
        print("Metadata:", chunks[0].metadata)
    return chunks

def save_to_chroma(chunks: list[Document]):
    # fresh rebuild
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma.from_documents(chunks, embeddings, persist_directory=CHROMA_PATH)
    db.persist()
    print(f"✅ Saved {len(chunks)} chunks to {CHROMA_PATH}.")

if __name__ == "__main__":
    main()
