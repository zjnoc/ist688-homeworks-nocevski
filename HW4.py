import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from pathlib import Path

def read_url_content(url):
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        }
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an exception for HTTP errors
        soup = BeautifulSoup(response.content, 'html.parser')
        return soup.get_text()
    except requests.RequestException as e:
        print(f"Error reading {url}: {e}")
        return None
    
def extract_text_from_html(html_path):
    """Read an HTML file and return its visible text as a single string."""
    with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
        soup = BeautifulSoup(f.read(), "html.parser")
    return soup.get_text(separator="\n", strip=True)


def chunk_text(text, num_chunks=2):
    """Split text into `num_chunks` roughly equal pieces.

    CHUNKING METHOD: Fixed-size chunking (splitting by character count into
    equal-sized pieces). We chose this over semantic chunking because these
    org pages are short and don't have consistent heading/section structure
    to split on reliably — a simple even split by length keeps the
    implementation simple while still giving each chunk a narrower, more
    query-matchable slice of the page than embedding the whole page at once.
    """
    if not text:
        return [""] * num_chunks

    chunk_size = max(1, len(text) // num_chunks)
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    # If splitting produced more pieces than requested (due to rounding),
    # merge any extra trailing pieces into the last chunk
    if len(chunks) > num_chunks:
        chunks[num_chunks - 1:] = ["".join(chunks[num_chunks - 1:])]

    return chunks
    
def add_chunks_to_collection(collection, chunks, file_name):
    """Embed each chunk with OpenAI and add it to the ChromaDB collection.
    Each chunk gets its own unique ID: '<file_name>_chunk0', '<file_name>_chunk1', etc.
    file_name is stored in metadata so we know which original page a chunk came from."""
    for i, chunk in enumerate(chunks):
        if not chunk.strip():
            continue  # skip empty chunks

        response = client.embeddings.create(
            input=chunk,
            model="text-embedding-3-small"
        )
        embedding = response.data[0].embedding

        collection.add(
            documents=[chunk],
            ids=[f"{file_name}_chunk{i}"],
            embeddings=[embedding],
            metadatas=[{"filename": file_name, "chunk_index": i}]
        )

def load_html_files_to_collection(folder_path, collection, limit=None):
    """Loop over every HTML file in folder_path, extract its text, split it
    into 2 chunks, and add both chunks to the collection.
    Returns the number of HTML files processed."""
    loaded = 0
    for html_path in Path(folder_path).glob("*.html"):
        if limit and loaded >= limit:
            break
        text = extract_text_from_html(html_path)
        chunks = chunk_text(text, num_chunks=2)
        add_chunks_to_collection(collection, chunks, html_path.name)
        loaded += 1
    return loaded

def build_hw4_vectordb():
    """Create (or open) the HW4Collection ChromaDB collection and populate
    it with the student org HTML pages (chunked), but only if it's currently empty."""
    chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_HW4")
    collection = chroma_client.get_or_create_collection("HW4Collection")

    if collection.count() == 0:
        load_html_files_to_collection("./HW-04-Data/", collection)

    return collection


# Show title and description.
st.title("iSchool Student Org Advisor")

st.write(
    "Hi! I'm your student organizations chatbot. I've read through the "
    "iSchool's student org pages, so ask me anything about clubs you can "
    "join, what they do, or how to get involved."
)

OPENAI_API_KEY = st.secrets.OPEN_AI_KEY

# Create an OpenAI client.
client = OpenAI(api_key=OPENAI_API_KEY)

model = "gpt-5-mini"

# Build the vector DB once and cache it in session_state
if 'HW4_VectorDB' not in st.session_state:
    with st.spinner("Loading student org pages into the knowledge base..."):
        st.session_state.HW4_VectorDB = build_hw4_vectordb()

collection = st.session_state.HW4_VectorDB

if "messages" not in st.session_state:
    st.session_state.messages = []

# Keep the whole conversation history displayed on screen
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Get user input.
if prompt := st.chat_input("What would you like to ask?"):
    # Add user's message to chat history.
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # ---- RAG retrieval step ----
    # Embed the user's question, then find the 3 most relevant org page chunks
    response = client.embeddings.create(input=prompt, model="text-embedding-3-small")
    query_embedding = response.data[0].embedding
    results = collection.query(query_embeddings=[query_embedding], n_results=3)

    retrieved_docs = results["documents"][0]
    retrieved_ids = results["ids"][0]

    context_text = "\n\n".join(
        f"[Source: {retrieved_ids[i]}]\n{retrieved_docs[i]}"
        for i in range(len(retrieved_docs))
    )
    
    # Build a system prompt that includes the retrieved context
    system_prompt = {
        "role": "system",
        "content": (
            "You are a helpful chatbot that answers questions about iSchool "
            "student organizations. Use the following excerpts from student "
            "org pages to answer the student's question when relevant. If "
            "you use information from these excerpts, explicitly say so "
            "(e.g., 'Based on the page for [org name]...'). If the excerpts "
            "don't contain relevant information, say you're answering from "
            "general knowledge instead.\n\n"
            f"--- Retrieved student org information ---\n{context_text}"
        )
    }

    # ---- Memory buffer: keep only the last 5 user/assistant interaction
    # pairs (10 messages total) so the conversation doesn't grow unbounded
    # and stays within a manageable context size for the LLM ----
    recent_messages = st.session_state.messages[-10:]

    # Generate an answer using the LLM
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model,
            messages=[system_prompt] + recent_messages,
            stream=True,
        )
        response = st.write_stream(stream)

    st.session_state.messages.append({"role": "assistant", "content": response})


