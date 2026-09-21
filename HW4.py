import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
import sys

__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import chromadb
from pathlib import Path
from PyPDF2 import PdfReader

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
    
def extract_text_from_pdf(pdf_path):
    """Read a PDF file and return its full text as a single string."""
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

def add_to_collection(collection, text, file_name):
    """Embed `text` with OpenAI and add it to the ChromaDB collection."""
    response = client.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    embedding = response.data[0].embedding

    collection.add(
        documents=[text],
        ids=[file_name],
        embeddings=[embedding],
        metadatas=[{"filename": file_name}]
    )
   
def load_pdfs_to_collection(folder_path, collection):
    """Loop over every PDF in folder_path, extract its text, and add it
    to the collection. Returns the number of documents loaded."""
    loaded = 0
    for pdf_path in Path(folder_path).glob("*.pdf"):
        text = extract_text_from_pdf(pdf_path)
        add_to_collection(collection, text, pdf_path.name)
        loaded += 1
    return loaded

def build_lab4_vectordb():
    """Create (or open) the Lab4Collection ChromaDB collection and populate
    it with the 7 syllabus PDFs, but only if it's currently empty."""
    chroma_client = chromadb.PersistentClient(path="./ChromaDB_for_Lab")
    collection = chroma_client.get_or_create_collection("Lab4Collection")

    if collection.count() == 0:
        load_pdfs_to_collection("./Lab-04-Data/", collection)

    return collection


# Show title and description.
st.title("Chad the Course Advisor")

st.write(
    "Hi! I'm Chad, your course advisor chatbot. I've read through the "
    "syllabi for several IST courses, so feel free to ask me about course "
    "topics, what you'll learn, or which class might be the right fit for you."
)

OPENAI_API_KEY = st.secrets.OPEN_AI_KEY



# Create an OpenAI client.
client = OpenAI(api_key=OPENAI_API_KEY)

model = "gpt-5-mini"

# Build the vector DB once and cache it in session_state
if 'Lab4_VectorDB' not in st.session_state:
    with st.spinner("Loading course syllabi into the knowledge base..."):
        st.session_state.Lab4_VectorDB = build_lab4_vectordb()

collection = st.session_state.Lab4_VectorDB

#### PART A TEST — verify the vectorDB
# topic = st.sidebar.text_input('Test topic', placeholder='e.g., Generative AI')
# if topic:
#    response = client.embeddings.create(input=topic, model="text-embedding-3-small")
#    query_embedding = response.data[0].embedding
#    results = collection.query(query_embeddings=[query_embedding], n_results=3)
#
#    st.subheader(f'Results for: {topic}')
#    for i in range(len(results['documents'][0])):
#        doc_id = results['ids'][0][i]
#        st.write(f'**{i+1}. {doc_id}**')



if "messages" not in st.session_state:
    st.session_state.messages = []

# Keep the whole conversation history
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
    # Embed the user's question, then find the 3 most relevant syllabus chunks
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
            "You are a helpful course advisor chatbot for the School of "
            "Information Studies. Use the following syllabus excerpts to "
            "answer the student's question when relevant. If you use "
            "information from these excerpts, explicitly say so (e.g., "
            "'Based on the syllabus for IST 488...'). If the excerpts "
            "don't contain relevant information, say you're answering "
            "from general knowledge instead.\n\n"
            f"--- Retrieved course information ---\n{context_text}"
        )
    }

    # Generate an answer using the LLM
    with st.chat_message("assistant"):
        stream = client.chat.completions.create(
            model=model,
            messages=[system_prompt] + st.session_state.messages[-6:],
            stream=True,
        )
        response = st.write_stream(stream)

  
    st.session_state.messages.append({"role": "assistant", "content": response})








