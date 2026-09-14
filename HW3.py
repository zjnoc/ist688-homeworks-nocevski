
import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup
from google import genai

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


# Show title and description.
st.title("Chad the Chatbot")

st.write(
    "Hi! I'm Chad, a chatbot that can read up to two web pages and answer "
    "questions about them. You can also choose which AI model I use, OpenAI "
    "or Gemini. I remember the last 6 messages of our conversation, so feel "
    "free to ask follow-up questions. Ready to chat?"
)


url_1 = st.sidebar.text_input("Enter URL 1:")
url_2 = st.sidebar.text_input("Enter URL 2:")

url_1_content = ""
url_2_content = ""

if url_1:
    url_1_content = read_url_content(url_1)

if url_2:
    url_2_content = read_url_content(url_2)



llm_choice = st.sidebar.selectbox(
    "Choose your LLM:",
    ("OpenAI", "Gemini")
)

if llm_choice == "OpenAI":
    model = "gpt-4.1"
else:
    model = "google/gemini-3.1-pro-preview"


OPENAI_API_KEY = st.secrets.OPEN_AI_KEY
OPENROUTER_API_KEY = st.secrets.OPENROUTER_API_KEY


# Create an OpenAI client.
client = OpenAI(api_key=OPENAI_API_KEY)

openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)


system_prompt = {
    "role": "system",
    "content": f"Explain things simply, like to a 10 year old. Answer the "
                f"question, then ask 'Do you want more info?' If they say yes, "
                f"give more detail and ask again. If they say no, ask what else "
                f"you can help with. Here is some reference information you can "
                f"use to answer questions: {url_1_content} {url_2_content}"
}

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


    # Pick the right client based on which vendor the user selected.
    if llm_choice == "OpenAI":
        active_client = client
    else:
        active_client = openrouter_client

    # Generate an answer using the selected LLM.
    with st.chat_message("assistant"):
        stream = active_client.chat.completions.create(
            model=model,
            messages=[system_prompt] + st.session_state.messages[-6:],
            stream=True,
        )
        response = st.write_stream(stream)


    
    st.session_state.messages.append({"role": "assistant", "content": response})








