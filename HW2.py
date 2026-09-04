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
st.title("URL Summary")
st.write(
    "Enter a web page URL below and select a summary type - GPT will summarize it for you! "
    
)

url = st.text_input("Enter a URL to summarize:")

summary_type = st.sidebar.selectbox(
    "Summarize the document by:",
    ("in 100 words:", "in 2 connecting paragraphs", "in 5 bullet points")
)

use_advanced_model = st.sidebar.checkbox("Use advanced model")
llm_choice = st.sidebar.selectbox(
    "Choose your LLM:",
    ("OpenAI", "Gemini")
)


language = st.sidebar.selectbox(
    "Output language:",
    ("English", "Spanish", "French")
)


if llm_choice == "OpenAI":
    model = "gpt-4.1" if use_advanced_model else "gpt-4.1-nano"
else:
    model = "google/gemini-3.1-pro-preview" if use_advanced_model else "google/gemini-3.5-flash"


OPENAI_API_KEY = st.secrets.OPEN_AI_KEY
GEMINI_API_KEY = st.secrets.GEMINI_API_KEY

openai_client = OpenAI(api_key=OPENAI_API_KEY)
OPENROUTER_API_KEY = st.secrets.OPENROUTER_API_KEY
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)



if url:

    # Get the text content from the URL.
    document = read_url_content(url)

    if document:
        prompt_text = f"Summarize the following document {summary_type} in {language}: {document}"

        if llm_choice == "OpenAI":
            stream = openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                stream=True,
            )
            st.write_stream(stream)
        else:
            stream = openrouter_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt_text}],
                stream=True,
            )
            st.write_stream(stream)
    else:
        st.error("Could not read content from that URL. Please check it and try again.")


