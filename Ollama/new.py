from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
import streamlit as st
import os
from dotenv import load_dotenv
import time

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

# Prompt Template
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant, please respond to my question"),
        ("user", "Question: {question}")
    ]
)

st.title('Langchain Demo with Ollama')
input_text = st.text_input("Search the Topic You Want")

# Ollama LLM with GPU and streaming enabled
llm = Ollama(
    model="llama3.1"
)

if input_text:
    # Placeholder to dynamically update text
    output_placeholder = st.empty()
    full_text = ""

    # Stream the output from the LLM
    for chunk in llm.stream(input_text):
        for char in chunk:
            full_text += char
            output_placeholder.markdown(full_text)
            time.sleep(0.005)  # faster per-character update
