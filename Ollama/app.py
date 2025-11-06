from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.llms import Ollama
import streamlit as st
import os 
from dotenv import load_dotenv

load_dotenv()

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")

prompt = ChatPromptTemplate.from_messages(

    [
        ("system", "You are a helpful assistant, please response to my question"),
        ("user", "Question: {question}" )
    ]
)

## Streamlit Framework

st.title('Langchain Demo with Ollama')
input_text = st.text_input("Search the Topic You Want")

# Ollama llm
llm = Ollama(model = "llama3.1")
output_parser = StrOutputParser()
chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({"question":input_text}))