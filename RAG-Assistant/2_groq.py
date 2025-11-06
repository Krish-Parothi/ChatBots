import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community import llms
from langchain_groq import ChatGroq 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader




load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

st.title("ChatGroq with Llama3.1")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama-3.1-8b-instant")

prompt = ChatPromptTemplate.from_template(

    """
Answer the Questions Based on the Provided context Only. Please Provide the most accurate response based on the question

<context>
{context}
<context>

Questions: {input}

"""

)

def vector_embedding():
    
    if "vectors" not in st.session_state:

        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        st.session_state.loader = PyPDFDirectoryLoader("./data") # Data Ingestion
        st.session_state.docs = st.session_state.loader.load() # Splitting
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200) # chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20]) #Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings) # Vector OpenA Embeddings



    

prompt1 = st.text_input("Enter Your Questions From Documents")

if st.button("Document Embedding"):
    vector_embedding()
    st.write("Vector Store DB is Ready")



import time

if prompt1:
    document_chain = create_stuff_documents_chain(llm,prompt)
    retreiver = st.session_state.vectors.as_retriever()
    retreival_chain = create_retrieval_chain(retreiver, document_chain)
    start = time.process_time()
    response = retreival_chain.invoke({"input":prompt1})
    print("Response Time:", time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------")



