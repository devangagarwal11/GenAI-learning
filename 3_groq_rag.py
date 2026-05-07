import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFDirectoryLoader

from dotenv import load_dotenv
load_dotenv()
## load the GROQ API Key
os.environ['GROQ_API_KEY']=os.getenv("GROQ_API_KEY")

##to get direct key
groq_api_key=os.getenv("GROQ_API_KEY")

llm=ChatGroq(groq_api_key=groq_api_key,model_name="llama-3.1-8b-instant")

prompt=ChatPromptTemplate.from_template(
    """"
    Answer the questions based on context provided only. Please give most accurate response based on question
    <context>
    {context}
    <context> 
    question:{input}
    """
)

##why use session state?
def create_vector_embedding():
    """read content from pdf and apply splitting then embed and store in db"""

    if "vectors" not in st.session_state:
        st.session_state.embeddings=OllamaEmbeddings(model="nomic-embed-text")

        st.session_state.loader=PyPDFDirectoryLoader("research_papers") ## data ingestion step

        st.session_state.docs=st.session_state.loader.load() ##document loading

        st.session_state.text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)

        st.session_state.final_documents=st.session_state.text_splitter.split_documents(st.session_state.docs[:50]) ##only starting 50 docs so that it does not take much time

        st.session_state.vectors=FAISS.from_documents(st.session_state.final_documents,st.session_state.embeddings) ## we also need not return tese vectors as we are saving in session state

user_prompt=st.text_input("Enter your query from research paper")

if st.button("Create Embedding"):
    create_vector_embedding()
    st.write("Vector db is ready!!")

if user_prompt:
    document_chain=create_stuff_documents_chain(llm,prompt) ## creates a chain to pass list of docs to model
    retriever=st.session_state.vectors.as_retriever()
    rag_chain=create_retrieval_chain(retriever,document_chain)

    response=rag_chain.invoke({"input":user_prompt})
    st.write(response["answer"])