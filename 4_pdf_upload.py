import streamlit as st
from langchain_classic.chains import create_history_aware_retriever,create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_chroma import Chroma
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings

import os
from dotenv import load_dotenv

os.environ["HF_TOKEN"]=os.getenv("HF_TOKEN")
embeddings=HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

##setup streamlit app
st.title("Conversational RAG with pdf upload and chat history")
st.write("Upload pdf and chat with their content")

##input groq api key
api_key=st.text_input("Enter groq api key:",type="password")

##CHECK IF GROQ API IS PROVIDED
if api_key:
    llm=ChatGroq(api_key=api_key,model_name="llama-3.1-8b-instant")

    session_id=st.text_input("Session ID:",value="default")

    if "store" not in st.session_state:
        st.session_state.store={}

    uploaded_files=st.file_uploader("Choose a PDF File",type="pdf",accept_multiple_files=True)

    if uploaded_files:
        documents=[]
        ## for each file that is uploaded we take it and first store it in local folder
        for uploaded_file in uploaded_files:
            temppdf=f"./temp.pdf"

            with open(temppdf,"wb") as file:
                file.write(uploaded_file.getvalue())
                file_name=uploaded_file.name

            loader=PyPDFLoader(temppdf)
            docs=loader.load()
            documents.extend(docs)

        ##split docs and create embeddings
        text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200)
        splits=text_splitter.split_documents(documents)
        vectorstore=Chroma.from_documents(documents=splits,embedding=OllamaEmbeddings(model="nomic-embed-text"))
        retiever=vectorstore.as_retriever()


        ## creating contextual system prompt
        contextualize_q_system_prompt=(
                "Given a chat history and the latest user question"
                "which might reference context in the chat history, "
                "formulate a standalone question which can be understood "
                "without the chat history. Do NOT answer the question, "
                "just reformulate it if needed and otherwise return it as is."
            )
        
        ##this will go to llm to rewite question
        contextualize_q_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",contextualize_q_system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),
            ]
        )

        history_aware_retiever=create_history_aware_retriever(llm,retiever,contextualize_q_prompt)

        ##answer question based on context here context is replaced by stuff document chain
        system_prompt = (
                    "You are an assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer "
                    "the question. If you don't know the answer, say that you "
                    "don't know. Use three sentences maximum and keep the "
                    "answer concise."
                    "\n\n"
                    "{context}"
                )
        
        qa_prompt=ChatPromptTemplate.from_messages(
            [
                ("system",system_prompt),
                MessagesPlaceholder("chat_history"), ## history is not filled automatically but we pass it while invoking chat gpt more
                ("human","{input}")
            ]
        )

        question_answer_chain=create_stuff_documents_chain(llm,qa_prompt)
        rag_chain=create_retrieval_chain(history_aware_retiever,question_answer_chain)

        def get_session_history(session:str)->BaseChatMessageHistory:
            if session_id not in st.session_state.store:
                st.session_state.store[session_id]=ChatMessageHistory()

            return st.session_state.store[session_id]
        
        conversational_rag_chain=RunnableWithMessageHistory(
                rag_chain,get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
    
        user_input = st.text_input("Your question:")
        if user_input:
            session_history=get_session_history(session_id)
            response = conversational_rag_chain.invoke(
                {"input": user_input},
                config={
                    "configurable": {"session_id":session_id}
                },  # constructs a key "abc123" in `store`.
            )
            st.write(st.session_state.store)
            st.write("Assistant:", response['answer'])
            st.write("Chat History:", session_history.messages)
else:
    st.warning("Please enter the GRoq API Key")