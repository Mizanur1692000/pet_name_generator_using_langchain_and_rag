from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

import os

def create_chain():
    loader = TextLoader("docs/pet_names.txt")
    docs = loader.load()

    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    chunks = splitter.split_documents(docs)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")  # 👈 specify model
    vectordb = FAISS.from_documents(chunks, embeddings)

    retriever = vectordb.as_retriever()
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama3-70b-8192")

    prompt = ChatPromptTemplate.from_template(
        "You are a creative pet name generator. Based on the following user input, generate 5 unique pet names.\n\nInput: {question}\n\nOnly give the names, nothing else."
    )

    chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=False)
    return chain