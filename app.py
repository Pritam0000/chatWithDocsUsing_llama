import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.llms import HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import tempfile
import os

# Get Hugging Face token from environment variable
HF_TOKEN = os.getenv('HF_TOKEN')

# Load Meta-Llama model
@st.cache_resource
def load_meta_llama_model():
    model_name = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    model = AutoModelForCausalLM.from_pretrained(model_name, use_auth_token=HF_TOKEN)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, max_length=2048, temperature=0.7)
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm

# Process the uploaded PDF
def process_pdf(pdf_path):
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(texts, embeddings)
    return db

# Streamlit app
st.title("Chat with Your PDF using Meta-Llama-3.1-8B-Instruct")

# File uploader
pdf_file = st.file_uploader("Upload your PDF", type="pdf")

if pdf_file is not None:
    # Create a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    db = process_pdf(tmp_file_path)
    llm = load_meta_llama_model()

    # Initialize ConversationalRetrievalChain
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    qa_chain = ConversationalRetrievalChain.from_llm(llm, db.as_retriever(), memory=memory)

    # Chat interface
    st.subheader("Chat with your PDF")
    user_question = st.text_input("Ask a question about your PDF:")
    
    if user_question:
        response = qa_chain({"question": user_question})
        st.write("Answer:", response['answer'])

    # Display chat history
    with st.expander("Chat History"):
        for message in memory.chat_memory.messages:
            if message.type == 'human':
                st.write("Human: ", message.content)
            elif message.type == 'ai':
                st.write("AI: ", message.content)

    # Remove the temporary file
    os.unlink(tmp_file_path)
