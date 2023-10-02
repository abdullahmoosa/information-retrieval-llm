# Import necessary modules

import streamlit as st
import fitz
import time
import yaml
import streamlit_authenticator as stauth
from yaml.loader import SafeLoader
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import HuggingFaceHub
from utilities.htmlTemplates import css,bot_template,user_template

# Loading configs from to initialize authentication
with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)
def get_pdf_text(pdf):
    """
    The function `get_pdf_text` extracts the text from each page of a PDF file and returns it as a
    single string.
    
    :param pdf: The parameter "pdf" is the file path or file object of the PDF file that you want to
    extract text from
    :return: the text extracted from the PDF file.
    """
    text = ""

    pdf_reader = PdfReader(pdf)
    for page in pdf_reader.pages:
        text += page.extract_text()

    return text

def get_ocr_text(pdf):
    """
    The function `get_ocr_text` takes a PDF file as input, saves it to a temporary location, opens it
    using the `fitz` library, extracts the text from each page, concatenates the text, and finally
    closes the PDF document.
    
    :param pdf: The parameter "pdf" is the PDF file that you want to extract text from. It should be a
    file-like object, such as the result of opening a file using the "open" function in Python or the
    result of reading a file using the "read" method
    """
    text = ""

    # Save the uploaded PDF to a temporary location
    with open("temp.pdf", "wb") as temp_file:
        temp_file.write(pdf.read())

    doc = fitz.open("temp.pdf")
    for page in doc:
        text += page.get_text()()

    # Close the PDF document
    doc.close()

    return text

def get_text_chunks(raw_text):
    """
    The function `get_text_chunks` takes in a raw text and splits it into chunks of a specified size,
    with a specified overlap, using a character-based text splitter.
    
    :param raw_text: The raw text that you want to split into chunks
    :return: a list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator= "\n",
        chunk_size =1000,
        chunk_overlap = 200,
        length_function = len
        )
    chunks = text_splitter.split_text(text=raw_text)
    return chunks

def get_vectorstore(text_chunks):
    """
    The function `get_vectorstore` takes a list of text chunks as input and returns a vector store
    created using the HuggingFaceInstructEmbeddings model.
    
    :param text_chunks: The `text_chunks` parameter is a list of text chunks or sentences that you want
    to create a vector store for. Each text chunk represents a piece of text that you want to encode
    into a vector representation
    :return: a vector store, which is created using the text chunks and their corresponding embeddings.
    """
    embeddings = HuggingFaceInstructEmbeddings(model_name = "BAAI/bge-large-en-v1.5", )
    vector_store = FAISS.from_texts(texts= text_chunks, embedding= embeddings)
    return vector_store

def get_conversation_chain(vectorstore):
    """
    The function `get_conversation_chain` creates a conversational retrieval chain using a language
    model and a vector store.
    
    :param vectorstore: The `vectorstore` parameter is an instance of a vector store, which is used as a
    retriever in the conversational retrieval chain. It contains precomputed vector representations of
    documents or sentences that can be used for similarity-based retrieval
    :return: a conversation chain object.
    """
    llm = HuggingFaceHub(repo_id= "google/flan-t5-large",model_kwargs={"temperature" :0.5,"max_length" :1024})
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages= True)
    conversation_chains = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory = memory
    )
    return conversation_chains

def handle_user_input(user_question):
    """
    The function `handle_user_input` takes a user question as input, sends it to a chatbot model, and
    displays the conversation history between the user and the chatbot.
    
    :param user_question: The user_question parameter is the input question or message from the user
    that needs to be processed and responded to
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']
    for index,message in enumerate(st.session_state.chat_history):
        if index % 2 == 0:
            st.write(user_template.replace("{{MSG}}",message.content),unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}",message.content), unsafe_allow_html=True)

def on_change():
    """
    The function `on_change()` resets the conversation and chat history in the session state.
    """
    st.session_state.conversation = None
    st.session_state.chat_history = None

# st.set_page_config(page_title="Bank AI", page_icon= ":bank:")
def main():
    # Loading the environmental variables
    load_dotenv()

    st.write(css,unsafe_allow_html=True)
    st.header("Chat with Bank-AI :bank:")
    name, authentication_status, username = authenticator.login('Login', 'main')

    # If the user inputs the proper username and password then the authentication status will be true
    if authentication_status:
        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        # Getting input question from the user
        user_question = st.text_input("Ask a question about uploaded documents...",key="user_input",)
        if st.button("Submit"):
            if user_question:
                if st.session_state.conversation is not None:
                    handle_user_input(user_question)

        # In the sidebar, user can upload documents from which they will ask questions
        with st.sidebar:
            st.subheader("Your Documents")
            pdf = st.file_uploader("Upload your PDFS here and click on 'Process'", on_change=on_change)
            if st.button("Process"):
                with st.spinner("Processing..."):
                    try:
                        raw_text = get_pdf_text(pdf)
                    except:
                        st.error("Error Loading PDF! Please try again", icon= "🚨")
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)

                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    

if __name__ == '__main__':
    main()