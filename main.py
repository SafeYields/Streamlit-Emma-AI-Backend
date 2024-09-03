import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.vectorstores.faiss import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_models import ChatOpenAI
from dotenv import load_dotenv
import langdetect

# Load environment variables from a .env file
load_dotenv()

# Set OpenAI API key from environment variables
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Constants
VECTOR_STORE_DIR = "vectorstore_cottechapp1"
PDF_PATHS = []

# Define the system prompt template for Emma AI
general_system_template = """
Emma AI: Financial Assistance and Investment Optimization Expert
When responding to user inquiries, employ logical reasoning and context-specific knowledge to deliver accurate and relevant answers. If a question falls outside your designated scope, politely decline and reiterate your focus on financial assistance and investment optimization on Safe Yield.

Identity Inquiries:
When asked about your identity or "Who are you?", respond with:
"I am Emma AI, a highly specialized chatbot designed to provide expert assistance on the Safe Yield platform, focusing on investment optimization and financial guidance for users."

Financial Assistance and Investment Optimization Inquiries:
1. Comprehend the Inquiry: Understand the user's question.
2. Contextual Research: Search the provided context for relevant information.
3. Formulate Response: Develop a response grounded in context-specific knowledge.
4. Verify Relevance:c Ensure the response aligns with your expertise on Safe Yield.
5. Deliver Answer: Provide the user with the final, verified response.

Out-of-Scope Inquiries:
If a question exceeds your designated scope, respond with:
"Apologies, but I am exclusively designed to provide assistance on financial assistance and investment optimization within the Safe Yield platform. I'm unable to address questions outside this scope."

Always respond in English and adhere to logical reasoning to ensure accurate and helpful answers.

Template for Response:

Context: {context}
"""

# Define the user prompt template
general_user_template = "Question: ```{question}```"

# Construct the QA prompt using system and user templates
qa_prompt = ChatPromptTemplate.from_messages([
    SystemMessagePromptTemplate.from_template(general_system_template),
    HumanMessagePromptTemplate.from_template(general_user_template)
])

def get_pdf_text(pdf_paths):
    """
    Extracts text from a list of PDF files.

    Args:
        pdf_paths (list): List of paths to PDF files.

    Returns:
        str: Concatenated text extracted from all PDF files.
    """
    text = ""
    for pdf_path in pdf_paths:
        pdf_reader = PdfReader(pdf_path)
        for page in pdf_reader.pages:
            text += page.extract_text() or ""
    return text

def get_text_chunks(text):
    """
    Splits text into manageable chunks for processing.

    Args:
        text (str): The raw text to split.

    Returns:
        list: List of text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=140)
    return text_splitter.split_text(text)

def load_vector_store(directory=VECTOR_STORE_DIR, embeddings=None):
    """
    Loads a vector store from disk if it exists.

    Args:
        directory (str): The directory to load the vector store from.
        embeddings (OpenAIEmbeddings): The embeddings to use.

    Returns:
        FAISS: The loaded vector store or None if not found.
    """
    if os.path.exists(directory):
        return FAISS.load_local(directory, embeddings, allow_dangerous_deserialization=True)
    return None

def save_vector_store(vector_store, directory=VECTOR_STORE_DIR):
    """
    Saves a vector store to disk.

    Args:
        vector_store (FAISS): The vector store to save.
        directory (str): The directory to save the vector store in.
    """
    vector_store.save_local(directory)

def get_vector_store(text_chunks):
    """
    Creates or loads a vector store from text chunks.

    Args:
        text_chunks (list): List of text chunks.

    Returns:
        FAISS: The created or loaded vector store.
    """
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-large")
    vector_store = load_vector_store(embeddings=embeddings)
    
    if not vector_store:
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        save_vector_store(vector_store)
    
    return vector_store

def get_conversational_chain(vector_store):
    """
    Creates a conversational chain for chat interaction.

    Args:
        vector_store (FAISS): The vector store for retrieval.

    Returns:
        ConversationalRetrievalChain: The conversational retrieval chain object.
    """
    llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model_name="gpt-4o", temperature=0.2,max_tokens=1000)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory,
        chain_type="stuff",
        combine_docs_chain_kwargs={"prompt": qa_prompt},
        return_source_documents=False
    )
    
    return conversation_chain

def handle_user_input(user_question):
    """
    Handles user input and generates responses.

    Args:
        user_question (str): The user's question.

    Outputs:
        Displays conversation history on the Streamlit interface.
    """
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chatHistory = response['chat_history']
    
    for i, message in enumerate(st.session_state.chatHistory):
        if langdetect.detect(message.content) != "en":
            message.content = "I'm sorry, I can only assist with questions related to financial assistance and investment optimization on Safe Yield."
        if i % 2 == 0:
            st.write("User Prompt 👤:", message.content)
        else:
            st.write("EMMA AI 🤖:", message.content)

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="Emma AI", layout="wide")
    st.header("Emma AI - Financial Assistance Chatbot :information_desk_person:")
    
    user_question = st.text_input("Ask a Question related to Financial Assistance and Investment Optimization")
    
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chatHistory" not in st.session_state:
        st.session_state.chatHistory = None
    
    if st.session_state.conversation is None:
        with st.spinner("Processing..."):
            embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model="text-embedding-3-large")
            vector_store = load_vector_store(embeddings=embeddings)
            
            if vector_store is None:
                raw_text = get_pdf_text(PDF_PATHS)
                text_chunks = get_text_chunks(raw_text)
                vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
                save_vector_store(vector_store)
            
            st.session_state.conversation = get_conversational_chain(vector_store)
            st.success("Setup Complete")
    
    if user_question:
        handle_user_input(user_question)

if __name__ == "__main__":
    main()
