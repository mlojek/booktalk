"""
Entrypoint script for the booktalk project.
"""

import tempfile
import time

import streamlit as st
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.web import cli


def main():
    """
    This function is called when user uses 'booktalk' command in CLI.
    It launches this script with streamlit.
    """
    cli.main_run(["src/booktalk/__main__.py"])


@st.cache_resource
def load_and_vectorize_book(book_bytes: str):
    # Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp:
        tmp.write(book_bytes.getbuffer())
        tmp_path = tmp.name

    # load the book, chunk it and load into chroma DB
    book = UnstructuredEPubLoader(tmp_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    book_chunks = text_splitter.split_documents(book)
    vector_database = Chroma.from_documents(
        documents=book_chunks,
        collection_name="book",
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
    )
    chroma_retriever = vector_database.as_retriever()

    return chroma_retriever


@st.cache_resource
def initialize_llm_chain():
    # create a prompt template and LLM chain
    model = OllamaLLM(model="llama3.2")
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful literature expert, which helps people with understanding books.
        Please act like you had been given the entire book, not only fragments.
        Also please avoid giving your answer as a one, continous block of text.
        Here are some relevant fragments from the book: {fragments}.
        Here is the user's question: {question}.
        """
    )
    chain = prompt | model

    return chain


if __name__ == "__main__":
    book_bytes = st.file_uploader("Drop in your book in epub format.", type="epub")

    if book_bytes:
        chroma_retriever = load_and_vectorize_book(book_bytes)

        chain = initialize_llm_chain()

        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Let's start chatting! 👇"}
            ]

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # Accept user input
        if prompt := st.chat_input("What is up?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)

            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                full_response = ""
                fragments = chroma_retriever.invoke(prompt, k=10)
                assistant_response = chain.invoke(
                    {
                        "fragments": fragments,
                        "question": prompt,
                    }
                )
                # Simulate stream of response with milliseconds delay
                for chunk in assistant_response.split():
                    full_response += chunk + " "
                    time.sleep(0.05)
                    # Add a blinking cursor to simulate typing
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
            # Add assistant response to chat history
            st.session_state.messages.append(
                {"role": "assistant", "content": full_response}
            )
