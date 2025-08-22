"""
Entrypoint script for the booktalk project.
"""

import tempfile
import time

import streamlit as st
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.base import RunnableSequence
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from streamlit.runtime.uploaded_file_manager import UploadedFile
from streamlit.web import cli


def main():
    """
    This function is called when user uses 'booktalk' command in CLI.
    It launches this script with streamlit.
    """
    cli.main_run(["src/booktalk/__main__.py"])


@st.cache_resource
def load_and_vectorize_book(book_bytes: UploadedFile) -> VectorStoreRetriever:
    """
    Given a book read from the streamlit GUI, load it into ChromaDB vector store
    and return a retriever object.

    Args:
        book_bytes (UploadedFile): An Epub file read from the disk. The UploadedFile
            extends BytesIO type and can be treated as a file handle.

    Returns:
        VectorStoreRetriever: Chroma object retriever for RAG usage.
    """
    # save the uploaded file to a temporary file so it can be read by langchain
    with tempfile.NamedTemporaryFile(delete=False, suffix=".epub") as tmp_file:
        tmp_file.write(book_bytes.getbuffer())
        tmp_path = tmp_file.name

    book = UnstructuredEPubLoader(tmp_path).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    book_chunks = text_splitter.split_documents(book)

    vector_database = Chroma.from_documents(
        documents=book_chunks,
        collection_name="book",
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
    )

    return vector_database.as_retriever()


@st.cache_resource
def initialize_llm_chain() -> RunnableSequence:
    """
    Create a LLM chain using a prompt template and Ollama LLM.

    Returns:
        RunnableSequence: A runnable LLM chain.
    """
    model = OllamaLLM(model="llama3.2")
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful literature expert, which helps people with understanding books.
        Please act like you had been given the entire book, not only fragments.
        Also please avoid giving your answer as a one, continous block of text.
        Instead, try to split your answer into a few paragraphs.
        Here are some relevant fragments from the book: {fragments}.
        Here is the user's question: {question}.
        """
    )
    chain = prompt | model

    return chain


if __name__ == "__main__":
    st.title('Booktalk')
    st.write("Don't want to read books? Let this AI chatbot explain them to you instead!")

    # read user-provided epub book
    book_bytes = st.file_uploader("Drop in your book in epub format.", type="epub")

    if book_bytes:
        # load the book into ChromaDB vector store
        chroma_retriever = load_and_vectorize_book(book_bytes)

        # init LLM chain
        chain = initialize_llm_chain()

        # init chat history
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "Hi, let's talk about your book!",},
            ]

        # display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # accept question from the user
        if prompt := st.chat_input("Ask me about the book."):
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
