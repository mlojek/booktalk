"""
Entrypoint script for the booktalk project.
"""

import argparse
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import UnstructuredEPubLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter


def main():
    """
    Entrypoint function for the booktalk project.
    This function is called when booktalk is ran from CLI.
    """
    parser = argparse.ArgumentParser(
        prog="booktalk",
        description="Run an interactive chat with AI to better understand a book.",
    )
    parser.add_argument("book_path", type=Path, help="Path to an EPUB book.")
    args = parser.parse_args()

    book = UnstructuredEPubLoader(args.book_path).load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)

    book_chunks = text_splitter.split_documents(book)

    vector_database = Chroma.from_documents(
        documents=book_chunks,
        collection_name="book",
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
    )

    chroma_retriever = vector_database.as_retriever()

    model = OllamaLLM(model="llama3.2")

    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful literature expert, which helps people with understanding books.
        Here are some relevant fragments from the book: {fragments}.
        Here is the user's question: {question}.
        """
    )

    chain = prompt | model

    while True:
        question = input("Type your question (or q to quit): ")

        if question == "q":
            break

        fragments = chroma_retriever.invoke(question, k=10)

        answer = chain.invoke(
            {
                "fragments": fragments,
                "question": question,
            }
        )
        print("\n", answer, "\n")
