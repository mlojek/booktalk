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
    parser.add_argument(
        "book_path",
        type=Path,
        help="Path to an EPUB book.",
    )
    parser.add_argument(
        "--model",
        type=str,
        help="Ollama LLM model to use (default: llama3.2).",
        default="llama3.2",
    )
    parser.add_argument(
        "--num_fragments",
        type=int,
        help="Number of relevant book fragment to use in RAG (default: 10).",
        default=10,
    )
    args = parser.parse_args()

    # load the book, chunk it and load into chroma DB
    book = UnstructuredEPubLoader(args.book_path).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)
    book_chunks = text_splitter.split_documents(book)
    vector_database = Chroma.from_documents(
        documents=book_chunks,
        collection_name="book",
        embedding=OllamaEmbeddings(model="mxbai-embed-large"),
    )
    chroma_retriever = vector_database.as_retriever()

    # create a prompt template and LLM chain
    model = OllamaLLM(model=args.model)
    prompt = ChatPromptTemplate.from_template(
        """
        You are a helpful literature expert, which helps people with understanding books.
        Here are some relevant fragments from the book: {fragments}.
        Here is the user's question: {question}.
        """
    )
    chain = prompt | model

    # in loop get the question from user and answer it using relevant context
    while True:
        question = input("Type your question (or q to quit): ")

        if question == "q":
            break

        fragments = chroma_retriever.invoke(question, k=args.num_fragments)

        answer = chain.invoke(
            {
                "fragments": fragments,
                "question": question,
            }
        )

        print("\n", answer, "\n")
