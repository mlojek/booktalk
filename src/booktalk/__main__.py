"""
Entrypoint script for the booktalk project.
"""

import argparse
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


from .read_epub import read_epub_book


# TODO add logger with INFO because the initialization is slow and user might
# need some progress info
def main():
    """
    Entrypoint function for the booktalk project. This function is called when booktalk
    is ran from CLI.
    """
    print("Hello world!")

    parser = argparse.ArgumentParser(
        prog="booktalk",
        description="Run an interactive chat with AI to better understand a book.",
    )
    parser.add_argument("book_path", type=Path, help="Path to an EPUB book.")
    args = parser.parse_args()

    book_contents = read_epub_book(args.book_path)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=500)

    book = Document(
        page_content=book_contents,
        metadata={},
        id='the_book'
    )


    book_chunks = text_splitter.split_documents([book])


    vector_database = Chroma.from_documents(
        documents=book_chunks,
        collection_name='book',
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
        print('\n', answer, '\n')
