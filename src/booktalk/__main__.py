"""
Entrypoint script for the booktalk project.
"""

import argparse
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

from .read_epub import read_epub_book


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

        answer = chain.invoke(
            {
                "fragments": [],
                "question": question,
            }
        )
        print('\n', answer, '\n')
