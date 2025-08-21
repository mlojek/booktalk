"""
EPUB file reading utilities.
"""

from pathlib import Path

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub
from langchain_core.documents import Document


def read_epub_book(path: Path) -> Document:
    """
    Read an epub book from a given file. Returns cleaned text as one string.

    Args:
        path (Path): Path to an EPUB file.

    Returns:
        Document: A langchain document with the contents of the book.
    """
    book = epub.read_epub(path)

    raw_text = ""

    # get only text items
    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            raw_text += item.get_body_content().decode()

    # remove all html tags
    soup = BeautifulSoup(raw_text, "html.parser")

    # also remove all anchors to remove the footnote numbers from text
    for anchor in soup.find_all("a"):
        anchor.decompose()

    book_contents = soup.get_text()
    book_title = book.get_metadata("DC", "title")[0][0]
    book_author = book.get_metadata("DC", "creator")[0][0]

    return Document(
        page_content=book_contents, metadata={"author": book_author}, id=book_title
    )
