# Booktalk
Interact with books like never before with this AI-powered chatbot!

I'm not a humanist, so back in high school while I got lots of As in maths and computer science, I also got a lot of Fs in polish. Main reason? I didn't read the readings. They just bored me. Not that I don't read any books - just not some archaic novels clearly written by mentally unwell people.

Now, back then LLMs were not a thing yet, but If I had to learn for my literature exams now, I'd surely use AI to help me out. Specifically, I'd make a chatbot with whom I could talk about the reading, and who would help me understand the reading on a deeper level.

So guess what? I made one!

## Tech stack
This project is made using `python`. I used `Ollama` to host LLMs locally so that I don't have to pay for OpenAI's API. Vector storage for RAG is done using `ChromaDB`. The project's logic is done in `LangChain`. I've also added an elegant and user-friendly web GUI with `Streamlit`.

## Installation
This project requires you to have [Ollama installed and running](https://ollama.com/download) on your system. Use python 3.11 or newer. Install the project by running:
```
make install
```
Then to run the app run:
```
booktalk
```

## Performance
I've tested this using a 3B llama3.2 with 4-bit quantization with satisfactory results. The chatbot answers questions about book's plot correctly and has no issues with hallucinations. Overall the performance is very nice given the low hardware requirements for this project.

## References
- Get free public domain EPUBs from Project Gutenberg [here](https://www.gutenberg.org).
- Streamlit has a great documentation with lots of practical examples [here](https://streamlit.io/playground).
- You can find a great tutorial video from Tech with Tim on youtube [here](https://www.youtube.com/watch?v=E4l91XKQSgw).
