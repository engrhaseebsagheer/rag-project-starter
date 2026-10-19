# RAG project starter: chat with a PDF in one Python file

A minimal **RAG (retrieval-augmented generation) project** you can read in ten minutes and run in two. Point it at a PDF, ask a question, and get an answer that cites the page it came from. It uses the free tier of the Gemini API, so you don't need a credit card.

```
$ python rag.py sample.pdf "How long do customers have to ask for a refund?"
Indexing 3 chunks from sample.pdf ...

Customers have fourteen days from delivery to request a refund [1].

Sources: [1] page 2, [2] page 1, [3] page 3
```

Ask something the PDF doesn't cover and it says it could not find the answer, instead of making one up.

## Run it

1. Get a free API key at [Google AI Studio](https://aistudio.google.com/apikey).
2. Install and run:

```bash
pip install -r requirements.txt
export GEMINI_API_KEY=your_key          # Windows: set GEMINI_API_KEY=your_key
python rag.py sample.pdf "How long do customers have to ask for a refund?"
python rag.py your-notes.pdf "your question"
python rag.py your-notes.pdf            # or ask several questions interactively
```

Python 3.9+ works.

## How it works

The whole pipeline is about 100 lines in [`rag.py`](rag.py):

| Step | What happens | Settings |
| --- | --- | --- |
| 1. Load | Text is extracted page by page with `pypdf`, and every chunk keeps its page number | |
| 2. Chunk | Text is split into overlapping pieces so no sentence is lost at a boundary | 1,000 characters, 150 overlap |
| 3. Embed | Each chunk becomes a vector with `gemini-embedding-001` (task type `RETRIEVAL_DOCUMENT`) | 768 dimensions, normalised |
| 4. Search | The question is embedded (`RETRIEVAL_QUERY`) and compared with every chunk by cosine similarity | top 6 chunks |
| 5. Answer | The chunks go into the prompt as numbered sources and the model must cite them | temperature 0.2 |

A step-by-step explanation of each part is in the guide: [How to build a RAG app](https://www.kitsbyai.com/how-to-build-a-rag-app).

## Ideas to make it your own

- Store vectors in **Postgres + pgvector** so you don't re-embed the PDF every run
- Add **hybrid search** (keyword + vector) for exact terms like error codes
- Support **several PDFs** and show which file each citation came from
- Add a small **evaluation set**: 20 questions with known pages, and measure how often the right page is retrieved
- Wrap it in a web UI and **deploy it**, so you have a live link for your resume

More ideas: [RAG project ideas and guide](https://www.kitsbyai.com/rag-project) · [AI projects for students](https://www.kitsbyai.com/ai-projects-for-students) · [Final year projects for CSE](https://www.kitsbyai.com/final-year-projects-computer-science)

Preparing for interviews? [30 RAG interview questions](https://www.kitsbyai.com/rag-interview-questions) · [LLM interview questions](https://www.kitsbyai.com/llm-interview-questions)

## Want the full web app?

This repo is the learning version. **[Chat with your PDFs](https://www.kitsbyai.com/)** is the complete project: a Next.js web app with uploads, streaming answers, clickable citations, Supabase pgvector storage, 17 tests, a 20-minute free deploy guide, a video walkthrough of every file and 15 interview answers. [Try the live demo](https://demo.kitsbyai.com).

## License

MIT. Use it, change it, learn from it.
