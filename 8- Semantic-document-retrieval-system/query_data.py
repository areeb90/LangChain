from dotenv import load_dotenv
load_dotenv()

import argparse
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # must match create_database.py
    embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    results = db.similarity_search_with_relevance_scores(query_text, k=5)

    if not results:
        print("No results returned. Is the DB built? Run `python create_database.py`.")
        return

    for i, (doc, score) in enumerate(results, 1):
        print(f"[{i}] score={score:.3f} | source={doc.metadata.get('source')}")
        print(doc.page_content[:200], "...\n")

    # gentle threshold during dev
    if results[0][1] < 0.2:
        print("Unable to find matching results (score below threshold).")
        return

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE).format(
        context=context_text, question=query_text
    )

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response_text = llm.predict(prompt)

    sources = [doc.metadata.get("source") for doc, _ in results]
    print(f"\nResponse: {response_text}\nSources: {sources}")

if __name__ == "__main__":
    main()
