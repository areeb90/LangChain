from dotenv import load_dotenv
load_dotenv()  # read OPENAI_API_KEY from .env

from langchain_openai import OpenAIEmbeddings
from langchain.evaluation import load_evaluator

def main():
    # Embed a single word
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vec = embeddings.embed_query("apple")
    print(f"Vector length for 'apple': {len(vec)}")

    # Compare two words via embedding distance
    evaluator = load_evaluator(
        "pairwise_embedding_distance",
        embeddings=embeddings,  # supply the same embeddings instance
        distance_metric="cosine",  # optional; "cosine" is typical
    )
    words = ("apple", "iphone")
    result = evaluator.evaluate_string_pairs(
        prediction=words[0],
        prediction_b=words[1],
    )
    print(f"Comparing {words}: {result}")  # contains 'score' and metadata

if __name__ == "__main__":
    main()
