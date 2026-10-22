
import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


BASE_DIR = Path(__file__).resolve().parent
POLICIES_PATH = BASE_DIR / "fraud_policies.json"
QUERIES_PATH = BASE_DIR / "retrieval_queries.json"


def load_policies():
    """Load policy records from the JSON file."""
    with POLICIES_PATH.open("r", encoding="utf-8") as file:
        policies = json.load(file)

    if not isinstance(policies, list):
        raise ValueError("fraud_policies.json must contain a list.")

    return policies


def build_retriever(policies, min_score=0.15):
    """Build a TF-IDF retriever and return a policy-search function."""
    if not policies:
        raise ValueError("At least one policy is required.")

    documents = [
        f"{policy['title']} {policy['text']}"
        for policy in policies
    ]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
    )
    policy_vectors = vectorizer.fit_transform(documents)

    def retrieve_policy(query):
        """Return the best matching policy, or None if the score is too low."""
        if not isinstance(query, str) or not query.strip():
            return None

        query_vector = vectorizer.transform([query])
        scores = cosine_similarity(query_vector, policy_vectors).ravel()

        best_index = int(scores.argmax())
        best_score = float(scores[best_index])

        if best_score < min_score:
            return None

        return {
            "policy": policies[best_index],
            "score": best_score,
        }

    return retrieve_policy


def run_retrieval_tests():
    """Run the supplied retrieval queries and print their results."""
    policies = load_policies()
    retrieve_policy = build_retriever(policies)

    with QUERIES_PATH.open("r", encoding="utf-8") as file:
        queries = json.load(file)

    correct = 0

    for item in queries:
        result = retrieve_policy(item["query"])
        actual_policy = result["policy"]["policy_id"] if result else None
        expected_policy = item["expected_policy"]
        passed = actual_policy == expected_policy

        if passed:
            correct += 1

        print(
            f"Query: {item['query']}\n"
            f"Expected: {expected_policy} | "
            f"Actual: {actual_policy} | "
            f"Score: {result['score']:.3f}" if result else
            f"Query: {item['query']}\n"
            f"Expected: {expected_policy} | Actual: None | Score: 0.000",
            f"| {'PASS' if passed else 'CHECK'}",
        )
        print("-" * 60)

    print(f"Retrieval tests passed: {correct}/{len(queries)}")


if __name__ == "__main__":
    run_retrieval_tests()