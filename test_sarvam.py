"""Verify extract_topics works after the fix."""
import logging
logging.basicConfig(level=logging.INFO)
from embedder import extract_topics

result = extract_topics("Unit 1: Linear Algebra - Matrices, Determinants, Eigenvalues. Unit 2: Calculus - Differentiation, Integration, Partial derivatives. Unit 3: Probability - Random variables, Distributions, Bayes theorem.")
print(f"Topics: {result}")
print(f"Count: {len(result)}")
assert len(result) > 0, "FAILED: No topics extracted!"
print("SUCCESS!")
