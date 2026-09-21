import sys
import os

# Add project root to sys.path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from adapters.vector_store import FaissAdapter
from adapters.embeddings import HuggingFaceAdapter
from use_cases.routing import LearnFromFeedbackUseCase

def seed():
    vector_store = FaissAdapter()
    embedding_engine = HuggingFaceAdapter()
    feedback_uc = LearnFromFeedbackUseCase(vector_store, embedding_engine)
    
    # QUICK
    quicks = [
        "what time is it",
        "how tall is the eiffel tower",
        "give me a quick summary of hashmaps",
        "turn off the living room lights",
        "who is the president of france",
        "translate hello to spanish"
    ]
    for q in quicks:
        feedback_uc.execute(q, "quick")
        
    # THINK
    thinks = [
        "design a scalable architecture for a react app",
        "analyze this concept and explain the pros and cons",
        "plan a detailed itinerary for my trip to japan",
        "walk me through the logic of a neural network",
        "refactor this code and explain the changes"
    ]
    for t in thinks:
        feedback_uc.execute(t, "think")
        
    # CODE
    codes = [
        "write a python script to list files",
        "implement a new feature in my project",
        "fix the bug in router.py",
        "build a flask api",
        "refactor the authentication logic",
        "write a javascript function to sort an array",
        "create a new react component",
        "write code for sorting an array",
        "develop a coding solution",
        "help me refactor some functions",
        "write a code script",
        "add a feature to the codebase"
    ]
    for c in codes:
        feedback_uc.execute(c, "code")

    print("Database seeded successfully.")

if __name__ == "__main__":
    seed()
