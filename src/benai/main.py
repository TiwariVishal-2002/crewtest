#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from benai.crew import Benai
from pathlib import Path

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

# This main file is intended to be a way for you to run your
# crew locally, so refrain from adding unnecessary logic into this file.
# Replace with inputs you want to test with, it will automatically
# interpolate any tasks and agents information

def run():
    """
    Run the crew.
    """
    # Define inputs for the crew
    inputs = {
        'topic': 'pinecode vs mem0',
        'current_year': str(datetime.now().year)
    }
    
    try:
        # Initialize and run the crew with the given inputs
        Benai().crew().kickoff(inputs=inputs)
    except Exception as e:
        # Raise an exception if an error occurs during crew execution
        raise Exception(f"An error occurred while running the crew: {e}")


def train():
    """
    Train the crew for a given number of iterations.
    """
    # Define inputs for training the crew
    inputs = {
        "topic": "pinecone vs mem0",
        'current_year': str(datetime.now().year)
    }
    try:
        # Train the crew with the specified number of iterations and filename
        Benai().crew().train(n_iterations=int(sys.argv[1]), filename=sys.argv[2], inputs=inputs)

        if result:
            # If training is successful, save the final article to a file
            output_path = Path("final_article.md")
            output_path.write_text(str(result), encoding="utf-8")
            print(f"\n✅ Final article saved to {output_path.resolve()}")

    except Exception as e:
        # Raise an exception if an error occurs during training
        raise Exception(f"An error occurred while training the crew: {e}")
def replay():
    """
    Replay the crew execution from a specific task.
    """
    try:
        # Replay the crew execution from the specified task ID
        Benai().crew().replay(task_id=sys.argv[1])

    except Exception as e:
        # Raise an exception if an error occurs during replaying
        raise Exception(f"An error occurred while replaying the crew: {e}")
def test():
    """
    Test the crew execution and returns the results.
    """
    # Define inputs for testing the crew
    inputs = {
        "topic": "pinecone vs mem0",
        "current_year": str(datetime.now().year)
    }
    
    try:
        # Test the crew with the specified number of iterations and evaluation LLM
        Benai().crew().test(n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs)

    except Exception as e:
        # Raise an exception if an error occurs during testing
        raise Exception(f"An error occurred while testing the crew: {e}")
