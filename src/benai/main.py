"""Main file to run the crew with Flask endpoint."""
import sys
import warnings
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify

from benai.crew import Benai

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

app = Flask(__name__)

@app.route('/run', methods=['POST'])
def run_crew():
    """Endpoint to run the crew."""
    try:
        data = request.get_json()
        topic = data.get('topic', 'pinecode vs mem0')
        inputs = {
            'topic': topic,
            'current_year': str(datetime.now().year)
        }
        result = Benai().crew().kickoff(inputs=inputs)
        return jsonify({'result': str(result)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train_crew():
    """Endpoint to train the crew."""
    try:
        data = request.get_json()
        topic = data.get('topic', 'pinecone vs mem0')
        n_iterations = data.get('n_iterations', 10)
        filename = data.get('filename', 'training_data.json')
        inputs = {
            "topic": topic,
            'current_year': str(datetime.now().year)
        }
        result = Benai().crew().train(n_iterations=n_iterations, filename=filename, inputs=inputs)
        return jsonify({'result': str(result)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/replay', methods=['POST'])
def replay_crew():
    """Endpoint to replay the crew execution."""
    try:
        data = request.get_json()
        task_id = data.get('task_id')
        Benai().crew().replay(task_id=task_id)
        return jsonify({'status': 'Replay completed'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/test', methods=['POST'])
def test_crew():
    """Endpoint to test the crew execution."""
    try:
        data = request.get_json()
        topic = data.get('topic', 'pinecone vs mem0')
        n_iterations = data.get('n_iterations', 10)
        eval_llm = data.get('eval_llm', 'default_eval_llm')
        inputs = {
            "topic": topic,
            "current_year": str(datetime.now().year)
        }
        result = Benai().crew().test(n_iterations=n_iterations, eval_llm=eval_llm, inputs=inputs)
        return jsonify({'result': str(result)}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
