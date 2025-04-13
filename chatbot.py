from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/chatbot', methods=['POST'])
def chatbot():
    # Get the question from the JSON payload
    data = request.get_json()
    question = data.get('question', '')
    
    # Process the question (this is where your logic goes)
    answer = process_question(question)
    
    # Return the answer as JSON
    return jsonify({'answer': answer})

def process_question(question):
    # Replace this with your actual logic or call to your ML model, etc.
    return "This is a sample answer for: " + question

if __name__ == '__main__':
    # Make sure your server is accessible (for example, on localhost port 5000)
    app.run(host='0.0.0.0', port=3000)