import os
from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from langchain_helper import create_chain

# Load env
load_dotenv()

app = Flask(__name__)
chain = create_chain()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/chatbot", methods=["POST"])
def chatbot():
    user_input = request.json.get("input", "")
    try:
        result = chain.run(user_input)
        return jsonify({"response": result})
    except Exception as e:
        return jsonify({"response": f"Error: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
