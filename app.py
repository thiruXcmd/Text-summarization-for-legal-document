from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import pipeline

app = Flask(__name__)
CORS(app)

# Load model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

@app.route("/summarize", methods=["POST"])
def summarize():
    data = request.get_json()
    text = data.get("text", "")
    if not text.strip():
        return jsonify({"summary": "Error: No text provided."}), 400

    try:
        # Truncate long input if needed
        if len(text.split()) > 1024:
            text = " ".join(text.split()[:1024])
        result = summarizer(text, max_length=150, min_length=50, do_sample=False)
        return jsonify({"summary": result[0]["summary_text"]})
    except Exception as e:
        return jsonify({"summary": f"Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
