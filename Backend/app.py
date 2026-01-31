from flask import Flask, request, jsonify
from flask_cors import CORS
import mlflow.pyfunc

# -------------------- Flask Setup --------------------
app = Flask(__name__)
CORS(app)  # allow React to talk to Flask

# -------------------- Load MLflow Model --------------------
MODEL_URI = "models:/Flipkart_Sentiment_Best_Model/2"

model = mlflow.pyfunc.load_model(MODEL_URI)

# -------------------- Routes --------------------
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST","GET"])
def predict():
    data = request.get_json()

    if "review" not in data:
        return jsonify({"error": "No review text provided"}), 400

    review = data["review"]

    prediction = model.predict([review])[0]

    sentiment = "Positive" if prediction == 1 else "Negative"

    return jsonify({
        "review": review,
        "sentiment": sentiment
    })


# -------------------- Run App --------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
