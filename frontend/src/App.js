import { useState } from "react";
import "./App.css";

function App() {
  const [review, setReview] = useState("");
  const [sentiment, setSentiment] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const analyzeSentiment = async () => {
    if (!review.trim()) {
      setError("Please enter a review");
      return;
    }

    setLoading(true);
    setSentiment("");
    setError("");

    try {
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ review }),
      });

      const data = await response.json();

      if (response.ok) {
        setSentiment(data.sentiment);
      } else {
        setError(data.error || "Something went wrong");
      }
    } catch (err) {
      setError("Cannot connect to backend");
    }

    setLoading(false);
  };

  return (
    <div className="container">
      <h1>Flipkart Review Sentiment Analysis</h1>

      <textarea
        placeholder="Enter product review here..."
        value={review}
        onChange={(e) => setReview(e.target.value)}
      />

      <button onClick={analyzeSentiment} disabled={loading}>
        {loading ? "Analyzing..." : "Analyze Sentiment"}
      </button>

      {sentiment && (
        <div className={`result ${sentiment.toLowerCase()}`}>
          Sentiment: {sentiment}
        </div>
      )}

      {error && <div className="error">{error}</div>}
    </div>
  );
}

export default App;
