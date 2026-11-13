from pathlib import Path
import math

import joblib
import streamlit as st


MODEL_PATH = Path(__file__).parent / "good_bad_model.pkl"
EXAMPLES = {
    "Positive example": "I loved this product! The quality is excellent and I would buy it again.",
    "Negative example": "This was disappointing and poorly made. I regret spending money on it.",
    "Tricky example": "The design is beautiful, but the product stopped working after one day.",
}


st.set_page_config(page_title="Sentiment Classifier", page_icon="💬")


@st.cache_resource
def load_model():
    """Load the fitted pipeline once per Streamlit process."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


st.title("Review Sentiment Classifier")
st.write("Enter a review to classify it as Positive or Negative.")

st.sidebar.header("About")
st.sidebar.write(
    "**Dataset:** IMDB movie reviews; training-data details unknown\n\n"
    "**Model:** TF-IDF + LinearSVC pipeline\n\n"
    "**Test accuracy:** 83.2% | **F1:** 0.83 "
    "(388-review held-out sample)"
)

st.subheader("Try an example")
example_columns = st.columns(3)
for column, (label, example) in zip(example_columns, EXAMPLES.items()):
    if column.button(label, use_container_width=True):
        st.session_state["review_text"] = example

review = st.text_area(
    "Review",
    key="review_text",
    height=150,
    placeholder="Type or paste a review here...",
)

if st.button("Predict", type="primary", use_container_width=True):
    cleaned_review = review.strip()
    if len(cleaned_review) < 3:
        st.info("Please enter a little more text (at least 3 characters).")
    else:
        try:
            model = load_model()
            # The fitted pipeline performs its original TF-IDF preprocessing.
            prediction = model.predict([cleaned_review])[0]
            margin = float(model.decision_function([cleaned_review])[0])
        except (FileNotFoundError, AttributeError, ValueError, OSError) as exc:
            st.error(f"Could not make a prediction: {exc}")
        else:
            prediction_name = str(prediction).lower()
            is_positive = prediction_name in {"pos", "positive", "1", "true"}
            sentiment = "Positive" if is_positive else "Negative"
            # LinearSVC margins are not calibrated probabilities; this is a
            # bounded, margin-based confidence indicator for the UI.
            confidence = 1.0 / (1.0 + math.exp(-min(abs(margin), 20.0)))
            st.success(f"Prediction: **{sentiment}**")
            st.metric("Confidence (margin-based)", f"{confidence:.1%}")
            st.progress(confidence)
