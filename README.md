# Review Sentiment Classifier

A small Streamlit app that classifies a text review as **Positive** or
**Negative** using the supplied fitted scikit-learn pipeline.

## Problem

Given a written review, predict whether its sentiment is positive or negative.

## Dataset

The task is binary sentiment classification of movie reviews into positive and
negative labels. I trained the model myself, but the composition and size of
the data used to train the saved pipeline are unknown.

The evaluation sample was drawn from the IMDB dataset. Its training-data
details should not be inferred from that evaluation sample.

## Approach

`good_bad_model.pkl` is a fitted scikit-learn `Pipeline` containing:

1. `TfidfVectorizer` with its learned vocabulary and default word-level
   preprocessing.
2. `LinearSVC` for binary classification.

The app loads this pipeline without refitting it. This means the app uses the
same preprocessing and learned TF-IDF vocabulary as training.

## Result

On a held-out sample of 388 IMDB reviews (191 negative and 197 positive), the
model achieved:

- Accuracy: **83.2%** (`0.8325`)
- F1 score: **0.830**
- Per-class F1: **0.83** for both `neg` and `pos`
- Majority-class baseline: **50.8%**

The displayed confidence is a bounded, margin-based indicator derived from
`LinearSVC.decision_function`; it is not a calibrated probability.

## Limitations

- The evaluation uses a relatively small held-out sample of 388 reviews.
- LinearSVC with TF-IDF is a bag-of-words approach and can miss sarcasm,
  context, and negation.

## Run locally

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL shown by Streamlit, usually
`http://localhost:8501`.

## Live demo

Live demo: <https://imdb-sentiment-analysis0.streamlit.app/>

## Streamlit Community Cloud deployment

1. Create a GitHub repository and add the files using the structure below.
2. Make sure `good_bad_model.pkl` is committed to the repository. Do not
   commit credentials or other sensitive files.
3. Go to <https://share.streamlit.io/> and sign in with GitHub.
4. Select **Create app**, then choose the repository, branch, and
   `app.py` as the main file.
5. Click **Deploy**. Community Cloud installs the pinned packages from
   `requirements.txt`.
6. After deployment, open the generated `streamlit.app` URL and test all
   three example buttons plus a custom review.

## Repository structure

```text
sentiment-streamlit-app/
├── app.py
├── good_bad_model.pkl
├── requirements.txt
└── README.md
```
