# Review Sentiment Classifier

A small Streamlit app that classifies a text review as **Positive** or
**Negative** using the supplied fitted scikit-learn pipeline.

## Problem

Given a written review, predict whether its sentiment is positive or negative.

## Dataset

The model was trained on the **Large Movie Review Dataset v1.0**, also known
as the **IMDB Dataset**, from the Stanford AI Lab.

- Total labeled reviews: 50,000
- Training reviews: 25,000
- Testing reviews: 25,000
- Labels: `pos` and `neg`
- Review type: movie reviews
- Official download:
  <https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz>

## Approach

`good_bad_model.pkl` is a fitted scikit-learn `Pipeline` containing:

1. `TfidfVectorizer` with its learned vocabulary and default word-level
   preprocessing.
2. `LinearSVC` for binary classification.

The app loads this pipeline without refitting it. This means the app uses the
same preprocessing and learned TF-IDF vocabulary as training.

## Result

Test accuracy was not provided with the model artifact. Add the measured
accuracy here when available.

The displayed confidence is a bounded, margin-based indicator derived from
`LinearSVC.decision_function`; it is not a calibrated probability.

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

Live demo link: `https://your-app-name.streamlit.app`

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
