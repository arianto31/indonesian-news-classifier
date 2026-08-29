# 📰 Indonesian News Classification — Dashboard

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://indonesian-news-classifier.streamlit.app/)

A Streamlit dashboard for presenting the complete Indonesian news topic classification pipeline:

**data cleaning → EDA → model performance**, along with a real-time news category prediction feature.

## Project Structure

```text
streamlit_app/

├── app.py                          # Main page (Overview)
├── utils.py                        # Shared functions: data/model loading, preprocessing, styling
├── requirements.txt
├── data/
│   ├── df_final.csv                # Final cleaned dataset (9,807 articles, 11 categories)
│   └── metrics.json                # Precomputed model metrics (classification report, confusion matrix)
├── model/
│   └── final_pipeline.pkl          # Trained Word + Character TF-IDF + LinearSVC pipeline
└── pages/

    ├── 1_🧹_Data_Cleaning.py       # Data cleaning process insights
    ├── 2_📊_EDA.py                 # Interactive exploratory data analysis
    ├── 3_🤖_Model_Performance.py   # Experiment comparison & final model evaluation
    └── 4_📰_News_Prediction.py     # New article prediction feature (text input/upload)
```

## How to Run

1. **Create a virtual environment (optional but recommended)**

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application**

   ```bash
   streamlit run app.py
   ```

4. The application will automatically open in your browser at `http://localhost:8501`.

   Navigation between pages is available through the **left sidebar**.

## Technical Notes

* **NLTK stopwords** are automatically downloaded when the *News Prediction* page is launched for the first time
  (requires an internet connection once). If the environment has no internet access, download them manually:

  ```python
  import nltk; nltk.download('stopwords')
  ```

* **Model** (`model/final_pipeline.pkl`) was trained using `scikit-learn`. If the `scikit-learn` version in your
  environment differs from the version used during training, an `InconsistentVersionWarning` may appear. This can
  safely be ignored as long as the predictions remain reasonable; however, matching the training version is
  recommended for production environments.

* **metrics.json** contains evaluation results from the held-out test set (20%, stratified split, `random_state=42`)
  using a pipeline with a configuration identical to `final_pipeline.pkl`. Therefore, the metrics displayed on the
  Model Performance page represent the model's true generalization performance rather than evaluation on the same
  data used for training.

## Key Features

| Page                 | Content                                                                  |
| -------------------- | ------------------------------------------------------------------------ |
| 🏠 Overview          | Summary KPIs, category distribution, key insights                        |
| 🧹 Data Cleaning     | Data cleaning funnel, preprocessing stages, 29→11 category consolidation |
| 📊 EDA               | Text length distribution, word frequency, word clouds by category        |
| 🤖 Model Performance | Algorithm comparison, classification report, confusion matrix            |
| 📰 News Prediction   | Text input/file upload → category prediction + confidence score          |
