# Real-Time Fake News Detection with NLP and Ensemble ML

## Overview

This project implements a Real-Time Fake News Detection system using state-of-the-art Natural Language Processing (NLP) and ensemble Machine Learning (ML) methods. The system classifies news articles as fake or real and can analyze both historical and live news via NewsAPI. It was developed predominantly without the use of advanced AI assistants, emphasizing hands-on expertise in tackling misinformation.

## Key Features

- **Real-Time News Fetching**: Integrates NewsAPI for live news articles and real-world predictions
- **High Accuracy**: Achieves up to 89% accuracy (Random Forest & Logistic Regression), 86% (XGBoost), with rigorous model optimization
- **Full NLP Pipeline**: Cleans and preprocesses news with regex, lowercasing, lemmatization, and stop-word removal
- **Feature Engineering**: Utilizes TF-IDF (1-2 n-grams), capped at 5,000 features for speed and accuracy
- **Ensemble Models**: Includes Random Forest, Logistic Regression, and XGBoost, with model output blending possible
- **Novelty Scoring**: Identifies unusual/rare content in real-time using term-frequency analytics
- **Output**: Saves all predictions with confidence and source in `fake_news_prediction.csv`
- **Secure API Key Handling**: API keys are managed securely via `.env` (never committed)
- **Demo**: The app can be run in a notebook or deployed via Streamlit for a web-based user interface

## Dataset

**WELFake** is a curated dataset of 72,134 news articles (35,028 real / 37,106 fake) made by merging four popular sources, chosen for robust ML training and evaluation.

### Data columns:
- `title`
- `text`
- `label` (1 = Fake, 0 = Real)

## Features

### Real-Time API Integration
Fetches new articles dynamically for up-to-date classification.

### NLP Preprocessing
- Cleans with regex: only a-zA-Z retained
- Lowercases, lemmatizes (WordNet), removes stop-words using NLTK

### Feature Engineering
TF-IDF with 1-2 n-grams, max_features=5000, min_df=2, max_df=0.8

### Ensemble ML Models
- **Random Forest**: 89% accuracy
- **Logistic Regression**: 89%
- **XGBoost**: 86%

### Novelty Scoring
Measures article "unusualness" using rare term statistics and against real-time news.

### Output
All predictions including title, prediction, confidence %, novelty score, and source, saved to `fake_news_prediction.csv`.

## Prerequisites

- Python 3.8+
- NewsAPI Key – for live news fetching
- Jupyter Notebook (for .ipynb development)
- All required dependencies (see `requirements.txt`)

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/deep-khimani/Real-Time-Fake-News-Detection-with-NLP-and-Ensemble-ML.git
cd Real-Time-Fake-News-Detection-with-NLP-and-Ensemble-ML
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

Make sure `streamlit`, `requests`, `nltk`, `scikit-learn`, `xgboost`, `pandas`, `numpy`, `python-dotenv`, and `joblib` are included.

### 3. Download NLTK Data

In Python:

```python
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
```

### 4. Set Up NewsAPI Key

Create a `.env` file in your project root (not tracked by git):

```
NEWSAPI_KEY=your_newsapi_key_here
```

**⚠️ Never commit your `.env`! It is excluded via `.gitignore`.**

## Usage

### 1. Train Models

1. Open `Fake_News_Detection.ipynb` in Jupyter Notebook
2. Run preprocessing, TF-IDF vectorization, and model training cells
3. Models and vectorizers are saved using joblib (e.g., `xgb_model.pkl`)

### 2. Fetch & Predict

1. Run prediction cells to fetch live news via NewsAPI, preprocess, and predict labels and novelty
2. Results auto-save to: `fake_news_prediction.csv`

With columns: Title, Prediction, Confidence (%), Novelty Score, Source.

#### Example Output:

```csv
Title,Prediction,Confidence (%),Novelty Score,Source
"New AI Breakthrough in Quantum Computing","Real","78.45","0.1234","BBC"
"Aliens Land in New York, Says Anonymous Source","Fake","65.32","0.5678","FakeNewsSite"
```

## Security

- **API Key**: Always store your NewsAPI Key in `.env` (never in code or notebooks)
- **`.gitignore`**: Ensures `.env`, model `.pkl` files, and sensitive outputs are never committed
- **Rotate Keys**: Regenerate your key anytime at NewsAPI if it is accidentally shared

## Limitations

- **Accuracy**: 89% is strong but may drop for current, out-of-distribution news
- **API Rate Limits**: Free NewsAPI tier allows 100 requests/day
- **Dataset Bias**: WELFake may not fully match live news trends
- **Real-Time Constraints**: Large-scale or commercial use may require API upgrades and further optimization

## Future Improvements

- Ensemble voting with VotingClassifier for model blending
- BERT or Transformer-based NLP for better feature extraction
- Additional datasets (e.g., LIAR, FakeNewsNet) for robustness
- Streamlit/web deployment for public use
- Advanced novelty/anomaly detection using BERT or similar transformer architectures

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Commit changes: `git commit -m "Add feature"`
4. Push to the branch: `git push origin feature-name`
5. Submit a pull request

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgments

- **WELFake Dataset**: Provided by Kaggle
- **NewsAPI**: For real-time news access
- **NLTK, scikit-learn, XGBoost**: For NLP & ML modeling
- **Streamlit**: For interactive web apps

## Contact

Questions or feedback?  
📧 Email: deepkhimani@gmail.com

