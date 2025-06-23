**Real-Time Fake News Detection with NLP and Ensemble ML**

**Project Overview**
This project is a Real-Time Fake News Detection system that leverages Natural Language Processing (NLP) and ensemble Machine Learning (ML) to classify news articles as fake or real. Developed with minimal use of ChatGPT or AI chatbots, it showcases hands-on expertise in combating misinformation. Key features include:

* Real-Time News Fetching: Integrates NewsAPI to fetch live news articles, enabling real-world predictions.
* High Accuracy: Achieves 89% accuracy with Random Forest and Logistic Regression, and 86% with XGBoost.
* NLP Pipeline: Preprocesses text using regex cleaning ([^a-zA-Z]), lowercasing, lemmatization, and stop-word removal.
* Feature Engineering: Uses TF-IDF vectorization with n-grams (1-2) and limits to 5000 features for efficiency.
* Output: Saves fetched news, predictions, confidence scores, and novelty scores to a .csv file (fake_news_prediction.csv).
* Journey: Improved from 52% to 89% accuracy through feature optimization and model tuning.

The project combines robust ML models with real-time data, making it a practical tool for addressing fake news.

**Dataset Description**

The WELFake dataset is a collection of news articles labeled as either real or fake, designed for training and evaluating machine learning models in fake news detection. It contains 72,134 articles, with 35,028 classified as real and 37,106 as fake. The dataset was created by merging four popular news datasets to provide a more robust and diverse collection. 

**Features**

* Real-Time API Integration: Fetches live news via NewsAPI for dynamic predictions.

* NLP Preprocessing:

* Cleans text with regex to keep only letters.

* Applies lowercasing, lemmatization (NLTK WordNet), and stop-word removal.


**Feature Engineering:**

TF-IDF vectorization with n-grams (1-2).
Limits to 5000 features with min_df=2, max_df=0.8.


**Ensemble ML Models:**

Random Forest: 89% accuracy
Logistic Regression: 89% accuracy
XGBoost: 86% accuracy


**Novelty Scoring:** Identifies unusual content using rare term frequencies and Real-time news fetching.
Output: Generates fake_news_prediction.csv with columns: Title, Prediction, Confidence (%), Novelty Score, Source.

**Project Structure**

Fake-News-Detection/
├── .env                          # Environment file for NewsAPI key (not committed)

├── Fake_News_Detection.ipynb     # Jupyter notebook with code (preprocessing, training, prediction)

├── fake_news_prediction.csv      # Output file with predictions

├── requirements.txt              # Python dependencies

├── WELFake_Dataset.csv           # Training dataset

├── .gitignore                    # Excludes sensitive files

├── README.md                     # This file

**Prerequisites**

Python 3.8+
NewsAPI Key: Obtain a free API key from https://newsapi.org/.
Jupyter Notebook: To run Fake_News_Detection.ipynb.
Dependencies:pip install -r requirements.txt



**Installation**

Clone the Repository:
git clone https://github.com/deep-khimani/Fake-News-Detection.git

cd Fake-News-Detection


**Install Dependencies:**
pip install -r requirements.txt

See requirements.txt for details (e.g., requests, newsapi-python, nltk, scikit-learn, xgboost, pandas, numpy, python-dotenv, joblib).

Download NLTK Data:
import nltk
nltk.download('stopwords')
nltk.download('wordnet')


**Set Up NewsAPI Key:**

Create a .env file in the project root:echo "NEWSAPI_KEY=your_newsapi_key_here" > .env


Replace your_newsapi_key_here with your NewsAPI key.
Important: .env is excluded via .gitignore to protect your key. Never commit it to GitHub.

**Run the Notebook:**

Open Fake_News_Detection.ipynb in Jupyter:jupyter notebook


Execute cells to preprocess data, train models, fetch news, and generate fake_news_prediction.csv.



**Usage**

Train Models:

In Fake_News_Detection.ipynb, run cells for preprocessing (WELFake_Dataset.csv), TF-IDF vectorization, and training (Random Forest, Logistic Regression, XGBoost).
Models and vectorizer are saved using joblib (e.g., xgb_model.pkl, tfidf_vectorizer.pkl).


**Fetch and Predict:**

Run cells to fetch live news via NewsAPI, preprocess, predict fake/real, and compute novelty scores.
Results are saved to fake_news_prediction.csv with columns: Title, Prediction, Confidence (%), Novelty Score, Source.


**Sample Output (fake_news_prediction.csv):**
Title,Prediction,Confidence (%),Novelty Score,Source
"New AI Breakthrough in Quantum Computing","Real","78.45","0.1234","BBC"
"Aliens Land in New York, Says Anonymous Source","Fake","65.32","0.5678","FakeNewsSite"



**Security**

API Key: Store your NewsAPI key in .env, not in Fake_News_Detection.ipynb. The notebook uses python-dotenv to load it securely.
GitHub: .gitignore excludes .env, model files (e.g., *.pkl), and sensitive outputs to prevent accidental leaks.
Regenerate Key if Exposed: If your key is shared, regenerate it at https://newsapi.org/ and update .env.

**Dependencies**
See requirements.txt for a complete list. Key packages:

requests: For HTTP requests.
newsapi-python: For NewsAPI integration.
nltk: For lemmatization and stop-words.
scikit-learn: For TF-IDF, models, and metrics.
xgboost: For XGBoost classifier.
pandas: For data handling and .csv output.
numpy: For numerical operations.
python-dotenv: For secure API key loading.
joblib: For model serialization.

**Limitations**

Model Accuracy: 89% accuracy is strong but may vary on live data due to differences from WELFake_Dataset.csv.
NewsAPI Limits: Free tier allows 100 requests/day. Consider paid plans or alternatives (e.g., GNews) for higher quotas.
Real-Time Constraints: API rate limits and processing time may affect real-time performance for large-scale use.
Dataset Bias: WELFake_Dataset.csv may not fully represent current news trends, impacting generalization.

**Future Improvements**

Ensemble Voting: Combine models using VotingClassifier for potentially higher accuracy.
BERT-Based Models: Use Hugging Face transformers for improved NLP performance.
Expanded Datasets: Incorporate additional datasets (e.g., LIAR, FakeNewsNet) for robustness.
Web Interface: Deploy via Flask or Streamlit for user-friendly access.
Advanced Novelty Detection: Integrate BERT-based anomaly detection or sentiment analysis.

**Contributing**
Contributions are welcome! Please:

Fork the repository.
Create a feature branch: git checkout -b feature-name.
Commit changes: git commit -m "Add feature".
Push to the branch: git push origin feature-name.
Submit a pull request.

**License**
This project is licensed under the MIT License. See LICENSE for details.

**Acknowledgments**
WELFake Dataset: Provided by Kaggle.
NewsAPI: For enabling real-time news fetching.
NLTK, scikit-learn, XGBoost: For robust NLP and ML tools.

**Contact**
For questions or feedback, reach out via [deepkhimani@gmail.com] or LinkedIn.
