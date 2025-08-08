from webcrawler.base_spider import BaseTopicalSpider
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
import pickle
import os
import json
import numpy as np


class TfNormVectorizer:
    """TF-Norm Vektorisierer: f_t,d / max_t' f_t',d"""

    def __init__(self, max_features=1000, ngram_range=(1, 2), min_df=1, max_df=0.95):
        self.count_vectorizer = CountVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df
        )
        self.scaler = MaxAbsScaler()

    def fit(self, documents):
        count_matrix = self.count_vectorizer.fit_transform(documents)
        self.scaler.fit(count_matrix)
        return self

    def transform(self, documents):
        count_matrix = self.count_vectorizer.transform(documents)
        return self.scaler.transform(count_matrix)

    def fit_transform(self, documents):
        count_matrix = self.count_vectorizer.fit_transform(documents)
        return self.scaler.fit_transform(count_matrix)


class NaiveBayesSpider(BaseTopicalSpider):
    """Naive Bayes Klassifikations-Strategie"""

    name = 'naivebayes_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_path = self.config['NAIVEBAYES']['MODEL_PATH']
        self.vectorizer_path = self.config['NAIVEBAYES']['VECTORIZER_PATH']
        self.training_data_path = self.config['NAIVEBAYES']['TRAINING_DATA_PATH']

        self.mode = self.config['NAIVEBAYES'].get('MODE', 'tfidf').lower()

        # Gewichtungen aus WEIGHTS Sektion
        self.title_weight = float(self.config['WEIGHTS']['TITLE_WEIGHT'])
        self.heading_weight = float(self.config['WEIGHTS']['HEADING_WEIGHT'])
        self.paragraph_weight = float(self.config['WEIGHTS']['PARAGRAPH_WEIGHT'])

        self.load_or_train_model()

        print(f"Naive Bayes Modell geladen/trainiert im {self.mode} Modus")
        self.write_to_report(f"Modell-Pfad: {self.model_path}\n")

    def load_or_train_model(self):
        """Lädt existierendes Modell oder trainiert neues"""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            with open(self.model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Existierendes Modell geladen")
        else:
            self.train_model()

    def train_model(self):
        """Trainiert Naive Bayes Klassifikator"""
        print("Trainiere neues Naive Bayes Modell...")

        # Lade und verarbeite Trainingsdaten
        texts = []
        labels = []

        with open(self.training_data_path, 'r', encoding='utf-8') as f:
            training_data = json.load(f)

        for sample in training_data:
            processed_text = self.preprocess_text(sample['text'])
            if processed_text:  # Nur nicht-leere Texte
                texts.append(processed_text)
                labels.append(sample['label'])

        if not texts:
            raise ValueError("Keine gültigen Trainingsdaten vorhanden!")

        # Initialisiere Vectorizer basierend auf Modus mit separaten Settings
        if self.mode == 'tf_norm':
            vectorizer_config = self.config['VECTORIZER_NB_TF_NORM']
            self.vectorizer = TfNormVectorizer(
                max_features=int(vectorizer_config['MAX_FEATURES']),
                ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                             int(vectorizer_config['NGRAM_MAX'])),
                min_df=int(vectorizer_config['MIN_DF']),
                max_df=float(vectorizer_config['MAX_DF'])
            )
            print("Verwende TF-Norm Vektorisierung")
        else:  # tfidf
            vectorizer_config = self.config['VECTORIZER_NB_TFIDF']
            self.vectorizer = TfidfVectorizer(
                max_features=int(vectorizer_config['MAX_FEATURES']),
                ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                             int(vectorizer_config['NGRAM_MAX'])),
                min_df=int(vectorizer_config['MIN_DF']),
                max_df=float(vectorizer_config['MAX_DF'])
            )
            print("Verwende TF-IDF Vektorisierung")

        # Vektorisiere und trainiere
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        self.classifier = MultinomialNB(alpha=0.1)
        self.classifier.fit(X, y)

        # Speichere Modell
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        print(f"Modell trainiert mit {len(texts)} Beispielen")

    def calculate_text_relevance(self, text):
        """Berechnet Relevanz mittels Naive Bayes Klassifikation"""
        if not text:
            return 0.0

        processed_text = self.preprocess_text(text)
        if not processed_text:
            return 0.0

        try:
            text_vector = self.vectorizer.transform([processed_text])
            probabilities = self.classifier.predict_proba(text_vector)[0]
            # Index 1 ist Wahrscheinlichkeit für "relevant"
            return float(probabilities[1] if len(probabilities) > 1 else 0.0)
        except Exception:
            return 0.0

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """Berechnet gewichtete Relevanz des Elterndokuments"""
        # Berechne individuelle Wahrscheinlichkeiten
        title_prob = self.calculate_text_relevance(title) if title else 0.0
        heading_prob = self.calculate_text_relevance(headings) if headings else 0.0
        paragraph_prob = self.calculate_text_relevance(paragraphs) if paragraphs else 0.0

        # Gewichtete Summe (Gewichte summieren sich zu 1.0)
        weighted_prob = (
                self.title_weight * title_prob +
                self.heading_weight * heading_prob +
                self.paragraph_weight * paragraph_prob
        )

        return min(1.0, weighted_prob)