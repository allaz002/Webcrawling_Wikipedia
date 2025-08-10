from webcrawler.base_spider import BaseTopicalSpider
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
import pickle
import os
import numpy as np


class NaiveBayesSpider(BaseTopicalSpider):
    """Naive Bayes Klassifikations-Strategie mit reinen Termhäufigkeiten"""

    name = 'naivebayes_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.model_path = self.config['NAIVEBAYES']['MODEL_PATH']
        self.vectorizer_path = self.config['NAIVEBAYES']['VECTORIZER_PATH']
        self.training_data_path = self.config['NAIVEBAYES']['TRAINING_DATA_PATH']

        self.load_or_train_model()
        print("Naive Bayes Modell mit TF geladen/trainiert")

    def select_training_labels(self, training_data):
        """Selektiert nur stark relevante (2) und irrelevante (0) Daten"""
        texts = []
        labels = []

        for sample in training_data:
            # Ignoriere mäßig relevante Daten (Label 1)
            if sample['label'] == 1:
                continue

            processed_text = self.preprocess_text(sample['text'])
            if processed_text:
                texts.append(processed_text)
                # Mappe Label 2 auf 1 für binäre Klassifikation
                labels.append(1 if sample['label'] == 2 else 0)

        return texts, labels

    def train_model(self, texts, labels):
        """Trainiert Naive Bayes Klassifikator"""
        if not texts:
            raise ValueError("Keine gültigen Trainingsdaten vorhanden!")

        # CountVectorizer mit Parametern aus Config
        vectorizer_config = self.config['NAIVEBAYES']
        self.vectorizer = CountVectorizer(
            max_features=int(vectorizer_config['MAX_FEATURES']),
            ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                         int(vectorizer_config['NGRAM_MAX'])),
            min_df=int(vectorizer_config['MIN_DF']),
            max_df=float(vectorizer_config['MAX_DF'])
        )

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

        n_relevant = sum(1 for l in labels if l == 1)
        n_irrelevant = len(labels) - n_relevant
        print(f"Trainiert mit {len(texts)} Beispielen ({n_relevant} relevant, {n_irrelevant} irrelevant)")

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