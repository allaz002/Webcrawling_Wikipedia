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

    def __init__(self, vocabulary=None, max_features=1000, ngram_range=(1, 2), min_df=1, max_df=0.95):
        if vocabulary:
            self.count_vectorizer = CountVectorizer(
                vocabulary=vocabulary,
                ngram_range=(1, 1)
            )
        else:
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

        # Pfade zu Trainingsdaten
        self.model_path = self.config['NAIVEBAYES']['MODEL_PATH']
        self.vectorizer_path = self.config['NAIVEBAYES']['VECTORIZER_PATH']
        self.training_data_path = self.config['NAIVEBAYES']['TRAINING_DATA_PATH']

        # Lade Modus (tf_norm oder tfidf) - wie VectorSpaceSpider
        self.mode = self.config['NAIVEBAYES'].get('MODE', 'tf_norm').lower()

        # Lade Bayes-spezifische Gewichtungen
        self.bayes_title_weight = float(self.config['NAIVEBAYES']['BAYES_TITLE_WEIGHT'])
        self.bayes_heading_weight = float(self.config['NAIVEBAYES']['BAYES_HEADING_WEIGHT'])
        self.bayes_paragraph_weight = float(self.config['NAIVEBAYES']['BAYES_PARAGRAPH_WEIGHT'])

        # Lade oder trainiere Modell
        self.load_or_train_model()

        print(f"Naive Bayes Modell geladen/trainiert im {self.mode} Modus")
        self.write_to_report(f"Modell-Pfad: {self.model_path}\n")

    def load_or_train_model(self):
        """Lädt existierendes Modell oder trainiert neues"""
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            # Lade existierendes Modell
            with open(self.model_path, 'rb') as f:
                self.classifier = pickle.load(f)
            with open(self.vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            print("Existierendes Modell geladen")
        else:
            # Trainiere neues Modell
            self.train_model()

    def train_model(self):
        """Trainiert Naive Bayes Klassifikator mit JSON-Trainingsdaten"""
        print("Trainiere neues Naive Bayes Modell...")

        # Lade JSON-Trainingsdaten
        texts = []
        labels = []

        if os.path.exists(self.training_data_path):
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)

            for sample in training_data:
                processed_text = self.preprocess_text(sample['text'])
                texts.append(processed_text)
                labels.append(sample['label'])
        else:
            raise ValueError(f"Trainingsdaten nicht gefunden: {self.training_data_path}")

        if not texts:
            raise ValueError("Keine gültigen Trainingsdaten vorhanden!")

        # Initialisiere Vectorizer basierend auf Modus
        if self.mode == 'tf_norm':
            # Verwende Keyword-Vokabular für TF-Norm
            keywords = [kw.strip().lower() for kw in
                        self.config['KEYWORD']['KEYWORDS'].split(',')]
            vocabulary = {keyword: idx for idx, keyword in enumerate(keywords)}
            self.vectorizer = TfNormVectorizer(vocabulary=vocabulary)
            print("Verwende TF-Norm Vektorisierung")
        else:
            # TF-IDF mit gemeinsamen Parametern und IDF aus Trainingsdaten
            self.vectorizer = TfidfVectorizer(
                max_features=int(self.config['VECTORIZER_SETTINGS']['MAX_FEATURES']),
                ngram_range=(int(self.config['VECTORIZER_SETTINGS']['NGRAM_MIN']),
                             int(self.config['VECTORIZER_SETTINGS']['NGRAM_MAX'])),
                min_df=int(self.config['VECTORIZER_SETTINGS']['MIN_DF']),
                max_df=float(self.config['VECTORIZER_SETTINGS']['MAX_DF'])
            )
            print("Verwende TF-IDF Vektorisierung basierend auf Trainingsdaten")

        # Vektorisiere Texte
        X = self.vectorizer.fit_transform(texts)
        y = np.array(labels)

        # Trainiere Klassifikator
        self.classifier = MultinomialNB(alpha=0.1)  # Smoothing-Parameter
        self.classifier.fit(X, y)

        # Speichere Modell und Vectorizer
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        print(f"Modell trainiert mit {len(texts)} Beispielen")

    def calculate_text_relevance(self, text):
        """
        Berechnet Relevanz mittels Naive Bayes Klassifikation
        Gibt Wahrscheinlichkeit für Klasse 'relevant' zurück
        """
        if not text:
            return 0.0

        # Textvorverarbeitung
        processed_text = self.preprocess_text(text)

        if not processed_text:
            return 0.0

        try:
            # Transformiere Text mit trainiertem Vectorizer
            text_vector = self.vectorizer.transform([processed_text])

            # Berechne Wahrscheinlichkeiten für beide Klassen
            probabilities = self.classifier.predict_proba(text_vector)[0]

            # Index 1 ist Wahrscheinlichkeit für "relevant"
            relevance_probability = probabilities[1] if len(probabilities) > 1 else 0.0

            return float(relevance_probability)

        except Exception as e:
            return 0.0

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """
        Berechnet Bayes-basierte Relevanz des Elterndokuments
        Kombiniert Wahrscheinlichkeiten der einzelnen Textteile
        """
        # Berechne individuelle Wahrscheinlichkeiten
        title_prob = self.calculate_text_relevance(title)
        heading_prob = self.calculate_text_relevance(headings)
        paragraph_prob = self.calculate_text_relevance(paragraphs)

        # Gewichtete arithmetische Mittelung mit Bayes-spezifischen Gewichten
        weighted_prob = (
                self.bayes_title_weight * title_prob +
                self.bayes_heading_weight * heading_prob +
                self.bayes_paragraph_weight * paragraph_prob
        )

        total_weight = self.bayes_title_weight + self.bayes_heading_weight + self.bayes_paragraph_weight
        avg_prob = weighted_prob / total_weight if total_weight > 0 else 0.0

        # Alternative: Geometrisches Mittel für Wahrscheinlichkeiten
        if title_prob > 0 and heading_prob > 0 and paragraph_prob > 0:
            geometric_prob = (
                                     (title_prob ** self.bayes_title_weight) *
                                     (heading_prob ** self.bayes_heading_weight) *
                                     (paragraph_prob ** self.bayes_paragraph_weight)
                             ) ** (1 / total_weight)
        else:
            geometric_prob = 0.0

        # Kombiniere beide Ansätze (70% arithmetisch, 30% geometrisch)
        final_prob = 0.7 * avg_prob + 0.3 * geometric_prob

        return min(1.0, final_prob)