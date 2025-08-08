from webcrawler.base_spider import BaseTopicalSpider
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import scipy.sparse as sp
import random


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


class VectorSpaceSpider(BaseTopicalSpider):
    """Vektorraum-Modell mit Cosinus-Ähnlichkeit"""

    name = 'vectorspace_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mode = self.config['VECTORSPACE'].get('MODE', 'tf_norm').lower()
        self.relevant_ratio = float(self.config['VECTORSPACE'].get('RELEVANT_RATIO', '1.0'))

        # Gewichtungen aus WEIGHTS Sektion
        self.title_weight = float(self.config['WEIGHTS']['TITLE_WEIGHT'])
        self.heading_weight = float(self.config['WEIGHTS']['HEADING_WEIGHT'])
        self.paragraph_weight = float(self.config['WEIGHTS']['PARAGRAPH_WEIGHT'])

        # Initialisiere Vectorizer basierend auf Modus mit separaten Settings
        if self.mode == 'tf_norm':
            vectorizer_config = self.config['VECTORIZER_TF_NORM']
            self.vectorizer = TfNormVectorizer(
                max_features=int(vectorizer_config['MAX_FEATURES']),
                ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                             int(vectorizer_config['NGRAM_MAX'])),
                min_df=int(vectorizer_config['MIN_DF']),
                max_df=float(vectorizer_config['MAX_DF'])
            )
            print("Verwende TF-Norm Vektorisierung")
        else:  # tfidf
            vectorizer_config = self.config['VECTORIZER_TFIDF']
            self.vectorizer = TfidfVectorizer(
                max_features=int(vectorizer_config['MAX_FEATURES']),
                ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                             int(vectorizer_config['NGRAM_MAX'])),
                min_df=int(vectorizer_config['MIN_DF']),
                max_df=float(vectorizer_config['MAX_DF'])
            )
            print("Verwende TF-IDF Vektorisierung")

        self.load_training_data()

        print(f"VectorSpace Spider initialisiert im {self.mode} Modus")
        self.write_to_report(f"Modus: {self.mode}\n")
        self.write_to_report(f"Relevant-Ratio: {self.relevant_ratio}\n")

    def load_training_data(self):
        """Lädt Trainingsdaten und erstellt Themen-Vektor"""
        with open(self.config['VECTORSPACE']['TRAINING_DATA_PATH'], 'r', encoding='utf-8') as f:
            training_data = json.load(f)

        # Separiere relevante und irrelevante Texte
        relevant_texts = []
        irrelevant_texts = []

        for sample in training_data:
            processed_text = self.preprocess_text(sample['text'])
            if processed_text:  # Nur nicht-leere Texte
                if sample['label'] == 1:
                    relevant_texts.append(processed_text)
                else:
                    irrelevant_texts.append(processed_text)

        # Bestimme Trainingsset basierend auf Modus und Ratio
        if self.relevant_ratio < 1.0:
            num_relevant = len(relevant_texts)
            num_irrelevant_needed = int(num_relevant * (1 - self.relevant_ratio) / self.relevant_ratio)

            # Sample irrelevante Dokumente
            sampled_irrelevant = (random.choices(irrelevant_texts, k=num_irrelevant_needed)
                                  if num_irrelevant_needed > len(irrelevant_texts)
                                  else random.sample(irrelevant_texts, num_irrelevant_needed))

            training_texts = relevant_texts + sampled_irrelevant
            print(f"Training mit {num_relevant} relevanten und {len(sampled_irrelevant)} irrelevanten Dokumenten")
        else:
            training_texts = relevant_texts
            print(f"Training nur mit {len(relevant_texts)} relevanten Dokumenten")

        # Fit Vectorizer und erstelle Themen-Vektor aus relevanten Dokumenten
        self.vectorizer.fit(training_texts)
        vectors = self.vectorizer.transform(relevant_texts)
        self.topic_vector = np.asarray(vectors.mean(axis=0)).reshape(1, -1)

    def calculate_text_relevance(self, text):
        """Berechnet Cosinus-Ähnlichkeit zwischen Text und Themenprofil"""
        if not text:
            return 0.0

        processed_text = self.preprocess_text(text)
        if not processed_text:
            return 0.0

        try:
            text_vector = self.vectorizer.transform([processed_text])
            if sp.issparse(text_vector):
                text_vector = text_vector.toarray()

            return float(cosine_similarity(text_vector, self.topic_vector)[0][0])
        except Exception:
            return 0.0

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """Berechnet gewichtete Relevanz des Parent-Textes"""
        # Berechne individuelle Ähnlichkeiten
        title_sim = self.calculate_text_relevance(title) if title else 0.0
        heading_sim = self.calculate_text_relevance(headings) if headings else 0.0
        paragraph_sim = self.calculate_text_relevance(paragraphs) if paragraphs else 0.0

        # Gewichtete Summe (Gewichte summieren sich zu 1.0)
        weighted_sim = (
                self.title_weight * title_sim +
                self.heading_weight * heading_sim +
                self.paragraph_weight * paragraph_sim
        )

        return min(1.0, weighted_sim)