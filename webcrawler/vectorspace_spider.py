from webcrawler.base_spider import BaseTopicalSpider
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import scipy.sparse as sp
import random
from sklearn.preprocessing import normalize

class VectorSpaceSpider(BaseTopicalSpider):
    """Vektorraum-Modell mit Cosinus-Ähnlichkeit"""

    name = 'vectorspace_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mode = self.config['VECTORSPACE'].get('MODE', 'tf').lower()
        self.relevant_ratio = float(self.config['VECTORSPACE'].get('RELEVANT_RATIO', '1.0'))

        # Bei TF-Modus immer nur relevante Dokumente verwenden
        if self.mode == 'tf':
            self.relevant_ratio = 1

        # Gewichtungen aus WEIGHTS Sektion
        self.title_weight = float(self.config['WEIGHTS']['TITLE_WEIGHT'])
        self.heading_weight = float(self.config['WEIGHTS']['HEADING_WEIGHT'])
        self.paragraph_weight = float(self.config['WEIGHTS']['PARAGRAPH_WEIGHT'])

        # Initialisiere Vectorizer basierend auf Modus
        if self.mode == 'tf':
            # Reine Termhäufigkeiten ohne Normalisierung
            vectorizer_config = self.config['VECTORIZER_TF']
            self.vectorizer = CountVectorizer(
                max_features=int(vectorizer_config['MAX_FEATURES']),
                ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                             int(vectorizer_config['NGRAM_MAX'])),
                min_df=int(vectorizer_config['MIN_DF']),
                max_df=float(vectorizer_config['MAX_DF'])
            )
        else:  # tfidf
            # TF-IDF ohne automatische Normalisierung
            vectorizer_config = self.config['VECTORIZER_TFIDF']
            self.vectorizer = TfidfVectorizer(
                max_features=int(vectorizer_config['MAX_FEATURES']),
                ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                             int(vectorizer_config['NGRAM_MAX'])),
                min_df=int(vectorizer_config['MIN_DF']),
                max_df=float(vectorizer_config['MAX_DF']),
                norm=None
            )

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

        # Bestimme Trainingsset basierend auf Ratio
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


        self.vectorizer.fit(training_texts)
        vectors = self.vectorizer.transform(relevant_texts)

        vectors = normalize(vectors, norm='l2', axis=1)
        topic_vec = np.asarray(vectors.mean(axis=0)).reshape(1, -1)

        self.topic_vector = normalize(topic_vec, norm='l2', axis=1)

    def calculate_text_relevance(self, text):
        """Berechnet Cosinus-Ähnlichkeit zwischen Text und Themenprofil"""
        if not text:
            return 0.0

        processed_text = self.preprocess_text(text)
        if not processed_text:
            return 0.0

        try:
            text_vector = self.vectorizer.transform([processed_text])  # bleibt sparse
            if hasattr(text_vector, "nnz") and text_vector.nnz == 0:
                return 0.0

            # cosine_similarity normalisiert intern, kein extra normalize nötig
            return float(cosine_similarity(text_vector, self.topic_vector)[0, 0])
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