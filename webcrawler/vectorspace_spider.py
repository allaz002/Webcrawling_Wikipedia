from webcrawler.base_spider import BaseTopicalSpider
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import pickle
import os


class VectorSpaceSpider(BaseTopicalSpider):
    """Vektorraum-Modell mit Cosinus-Ähnlichkeit"""

    name = 'vectorspace_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Pfade aus Config lesen
        self.model_path = self.config['VECTORSPACE']['MODEL_PATH']
        self.vectorizer_path = self.config['VECTORSPACE']['VECTORIZER_PATH']
        self.training_data_path = self.config['VECTORSPACE']['TRAINING_DATA_PATH']

        # IDF-Trainingsmischung aus Config
        self.idf_ratio_irrelevant = float(self.config['VECTORSPACE'].get('IDF_RATIO_IRRELEVANT', 0.33))
        self.idf_ratio_moderate = float(self.config['VECTORSPACE'].get('IDF_RATIO_MODERATE', 0.33))
        self.idf_ratio_relevant = float(self.config['VECTORSPACE'].get('IDF_RATIO_RELEVANT', 0.34))

        # Validierung der IDF-Ratios
        ratio_sum = self.idf_ratio_irrelevant + self.idf_ratio_moderate + self.idf_ratio_relevant
        if abs(ratio_sum - 1.0) > 0.001:
            raise ValueError(f"IDF-Ratios summieren sich nicht zu 1.0: {ratio_sum}")

        self.load_or_train_model()

        # Setze topic_vector aus classifier für VectorSpace
        if hasattr(self, 'classifier'):
            self.topic_vector = self.classifier

        print("VectorSpace Spider mit TF-IDF initialisiert")

    def select_training_labels(self, training_data):
        """Behält alle drei Klassen für IDF-Training und Topic-Vektor"""
        irrelevant_texts = []
        moderate_texts = []
        relevant_texts = []

        for sample in training_data:
            processed_text = self.preprocess_text(sample['text'])
            if processed_text:
                if sample['label'] == 0:
                    irrelevant_texts.append(processed_text)
                elif sample['label'] == 1:
                    moderate_texts.append(processed_text)
                elif sample['label'] == 2:
                    relevant_texts.append(processed_text)

        return (irrelevant_texts, moderate_texts, relevant_texts), None

    def train_model(self, texts_tuple, labels):
        """Trainiert TF-IDF Vectorizer und erstellt Topic-Vektor"""
        irrelevant_texts, moderate_texts, relevant_texts = texts_tuple

        print(f"Trainingsdaten: {len(relevant_texts)} relevant, "
              f"{len(moderate_texts)} mäßig, {len(irrelevant_texts)} irrelevant")

        # Erstelle Trainingsmischung gemäß IDF-Ratios
        training_corpus = []
        total_samples = 100
        n_irrelevant = int(total_samples * self.idf_ratio_irrelevant)
        n_moderate = int(total_samples * self.idf_ratio_moderate)
        n_relevant = int(total_samples * self.idf_ratio_relevant)

        # Over/Undersampling für ausgewogene Mischung
        if irrelevant_texts:
            for i in range(n_irrelevant):
                training_corpus.append(irrelevant_texts[i % len(irrelevant_texts)])

        if moderate_texts:
            for i in range(n_moderate):
                training_corpus.append(moderate_texts[i % len(moderate_texts)])

        if relevant_texts:
            for i in range(n_relevant):
                training_corpus.append(relevant_texts[i % len(relevant_texts)])

        # TF-IDF Vectorizer
        vectorizer_config = self.config['VECTORSPACE']
        self.vectorizer = TfidfVectorizer(
            max_features=int(vectorizer_config.get('MAX_FEATURES', 1000)),
            ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                         int(vectorizer_config['NGRAM_MAX'])),
            min_df=int(vectorizer_config['MIN_DF']),
            max_df=float(vectorizer_config['MAX_DF']),
            norm=None
        )

        # Trainiere Vectorizer auf gemischtem Corpus
        self.vectorizer.fit(training_corpus)

        # Topic-Vektor nur aus voll relevanten Dokumenten
        vectors = self.vectorizer.transform(relevant_texts)
        vectors = normalize(vectors, norm='l2', axis=1)
        topic_vec = np.asarray(vectors.mean(axis=0)).reshape(1, -1)
        self.topic_vector = normalize(topic_vec, norm='l2', axis=1)

        # Speichere als classifier für Kompatibilität mit Basisklasse
        self.classifier = self.topic_vector

        # Speichere Modell
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.classifier, f)
        with open(self.vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)

        print(f"Topic-Vektor aus {len(relevant_texts)} relevanten Dokumenten erstellt")

    def calculate_text_relevance(self, text):
        """Berechnet Cosinus-Ähnlichkeit zwischen Text und Themenprofil"""
        if not text:
            return 0.0

        processed_text = self.preprocess_text(text)
        if not processed_text:
            return 0.0

        try:
            text_vector = self.vectorizer.transform([processed_text])
            if hasattr(text_vector, "nnz") and text_vector.nnz == 0:
                return 0.0

            # Normalisiere Text-Vektor für Cosinus-Ähnlichkeit
            text_vector = normalize(text_vector, norm='l2', axis=1)

            # Berechne Cosinus-Ähnlichkeit
            similarity = float(cosine_similarity(text_vector, self.topic_vector)[0, 0])
            return max(0.0, similarity)

        except Exception:
            return 0.0