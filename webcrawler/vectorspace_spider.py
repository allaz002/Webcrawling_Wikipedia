from webcrawler.base_spider import BaseTopicalSpider
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import numpy as np
import json


class VectorSpaceSpider(BaseTopicalSpider):
    """Vektorraum-Modell mit Cosinus-Ähnlichkeit"""

    name = 'vectorspace_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Gewichtungen aus WEIGHTS Sektion
        self.title_weight = float(self.config['WEIGHTS']['TITLE_WEIGHT'])
        self.heading_weight = float(self.config['WEIGHTS']['HEADING_WEIGHT'])
        self.paragraph_weight = float(self.config['WEIGHTS']['PARAGRAPH_WEIGHT'])

        # IDF-Trainingsmischung aus VECTORSPACE Sektion
        self.idf_ratio_irrelevant = float(self.config['VECTORSPACE'].get('IDF_RATIO_IRRELEVANT', 0.33))
        self.idf_ratio_moderate = float(self.config['VECTORSPACE'].get('IDF_RATIO_MODERATE', 0.33))
        self.idf_ratio_relevant = float(self.config['VECTORSPACE'].get('IDF_RATIO_RELEVANT', 0.34))

        # Validierung der IDF-Ratios
        ratio_sum = self.idf_ratio_irrelevant + self.idf_ratio_moderate + self.idf_ratio_relevant
        if abs(ratio_sum - 1.0) > 0.001:
            raise ValueError(f"IDF-Ratios summieren sich nicht zu 1.0: {ratio_sum}")
        if any(r < 0 for r in [self.idf_ratio_irrelevant, self.idf_ratio_moderate, self.idf_ratio_relevant]):
            raise ValueError("IDF-Ratios dürfen nicht negativ sein")

        # TF-IDF Vectorizer - alle Parameter aus VECTORSPACE Sektion
        vectorizer_config = self.config['VECTORSPACE']
        self.vectorizer = TfidfVectorizer(
            max_features=int(vectorizer_config.get('MAX_FEATURES', 1000)),
            ngram_range=(int(vectorizer_config['NGRAM_MIN']),
                         int(vectorizer_config['NGRAM_MAX'])),
            min_df=int(vectorizer_config['MIN_DF']),
            max_df=float(vectorizer_config['MAX_DF']),
            norm=None
        )

        self.load_training_data()

        print(f"VectorSpace Spider initialisiert (TF-IDF)")
        self.write_to_report(f"Modus: TF-IDF\n")
        self.write_to_report(f"IDF-Ratios: irrelevant={self.idf_ratio_irrelevant:.2f}, "
                             f"moderate={self.idf_ratio_moderate:.2f}, "
                             f"relevant={self.idf_ratio_relevant:.2f}\n")

    def load_training_data(self):
        """Lädt Trainingsdaten und erstellt Themen-Vektor"""
        with open(self.config['VECTORSPACE']['TRAINING_DATA_PATH'], 'r', encoding='utf-8') as f:
            training_data = json.load(f)

        # Separiere nach drei Relevanzstufen
        irrelevant_texts = []
        moderate_texts = []
        relevant_texts = []

        for sample in training_data:
            processed_text = self.preprocess_text(sample['text'])
            if processed_text:  # Nur nicht-leere Texte
                if sample['label'] == 0:
                    irrelevant_texts.append(processed_text)
                elif sample['label'] == 1:
                    moderate_texts.append(processed_text)
                elif sample['label'] == 2:
                    relevant_texts.append(processed_text)

        print(f"Geladene Trainingsdaten: {len(relevant_texts)} voll relevant, "
              f"{len(moderate_texts)} mäßig relevant, {len(irrelevant_texts)} irrelevant")

        # Erstelle Trainingsmischung gemäß IDF-Ratios
        training_corpus = []

        # Berechne Anzahl Samples pro Kategorie
        total_samples = 100  # Basis für Proportionen
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

        # Trainiere Vectorizer auf gemischtem Corpus
        self.vectorizer.fit(training_corpus)

        # Topic-Vektor nur aus voll relevanten Dokumenten
        vectors = self.vectorizer.transform(relevant_texts)
        vectors = normalize(vectors, norm='l2', axis=1)
        topic_vec = np.asarray(vectors.mean(axis=0)).reshape(1, -1)
        self.topic_vector = normalize(topic_vec, norm='l2', axis=1)

        print(f"TF-IDF Topic-Vektor aus {len(relevant_texts)} voll relevanten Dokumenten")
        print(f"Vectorizer trainiert auf {len(training_corpus)} gemischten Dokumenten")

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
            return max(0.0, similarity)  # Sicherstellen dass >= 0

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