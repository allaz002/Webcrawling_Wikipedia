from base_spider import BaseTopicalSpider
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os
import scipy.sparse as sp
import random


class TfNormVectorizer:
    """TF-Norm Vektorisierer: f_t,d / max_t' f_t',d"""

    def __init__(self, vocabulary=None, max_features=1000, ngram_range=(1, 2), min_df=1, max_df=0.95):
        # Wenn Vokabular übergeben, verwende es
        if vocabulary:
            self.count_vectorizer = CountVectorizer(
                vocabulary=vocabulary,
                ngram_range=(1, 1)  # Nur Unigrams für Keywords
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
        """Fit Vektorisierer auf Dokumenten"""
        count_matrix = self.count_vectorizer.fit_transform(documents)
        self.scaler.fit(count_matrix)
        return self

    def transform(self, documents):
        """Transformiere Dokumente zu TF-Norm Vektoren"""
        count_matrix = self.count_vectorizer.transform(documents)
        return self.scaler.transform(count_matrix)

    def fit_transform(self, documents):
        """Fit und Transform in einem Schritt"""
        count_matrix = self.count_vectorizer.fit_transform(documents)
        return self.scaler.fit_transform(count_matrix)


class VectorSpaceSpider(BaseTopicalSpider):
    """Vektorraum-Modell mit Cosinus-Ähnlichkeit"""

    name = 'vectorspace_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lade Modus (tf_norm oder tfidf) - Default ist tf_norm
        self.mode = self.config['VECTORSPACE'].get('MODE', 'tf_norm').lower()

        # Lade IDF-Verhältnis für TF-IDF Modus
        self.idf_relevant_ratio = float(self.config['VECTORSPACE'].get('IDF_RELEVANT_RATIO', '1.0'))

        # Lade Multiplikatoren aus Konfiguration
        self.title_multiplier = int(self.config['VECTORSPACE']['TITLE_MULTIPLIER'])
        self.heading_multiplier = int(self.config['VECTORSPACE']['HEADING_MULTIPLIER'])
        self.paragraph_multiplier = int(self.config['VECTORSPACE']['PARAGRAPH_MULTIPLIER'])

        # Initialisiere Vectorizer basierend auf Modus
        if self.mode == 'tf_norm':
            # Verwende Keyword-Vokabular für TF-Norm
            keywords = [kw.strip().lower() for kw in
                        self.config['VECTORSPACE']['KEYWORDS'].split(',')]
            # Erstelle Vokabular-Dictionary für scikit-learn
            vocabulary = {keyword: idx for idx, keyword in enumerate(keywords)}

            self.vectorizer = TfNormVectorizer(
                vocabulary=vocabulary
            )
            print(f"Verwende TF-Norm Vektorisierung mit Keyword-Vokabular")
        else:
            # TF-IDF Modus mit gemeinsamen Parametern
            self.vectorizer = TfidfVectorizer(
                max_features=int(self.config['VECTORIZER_SETTINGS']['MAX_FEATURES']),
                ngram_range=(int(self.config['VECTORIZER_SETTINGS']['NGRAM_MIN']),
                             int(self.config['VECTORIZER_SETTINGS']['NGRAM_MAX'])),
                min_df=int(self.config['VECTORIZER_SETTINGS']['MIN_DF']),
                max_df=float(self.config['VECTORIZER_SETTINGS']['MAX_DF'])
            )
            print(f"Verwende TF-IDF Vektorisierung mit Relevant-Ratio: {self.idf_relevant_ratio}")

        # Lade Trainingsdaten für IDF-Berechnung bei TF-IDF
        self.load_training_data_for_idf()

        print(f"VectorSpace Spider initialisiert im {self.mode} Modus")
        self.write_to_report(f"Modus: {self.mode}\n")
        if self.mode == 'tfidf':
            self.write_to_report(f"IDF Relevant-Ratio: {self.idf_relevant_ratio}\n")

    def load_training_data_for_idf(self):
        """Lädt Trainingsdaten und erstellt Themen-Vektor basierend auf IDF"""
        training_path = self.config['VECTORSPACE']['TRAINING_DATA_PATH']

        if self.mode == 'tfidf' and os.path.exists(training_path):
            # Lade JSON-Trainingsdaten
            with open(training_path, 'r', encoding='utf-8') as f:
                training_data = json.load(f)

            # Separiere relevante und irrelevante Texte
            relevant_texts = []
            irrelevant_texts = []

            for sample in training_data:
                processed_text = self.preprocess_text(sample['text'])
                if sample['label'] == 1:
                    relevant_texts.append(processed_text)
                else:
                    irrelevant_texts.append(processed_text)

            if relevant_texts and irrelevant_texts:
                # Berechne gewünschte Anzahl irrelevanter Dokumente basierend auf Ratio
                # Bei ratio=0.75: 75% relevant, 25% irrelevant
                # Also: irrelevant_count = relevant_count * (1-ratio) / ratio
                num_relevant = len(relevant_texts)

                if self.idf_relevant_ratio < 1.0:
                    # Berechne Anzahl der irrelevanten Dokumente für gewünschtes Verhältnis
                    num_irrelevant_needed = int(num_relevant * (1 - self.idf_relevant_ratio) / self.idf_relevant_ratio)

                    # Sample aus irrelevanten Texten (mit Wiederholung wenn nötig)
                    if num_irrelevant_needed > len(irrelevant_texts):
                        # Wenn wir mehr brauchen als vorhanden, sample mit replacement
                        sampled_irrelevant = random.choices(irrelevant_texts, k=num_irrelevant_needed)
                    else:
                        # Sonst sample ohne replacement
                        sampled_irrelevant = random.sample(irrelevant_texts, num_irrelevant_needed)

                    # Kombiniere für IDF-Berechnung
                    idf_training_texts = relevant_texts + sampled_irrelevant

                    print(
                        f"IDF-Training mit {num_relevant} relevanten und {len(sampled_irrelevant)} irrelevanten Dokumenten")
                else:
                    # Ratio = 1.0, nur relevante Dokumente verwenden
                    idf_training_texts = relevant_texts
                    print(f"IDF-Training nur mit {num_relevant} relevanten Dokumenten")

                # Fit Vectorizer auf angepasstem Trainingsset
                self.vectorizer.fit(idf_training_texts)

                # Erstelle Themen-Vektor NUR aus relevanten Dokumenten
                vectors = self.vectorizer.transform(relevant_texts)
                # Konvertiere zu dense Array und berechne Durchschnitt
                self.topic_vector = np.asarray(vectors.mean(axis=0)).reshape(1, -1)

                print(f"Themen-Vektor aus {len(relevant_texts)} relevanten Dokumenten erstellt")
            else:
                # Fallback: Fit auf einzelnem Dummy-Dokument
                dummy_text = "künstliche intelligenz machine learning"
                self.vectorizer.fit([dummy_text])
                self.topic_vector = self.vectorizer.transform([dummy_text]).toarray()
                print("Warnung: Keine gültigen Trainingsdaten gefunden, verwende Fallback")

        elif self.mode == 'tf_norm':
            # TF-Norm: Fit auf Keywords
            keywords_text = ' '.join([kw.strip().lower() for kw in
                                      self.config['VECTORSPACE']['KEYWORDS'].split(',')])
            processed_keywords = self.preprocess_text(keywords_text)
            self.topic_vector = self.vectorizer.fit_transform([processed_keywords]).toarray()
            print("Vectorizer auf Keywords trainiert")
        else:
            # Fallback für TF-IDF ohne Trainingsdaten
            dummy_text = "künstliche intelligenz machine learning"
            self.vectorizer.fit([dummy_text])
            self.topic_vector = self.vectorizer.transform([dummy_text]).toarray()
            print("Warnung: Keine Trainingsdaten gefunden, verwende Fallback")

    def calculate_text_relevance(self, text):
        """
        Berechnet Cosinus-Ähnlichkeit zwischen Text und Themenprofil
        Wert nahe 1 = hohe Ähnlichkeit
        """
        if not text:
            return 0.0

        # Textvorverarbeitung
        processed_text = self.preprocess_text(text)

        if not processed_text:
            return 0.0

        try:
            # Transformiere Text in Vektor mit gleichem Vokabular
            text_vector = self.vectorizer.transform([processed_text])

            # Konvertiere zu dense wenn nötig
            if sp.issparse(text_vector):
                text_vector = text_vector.toarray()

            # Berechne Cosinus-Ähnlichkeit
            similarity = cosine_similarity(text_vector, self.topic_vector)[0][0]

            # Similarity ist bereits im Bereich [0, 1]
            return float(similarity)

        except Exception as e:
            # Bei Fehler (z.B. leerer Text nach Vorverarbeitung)
            return 0.0

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """
        Optimierte Vektorraum-Relevanz nur mit Multiplikatoren
        Berechnet gewichtete Vektorsumme und Cosinus-Ähnlichkeit
        """
        # Initialisiere Null-Vektor mit gleicher Dimension wie Topic-Vektor
        vector_dim = self.topic_vector.shape[1]
        combined_vector = np.zeros((1, vector_dim))

        # Verarbeite Titel mit Multiplikator
        if title:
            processed_title = self.preprocess_text(title)
            if processed_title:
                try:
                    title_vector = self.vectorizer.transform([processed_title])
                    if sp.issparse(title_vector):
                        title_vector = title_vector.toarray()
                    combined_vector += title_vector * self.title_multiplier
                except:
                    pass

        # Verarbeite Überschriften mit Multiplikator
        if headings:
            processed_headings = self.preprocess_text(headings)
            if processed_headings:
                try:
                    heading_vector = self.vectorizer.transform([processed_headings])
                    if sp.issparse(heading_vector):
                        heading_vector = heading_vector.toarray()
                    combined_vector += heading_vector * self.heading_multiplier
                except:
                    pass

        # Verarbeite Paragraphen mit Multiplikator
        if paragraphs:
            processed_paragraphs = self.preprocess_text(paragraphs)
            if processed_paragraphs:
                try:
                    paragraph_vector = self.vectorizer.transform([processed_paragraphs])
                    if sp.issparse(paragraph_vector):
                        paragraph_vector = paragraph_vector.toarray()
                    combined_vector += paragraph_vector * self.paragraph_multiplier
                except:
                    pass

        # Berechne Cosinus-Ähnlichkeit zwischen kombiniertem Vektor und Themenprofil
        if np.any(combined_vector):
            # Stelle sicher, dass beide Vektoren die richtige Form haben
            if combined_vector.ndim == 1:
                combined_vector = combined_vector.reshape(1, -1)
            if self.topic_vector.ndim == 1:
                self.topic_vector = self.topic_vector.reshape(1, -1)

            similarity = cosine_similarity(combined_vector, self.topic_vector)[0][0]

            # Leichte Verstärkung für bessere Score-Verteilung
            # Entfernt sqrt() Boost - verwende direkten Similarity-Wert
            # Dies führt zu realistischeren Scores
            boosted_score = np.sqrt(similarity)

            return float(min(1.0, boosted_score))
        else:
            return 0.0