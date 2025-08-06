from base_spider import BaseTopicalSpider
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


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

        # Lade Themenprofil-Text aus Konfiguration
        self.topic_profile = self.config['VECTORSPACE']['TOPIC_PROFILE']

        # Lade Modus (tf_norm oder tfidf) - Default ist tf_norm
        self.mode = self.config['VECTORSPACE'].get('MODE', 'tf_norm').lower()

        # Lade Multiplikatoren aus Konfiguration
        self.title_multiplier = int(self.config['VECTORSPACE']['TITLE_MULTIPLIER'])
        self.heading_multiplier = int(self.config['VECTORSPACE']['HEADING_MULTIPLIER'])
        self.paragraph_multiplier = int(self.config['VECTORSPACE']['PARAGRAPH_MULTIPLIER'])

        # Initialisiere Vectorizer basierend auf Modus
        if self.mode == 'tf_norm':
            # Verwende Keyword-Vokabular für TF-Norm
            keywords = [kw.strip().lower() for kw in
                        self.config['KEYWORD']['KEYWORDS'].split(',')]
            # Erstelle Vokabular-Dictionary für scikit-learn
            vocabulary = {keyword: idx for idx, keyword in enumerate(keywords)}

            self.vectorizer = TfNormVectorizer(
                vocabulary=vocabulary
            )
            print(f"Verwende TF-Norm Vektorisierung mit Keyword-Vokabular")
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.95
            )
            print(f"Verwende TF-IDF Vektorisierung")

        # Erstelle Themen-Vektor (einmalig)
        self.create_topic_vector()

        print(f"Themenprofil initialisiert mit {len(self.topic_profile)} Zeichen")
        self.write_to_report(f"Themenprofil: {self.topic_profile[:200]}...\n")

    def create_topic_vector(self):
        """Erstellt einmalig den Themen-Vektor aus dem Profil"""
        # Vorverarbeitung des Themenprofils
        processed_profile = self.preprocess_text(self.topic_profile)

        # Fit Vectorizer mit Themenprofil und erstelle Vektor
        self.topic_vector = self.vectorizer.fit_transform([processed_profile])

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
        if hasattr(self.vectorizer, 'vocabulary'):
            # TF-Norm mit festem Vokabular
            vector_dim = len(self.vectorizer.count_vectorizer.vocabulary_)
        else:
            # TF-IDF
            vector_dim = self.topic_vector.shape[1]

        combined_vector = np.zeros((1, vector_dim))

        # Verarbeite Titel mit Multiplikator
        if title:
            processed_title = self.preprocess_text(title)
            if processed_title:
                try:
                    title_vector = self.vectorizer.transform([processed_title])
                    combined_vector += title_vector.toarray() * self.title_multiplier
                except:
                    pass

        # Verarbeite Überschriften mit Multiplikator
        if headings:
            processed_headings = self.preprocess_text(headings)
            if processed_headings:
                try:
                    heading_vector = self.vectorizer.transform([processed_headings])
                    combined_vector += heading_vector.toarray() * self.heading_multiplier
                except:
                    pass

        # Verarbeite Paragraphen mit Multiplikator
        if paragraphs:
            processed_paragraphs = self.preprocess_text(paragraphs)
            if processed_paragraphs:
                try:
                    paragraph_vector = self.vectorizer.transform([processed_paragraphs])
                    combined_vector += paragraph_vector.toarray() * self.paragraph_multiplier
                except:
                    pass

        # Berechne Cosinus-Ähnlichkeit zwischen kombiniertem Vektor und Themenprofil
        if np.any(combined_vector):
            # Berechne Ähnlichkeit OHNE zusätzliche Normalisierung
            # Die Multiplikatoren bleiben so erhalten
            similarity = cosine_similarity(combined_vector, self.topic_vector)[0][0]

            # Leichte Verstärkung für bessere Score-Verteilung
            # sqrt verschiebt niedrige Werte nach oben, behält hohe Werte
            boosted_score = np.sqrt(similarity)

            return float(min(1.0, boosted_score))
        else:
            return 0.0