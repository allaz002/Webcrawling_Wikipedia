from base_spider import BaseTopicalSpider
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import MaxAbsScaler
from sklearn.metrics.pairwise import cosine_similarity
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
        
        # Lade Modus (tf_norm oder tfidf)
        self.mode = self.config['VECTORSPACE'].get('MODE', 'tfidf').lower()
        
        # Lade Multiplikatoren aus Konfiguration
        self.title_multiplier = int(self.config['MULTIPLIERS']['TITLE_MULTIPLIER'])
        self.heading_multiplier = int(self.config['MULTIPLIERS']['HEADING_MULTIPLIER'])
        self.paragraph_multiplier = int(self.config['MULTIPLIERS']['PARAGRAPH_MULTIPLIER'])
        
        # Initialisiere Vectorizer basierend auf Modus
        if self.mode == 'tf_norm':
            self.vectorizer = TfNormVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=1
            )
            print(f"Verwende TF-Norm Vektorisierung")
        else:
            self.vectorizer = TfidfVectorizer(
                max_features=1000,
                ngram_range=(1, 2),
                min_df=1,
                max_df=1
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
        Berechnet gewichtete Vektorraum-Relevanz des Elterndokuments
        Verwendet konfigurierbare Multiplikatoren
        """
        # Erstelle gewichteten Gesamttext mit konfigurierbaren Multiplikatoren
        weighted_text = ""

        # Titel mit konfiguriertem Multiplikator
        if title:
            weighted_text += (title + " ") * self.title_multiplier

        # Überschriften mit konfiguriertem Multiplikator
        if headings:
            weighted_text += (headings + " ") * self.heading_multiplier

        # Paragraphen mit konfiguriertem Multiplikator
        if paragraphs:
            weighted_text += (paragraphs + " ") * self.paragraph_multiplier

        if not weighted_text:
            return 0.0

        # Berechne Ähnlichkeit für kombinierten Text
        combined_score = self.calculate_text_relevance(weighted_text)

        # Alternative: Individuelle Scores mit Gewichtung
        # (für bessere Nachvollziehbarkeit)
        title_score = self.calculate_text_relevance(title)
        heading_score = self.calculate_text_relevance(headings)
        paragraph_score = self.calculate_text_relevance(paragraphs)

        # Gewichtete Summe der Einzelscores
        weighted_score = (
                self.title_weight * title_score +
                self.heading_weight * heading_score +
                self.paragraph_weight * paragraph_score
        )

        total_weight = self.title_weight + self.heading_weight + self.paragraph_weight
        individual_weighted = weighted_score / total_weight if total_weight > 0 else 0.0

        # Kombiniere beide Ansätze (50/50)
        # Dies nutzt sowohl die Vorteile der kombinierten Vektorisierung
        # als auch die explizite Gewichtung
        final_score = (combined_score + individual_weighted) / 2

        return min(1.0, final_score)