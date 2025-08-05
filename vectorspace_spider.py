from base_spider import BaseTopicalSpider
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class VectorSpaceSpider(BaseTopicalSpider):
    """Vektorraum-Modell mit Cosinus-Ähnlichkeit"""
    
    name = 'vectorspace_crawler'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Lade Themenprofil-Text aus Konfiguration
        self.topic_profile = self.config['VECTORSPACE']['TOPIC_PROFILE']
        
        # Initialisiere TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',  # scikit-learn Stoppwort-Entfernung
            lowercase=True,  # Case Folding
            ngram_range=(1, 2),  # Uni- und Bigrams
            min_df=1,
            max_df=0.95
        )
        
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
        
        # Speichere Vokabular für spätere Transformationen
        self.vocabulary = self.vectorizer.vocabulary_
        
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
        Kombiniert verschiedene Textteile zu einem Gesamtvektor
        """
        # Erstelle gewichteten Gesamttext
        # Titel und Überschriften werden mehrfach eingefügt für höhere Gewichtung
        weighted_text = ""
        
        # Titel mit höherer Gewichtung (3x wiederholen)
        if title:
            weighted_text += (title + " ") * 3
            
        # Überschriften mit mittlerer Gewichtung (2x)
        if headings:
            weighted_text += (headings + " ") * 2
            
        # Paragraphen normal (1x)
        if paragraphs:
            weighted_text += paragraphs
            
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