from base_spider import BaseTopicalSpider
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import numpy as np


class NaiveBayesSpider(BaseTopicalSpider):
    """Naive Bayes Klassifikations-Strategie"""
    
    name = 'naivebayes_crawler'
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Pfade zu Trainingsdaten
        self.model_path = self.config['NAIVEBAYES']['MODEL_PATH']
        self.vectorizer_path = self.config['NAIVEBAYES']['VECTORIZER_PATH']
        self.training_data_path = self.config['NAIVEBAYES']['TRAINING_DATA_PATH']
        
        # Lade oder trainiere Modell
        self.load_or_train_model()
        
        print(f"Naive Bayes Modell geladen/trainiert")
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
        """Trainiert Naive Bayes Klassifikator mit Trainingsdaten"""
        print("Trainiere neues Naive Bayes Modell...")
        
        # Initialisiere Vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        
        # Lade Trainingsdaten
        # Format: Datei mit Zeilen "label<TAB>text"
        # label: 1 für relevant, 0 für irrelevant
        texts = []
        labels = []
        
        if os.path.exists(self.training_data_path):
            with open(self.training_data_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if '\t' in line:
                        label, text = line.strip().split('\t', 1)
                        texts.append(self.preprocess_text(text))
                        labels.append(int(label))
        else:
            # Erstelle Beispiel-Trainingsdaten wenn keine vorhanden
            print("Keine Trainingsdaten gefunden, erstelle Beispieldaten...")
            self.create_sample_training_data()
            return self.train_model()  # Rekursiv mit neuen Daten
            
        if not texts:
            raise ValueError("Keine Trainingsdaten vorhanden!")
            
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
        
    def create_sample_training_data(self):
        """Erstellt Beispiel-Trainingsdaten falls keine vorhanden"""
        sample_data = [
            # Relevante Beispiele (Label 1)
            "1\tMachine Learning ist ein Teilgebiet der künstlichen Intelligenz",
            "1\tDeep Learning nutzt neuronale Netze zur Mustererkennung",
            "1\tKünstliche Intelligenz revolutioniert die Datenanalyse",
            "1\tAlgorithmen des maschinellen Lernens verbessern sich kontinuierlich",
            "1\tNeuronale Netzwerke simulieren das menschliche Gehirn",
            # Irrelevante Beispiele (Label 0)
            "0\tDas Wetter heute ist sonnig und warm",
            "0\tFußball ist eine beliebte Sportart weltweit",
            "0\tKochen macht Spaß und ist kreativ",
            "0\tMusik beeinflusst unsere Stimmung positiv",
            "0\tReisen erweitert den Horizont und bildet"
        ]
        
        os.makedirs(os.path.dirname(self.training_data_path), exist_ok=True)
        with open(self.training_data_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sample_data))
            
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
            # Index 0 wäre "irrelevant"
            relevance_probability = probabilities[1] if len(probabilities) > 1 else 0.0
            
            return float(relevance_probability)
            
        except Exception as e:
            # Bei Fehler (z.B. unbekannte Wörter)
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
        
        # Kombiniere Wahrscheinlichkeiten mit Bayes-Regel
        # P(relevant|alle_teile) approximiert durch gewichtete Kombination
        
        # Gewichtete arithmetische Mittelung
        weighted_prob = (
            self.title_weight * title_prob +
            self.heading_weight * heading_prob +
            self.paragraph_weight * paragraph_prob
        )
        
        total_weight = self.title_weight + self.heading_weight + self.paragraph_weight
        avg_prob = weighted_prob / total_weight if total_weight > 0 else 0.0
        
        # Alternative: Geometrisches Mittel für Wahrscheinlichkeiten
        # (berücksichtigt alle Teile müssen relevant sein)
        if title_prob > 0 and heading_prob > 0 and paragraph_prob > 0:
            geometric_prob = (
                (title_prob ** self.title_weight) *
                (heading_prob ** self.heading_weight) *
                (paragraph_prob ** self.paragraph_weight)
            ) ** (1 / total_weight)
        else:
            geometric_prob = 0.0
            
        # Kombiniere beide Ansätze (70% arithmetisch, 30% geometrisch)
        # Arithmetisch ist toleranter, geometrisch strenger
        final_prob = 0.7 * avg_prob + 0.3 * geometric_prob
        
        return min(1.0, final_prob)