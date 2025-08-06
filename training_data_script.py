#!/usr/bin/env python
"""
Skript zur Erstellung von Trainingsdaten für Naive Bayes Spider
Extrahiert Inhalte von URLs und speichert sie gelabelt
"""

import requests
from bs4 import BeautifulSoup
import configparser
import os
import re
from pathlib import Path
from datetime import datetime


class TrainingDataGenerator:
    """Generator für Naive Bayes Trainingsdaten"""
    
    def __init__(self, config_file='training_config.ini'):
        """Initialisiert Generator mit Konfiguration"""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)
        
        # Lade Konfiguration
        self.output_path = self.config['PATHS']['OUTPUT_FILE']
        self.backup_dir = self.config['PATHS']['BACKUP_DIR']
        self.user_agent = self.config['SETTINGS']['USER_AGENT']
        self.timeout = int(self.config['SETTINGS']['TIMEOUT'])
        self.min_text_length = int(self.config['SETTINGS']['MIN_TEXT_LENGTH'])
        
        # Erstelle Verzeichnisse
        Path(os.path.dirname(self.output_path)).mkdir(parents=True, exist_ok=True)
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)
        
        # Headers für Requests
        self.headers = {'User-Agent': self.user_agent}
        
        # Statistiken
        self.stats = {
            'relevant': 0,
            'irrelevant': 0,
            'failed': 0
        }
        
    def extract_content(self, url):
        """Extrahiert relevanten Inhalt von einer URL"""
        try:
            # Hole Seite
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Entferne Script und Style Tags
            for script in soup(['script', 'style']):
                script.decompose()
                
            # Extrahiere Titel
            title = soup.find('title')
            title_text = title.text if title else ''
            
            # Extrahiere alle Überschriften (h1-h6)
            headings = []
            for i in range(1, 7):
                for heading in soup.find_all(f'h{i}'):
                    headings.append(heading.get_text(strip=True))
            headings_text = ' '.join(headings)
            
            # Extrahiere Paragraphen
            paragraphs = []
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                if len(text) > 20:  # Ignoriere sehr kurze Paragraphen
                    paragraphs.append(text)
            paragraphs_text = ' '.join(paragraphs)
            
            # Extrahiere Ankertexte
            anchors = []
            for a in soup.find_all('a'):
                anchor_text = a.get_text(strip=True)
                if anchor_text and len(anchor_text) > 2:
                    anchors.append(anchor_text)
            anchors_text = ' '.join(anchors[:50])  # Limitiere auf 50 Ankertexte
            
            # Kombiniere alle Inhalte
            combined_text = f"{title_text} {headings_text} {paragraphs_text} {anchors_text}"
            
            # Bereinige Text
            combined_text = re.sub(r'\s+', ' ', combined_text)
            combined_text = combined_text.strip()
            
            return combined_text
            
        except Exception as e:
            print(f"Fehler bei {url}: {str(e)}")
            self.stats['failed'] += 1
            return None
            
    def process_urls(self, relevant_urls, irrelevant_urls):
        """Verarbeitet Listen von URLs und erstellt Trainingsdaten"""
        
        # Backup existierende Datei
        if os.path.exists(self.output_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(self.backup_dir, f"training_backup_{timestamp}.txt")
            os.rename(self.output_path, backup_file)
            print(f"Existierende Datei gesichert nach: {backup_file}")
            
        # Öffne Output-Datei
        with open(self.output_path, 'w', encoding='utf-8') as f:
            
            # Verarbeite relevante URLs (Label = 1)
            print("\nVerarbeite relevante URLs...")
            for i, url in enumerate(relevant_urls, 1):
                print(f"  [{i}/{len(relevant_urls)}] {url}")
                content = self.extract_content(url)
                
                if content and len(content) >= self.min_text_length:
                    # Schreibe als: 1<TAB>text
                    f.write(f"1\t{content}\n")
                    self.stats['relevant'] += 1
                    
            # Verarbeite irrelevante URLs (Label = 0)
            print("\nVerarbeite irrelevante URLs...")
            for i, url in enumerate(irrelevant_urls, 1):
                print(f"  [{i}/{len(irrelevant_urls)}] {url}")
                content = self.extract_content(url)
                
                if content and len(content) >= self.min_text_length:
                    # Schreibe als: 0<TAB>text
                    f.write(f"0\t{content}\n")
                    self.stats['irrelevant'] += 1
                    
    def print_statistics(self):
        """Gibt Statistiken aus"""
        total = self.stats['relevant'] + self.stats['irrelevant']
        print(f"""
{'='*50}
TRAININGSDATEN ERFOLGREICH ERSTELLT
{'='*50}
Ausgabedatei: {self.output_path}
Relevante Samples: {self.stats['relevant']}
Irrelevante Samples: {self.stats['irrelevant']}
Gesamt: {total}
Fehlgeschlagen: {self.stats['failed']}

Balance: {self.stats['relevant']/max(1,total)*100:.1f}% relevant / {self.stats['irrelevant']/max(1,total)*100:.1f}% irrelevant
{'='*50}
""")


def main():
    """Hauptfunktion mit Beispiel-URLs"""
    
    # HIER IHRE URLs EINFÜGEN!
    # ========================
    
    # Relevante URLs (zum Thema KI/Machine Learning)
    relevant_urls = [
        "https://de.wikipedia.org/wiki/K%C3%BCnstliche_Intelligenz",
        "https://de.wikipedia.org/wiki/Maschinelles_Lernen",
        "https://de.wikipedia.org/wiki/Generatives_KI-Modell",
        "https://de.wikipedia.org/wiki/Support_Vector_Machine",
        "https://de.wikipedia.org/wiki/Random_Forest",
        "https://de.wikipedia.org/wiki/K%C3%BCnstliches_neuronales_Netz",
        "https://de.wikipedia.org/wiki/Multimodale_k%C3%BCnstliche_Intelligenz",
        "https://de.wikipedia.org/wiki/Geschichte_der_k%C3%BCnstlichen_Intelligenz",
        "https://de.wikipedia.org/wiki/Halluzination_%28K%C3%BCnstliche_Intelligenz%29",
        "https://de.wikipedia.org/wiki/K%C3%BCnstliche_Intelligenz_in_der_Medizin",
        "https://de.wikipedia.org/wiki/K%C3%BCnstliche_Intelligenz_in_der_Materialwissenschaft",
        "https://de.wikipedia.org/wiki/Anwendungen_k%C3%BCnstlicher_Intelligenz",
        "https://de.wikipedia.org/wiki/Parameter_%28K%C3%BCnstliche_Intelligenz%29",
        "https://de.wikipedia.org/wiki/Ethik_der_k%C3%BCnstlichen_Intelligenz",
        "https://de.wikipedia.org/wiki/Existenzielles_Risiko_durch_k%C3%BCnstliche_Intelligenz",
        "https://de.wikipedia.org/wiki/European_Association_for_Artificial_Intelligence",
        "https://de.wikipedia.org/wiki/Enquete-Kommission_K%C3%BCnstliche_Intelligenz",
        "https://de.wikipedia.org/wiki/Emotionserkennung",
        "https://de.wikipedia.org/wiki/Embodiment",
        "https://de.wikipedia.org/wiki/ELIZA-Effekt",
        "https://de.wikipedia.org/wiki/Erica_%28Androidin%29",
        "https://de.wikipedia.org/wiki/European_Conference_on_Artificial_Intelligence",
        "https://de.wikipedia.org/wiki/Expertensystem",
        "https://de.wikipedia.org/wiki/IBM_Watson",
        "https://de.wikipedia.org/wiki/Chatbot",
        "https://de.wikipedia.org/wiki/Computer_Vision",
        "https://de.wikipedia.org/wiki/Natural_language_processing",
        "https://de.wikipedia.org/wiki/Neuronales_Netz",
        "https://de.wikipedia.org/wiki/Deep_Learning",
        "https://de.wikipedia.org/wiki/Turing-Test",
        "https://de.wikipedia.org/wiki/John_McCarthy",
        "https://de.wikipedia.org/wiki/Alan_Turing",
        "https://de.wikipedia.org/wiki/Peter_Norvig",
        "https://de.wikipedia.org/wiki/J%C3%BCrgen_Schmidhuber",
        "https://de.wikipedia.org/wiki/GPT-3",
        "https://de.wikipedia.org/wiki/DeepMind",
        "https://de.wikipedia.org/wiki/Reinforcement_Learning",
        "https://de.wikipedia.org/wiki/Bayes%27sche_Optimierung",
        "https://de.wikipedia.org/wiki/Emotionserkennung",
        "https://de.wikipedia.org/wiki/Computerlinguistik",
        "https://de.wikipedia.org/wiki/Nat%C3%BCrliche_Sprache",
        "https://de.wikipedia.org/wiki/Big_Data",
        "https://de.wikipedia.org/wiki/Merkmalsauswahl",
        "https://de.wikipedia.org/wiki/TensorFlow",
        "https://de.wikipedia.org/wiki/Yann_LeCun",
        "https://de.wikipedia.org/wiki/Gemini_(Sprachmodell)",
        "https://de.wikipedia.org/wiki/Datenwissenschaft",
        "https://de.wikipedia.org/wiki/Data_Mining",
        "https://de.wikipedia.org/wiki/Support_Vector_Machine",
        "https://de.wikipedia.org/wiki/Random_Forest",
    ]

    # Irrelevante URLs (andere Themen)
    irrelevant_urls = [
        "https://de.wikipedia.org/wiki/Deutschland",
        "https://de.wikipedia.org/wiki/Kalifornien",
        "https://de.wikipedia.org/wiki/Boston",
        "https://de.wikipedia.org/wiki/Memphis_%28Tennessee%29",
        "https://de.wikipedia.org/wiki/Los_Angeles",
        "https://de.wikipedia.org/wiki/New_York_City",
        "https://de.wikipedia.org/wiki/San_Francisco",
        "https://de.wikipedia.org/wiki/New_Orleans",
        "https://de.wikipedia.org/wiki/San_Diego",
        "https://de.wikipedia.org/wiki/Paris",
        "https://de.wikipedia.org/wiki/London",
        "https://de.wikipedia.org/wiki/Musik",
        "https://de.wikipedia.org/wiki/Sport",
        "https://de.wikipedia.org/wiki/Geschichte",
        "https://de.wikipedia.org/wiki/Geographie",
        "https://de.wikipedia.org/wiki/Politik",
        "https://de.wikipedia.org/wiki/Kultur",
        "https://de.wikipedia.org/wiki/Filme",
        "https://de.wikipedia.org/wiki/Literatur",
        "https://de.wikipedia.org/wiki/Kunst",
        "https://de.wikipedia.org/wiki/Reisen",
        "https://de.wikipedia.org/wiki/Architektur",
        "https://de.wikipedia.org/wiki/Philosophie",
        "https://de.wikipedia.org/wiki/Biologie",
        "https://de.wikipedia.org/wiki/Chemie",
        "https://de.wikipedia.org/wiki/Physik",
        "https://de.wikipedia.org/wiki/Vogel",
        "https://de.wikipedia.org/wiki/Baum",
        "https://de.wikipedia.org/wiki/Haustier",
        "https://de.wikipedia.org/wiki/Auto",
        "https://de.wikipedia.org/wiki/Fußball",
        "https://de.wikipedia.org/wiki/Basketball",
        "https://de.wikipedia.org/wiki/Tennis",
        "https://de.wikipedia.org/wiki/Klavier",
        "https://de.wikipedia.org/wiki/Oper",
        "https://de.wikipedia.org/wiki/Papierflugzeug",
        "https://de.wikipedia.org/wiki/Wetter",
        "https://de.wikipedia.org/wiki/Gastronomie",
        "https://de.wikipedia.org/wiki/Mode",
        "https://de.wikipedia.org/wiki/Tourismus",
        "https://de.wikipedia.org/wiki/Schokolade",
        "https://de.wikipedia.org/wiki/Wein",
        "https://de.wikipedia.org/wiki/Bier",
        "https://de.wikipedia.org/wiki/Rugby",
        "https://de.wikipedia.org/wiki/Baseball",
        "https://de.wikipedia.org/wiki/Cricket",
        "https://de.wikipedia.org/wiki/Lyrik",
        "https://de.wikipedia.org/wiki/Malerei",
        "https://de.wikipedia.org/wiki/Theater",
        "https://de.wikipedia.org/wiki/Sehensw%C3%BCrdigkeit"
    ]
    
    # ========================
    
    # Erstelle Generator und verarbeite URLs
    generator = TrainingDataGenerator()
    
    print(f"""
{'='*50}
TRAININGSDATEN-GENERATOR
{'='*50}
Relevante URLs: {len(relevant_urls)}
Irrelevante URLs: {len(irrelevant_urls)}
{'='*50}
""")
    
    # Verarbeite URLs
    generator.process_urls(relevant_urls, irrelevant_urls)
    
    # Zeige Statistiken
    generator.print_statistics()


if __name__ == '__main__':
    main()