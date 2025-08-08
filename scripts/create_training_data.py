
"""
Erstellt Trainingsdaten für Vectorspace Spider und Naive Bayes Spider
Extrahiert Webinhalte anhand gegebener URLs und speichert sie als JSON Datei
"""

import requests
from bs4 import BeautifulSoup
import configparser
import os
import re
import json
from pathlib import Path
from datetime import datetime


class TrainingDataGenerator:
    """Generator für Trainingsdaten"""

    def __init__(self, config_file='training_config.ini'):
        """Initialisiert Konfigurationen für Trainingsdaten"""
        self.config = configparser.ConfigParser(interpolation=None)
        self.config.read(config_file)

        # Konfiguration laden
        self.output_path = self.config['PATHS']['OUTPUT_FILE']
        self.backup_dir = self.config['PATHS']['BACKUP_DIR']
        self.user_agent = self.config['SETTINGS']['USER_AGENT']
        self.timeout = int(self.config['SETTINGS']['TIMEOUT'])
        self.min_text_length = int(self.config['SETTINGS']['MIN_TEXT_LENGTH'])

        # URLs Laden
        relevant_raw = self.config['URLS']['RELEVANT_URLS'].split(',')
        irrelevant_raw = self.config['URLS']['IRRELEVANT_URLS'].split(',')
        self.relevant_urls = sorted(list(set([url.strip() for url in relevant_raw if url.strip()])))
        self.irrelevant_urls = sorted(list(set([url.strip() for url in irrelevant_raw if url.strip()])))


        # Verzeichnisse erstellen
        Path(os.path.dirname(self.output_path)).mkdir(parents=True, exist_ok=True)
        Path(self.backup_dir).mkdir(parents=True, exist_ok=True)

        # Requests Header
        self.headers = {'User-Agent': self.user_agent}

        # Alle Trainingsdaten sammeln
        self.training_data = []

        # Statistiken
        self.stats = {
            'relevant': 0,
            'irrelevant': 0,
            'failed': 0
        }

    def extract_content(self, url):
        """Extrahiert relevanten Inhalt anhand gegebener URL"""
        try:
            # Seite herunterladen
            response = requests.get(url, headers=self.headers, timeout=self.timeout)
            response.raise_for_status()

            # HTML parsen
            soup = BeautifulSoup(response.text, 'html.parser')

            # Script und Style entfernen
            for script in soup(['script', 'style']):
                script.decompose()

            # Titel extrahieren
            title = soup.find('title')
            title_text = title.text if title else ''

            # Überschriften extrahieren
            headings = []
            for i in range(1, 7):
                for heading in soup.find_all(f'h{i}'):
                    headings.append(heading.get_text(strip=True))
            headings_text = ' '.join(headings)

            # Paragraphen extrahieren
            paragraphs = []
            for p in soup.find_all('p'):
                text = p.get_text(strip=True)
                # Ignoriere sehr kurze Paragraphen
                if len(text) > 20:
                    paragraphs.append(text)
            paragraphs_text = ' '.join(paragraphs)

            # Extrahiere Ankertexte
            anchors = []
            for a in soup.find_all('a'):
                anchor_text = a.get_text(strip=True)
                if anchor_text and len(anchor_text) > 2:
                    anchors.append(anchor_text)
            # Begrenzt auf 50 Ankertexte
            anchors_text = ' '.join(anchors[:50])

            # Alle Inhalte kopieren
            combined_text = f"{title_text} {headings_text} {paragraphs_text} {anchors_text}"

            # Text bereinigen
            combined_text = re.sub(r'\s+', ' ', combined_text)
            combined_text = combined_text.strip()

            return combined_text

        except Exception as e:
            print(f"Fehler bei {url}: {str(e)}")
            self.stats['failed'] += 1
            return None

    def process_urls(self, relevant_urls, irrelevant_urls):
        """Verarbeitet URLs und speichert Inhalt in JSON Datei"""

        # Backup, falls Trainingsdaten bereits vorhanden
        if os.path.exists(self.output_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_file = os.path.join(self.backup_dir, f"training_backup_{timestamp}.json")
            os.rename(self.output_path, backup_file)
            print(f"Backup von existierenden Daten erstellt und verschoben nach: {backup_file}")

        # Relevante URLs mit Label = 1 verarbeiten
        print("\nVerarbeite relevante URLs...")
        for i, url in enumerate(relevant_urls, 1):
            print(f"  [{i}/{len(relevant_urls)}] {url}")
            content = self.extract_content(url)

            if content and len(content) >= self.min_text_length:
                # Eintrag hinzufügen
                self.training_data.append({
                    "label": 1,
                    "text": content
                })
                self.stats['relevant'] += 1

        # Irrelevante URLs mit Label = 0 verarbeiten
        print("\nVerarbeite irrelevante URLs...")
        for i, url in enumerate(irrelevant_urls, 1):
            print(f"  [{i}/{len(irrelevant_urls)}] {url}")
            content = self.extract_content(url)

            if content and len(content) >= self.min_text_length:
                # Eintrag hinzufügen
                self.training_data.append({
                    "label": 0,
                    "text": content
                })
                self.stats['irrelevant'] += 1

        # Als JSON Datei speichern
        with open(self.output_path, 'w', encoding='utf-8') as f:
            json.dump(self.training_data, f, ensure_ascii=False, indent=2)

    def print_statistics(self):
        """Gibt Statistiken über die Erstellung der Trainingsdaten aus"""
        total = self.stats['relevant'] + self.stats['irrelevant']
        print(f"""
{'=' * 50}
TRAININGSDATEN ERFOLGREICH ERSTELLT
{'=' * 50}
Ausgabedatei: {self.output_path}
Format: JSON
Relevante Samples: {self.stats['relevant']}
Irrelevante Samples: {self.stats['irrelevant']}
Gesamt: {total}
Fehlgeschlagen: {self.stats['failed']}

Balance: {self.stats['relevant'] / max(1, total) * 100:.1f}% relevant / {self.stats['irrelevant'] / max(1, total) * 100:.1f}% irrelevant
{'=' * 50}
""")


def main():
    """Hauptfunktion"""

    # Generator erstellen
    generator = TrainingDataGenerator()

    print(f"""
{'=' * 50}
TRAININGSDATEN-GENERATOR
{'=' * 50}
Relevante URLs: {len(generator.relevant_urls)}
Irrelevante URLs: {len(generator.irrelevant_urls)}
{'=' * 50}
""")

    # Inhalte extrahieren und Erfolg ausgeben
    generator.process_urls(generator.relevant_urls, generator.irrelevant_urls)
    generator.print_statistics()


if __name__ == '__main__':
    main()