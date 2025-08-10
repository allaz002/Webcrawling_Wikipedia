import scrapy
from scrapy import signals
from scrapy.exceptions import CloseSpider
import heapq
from datetime import datetime, timedelta
import json
import os
from urllib.parse import urlparse, unquote
from collections import defaultdict
import configparser
from bs4 import BeautifulSoup
import re
from pathlib import Path
import nltk
from nltk.corpus import stopwords
import sys
import time
import pickle
import logging
from twisted.python import log as twisted_log

class BaseTopicalSpider(scrapy.Spider):
    """Basisklasse für alle Topical Crawling Strategien"""

    def __init__(self, *args, **kwargs):

        logging.getLogger('scrapy').setLevel(logging.CRITICAL)
        logging.getLogger('twisted').setLevel(logging.CRITICAL)

        for obs in list(twisted_log.theLogPublisher.observers):
            twisted_log.removeObserver(obs)

        super().__init__(*args, **kwargs)

        # Konfiguration laden
        self.config = configparser.ConfigParser()
        self.config.read('crawler_config.ini')

        # Allgemeine Einstellungen
        self.batch_size = int(self.config['CRAWLER']['BATCH_SIZE'])
        self.max_pages = int(self.config['CRAWLER']['MAX_PAGES'])
        self.max_relevant = int(self.config['CRAWLER']['MAX_RELEVANT_PAGES'])
        self.relevance_threshold = float(self.config['CRAWLER']['RELEVANCE_THRESHOLD'])
        self.max_runtime = int(self.config['CRAWLER']['MAX_RUNTIME_MINUTES'])
        self.frontier_max_size = int(self.config['CRAWLER']['FRONTIER_MAX_SIZE'])

        # Flag für finalen Report
        self.final_report_done = False

        # Domain und Namespace Filter aus Config
        self.allowed_domains = [d.strip() for d in self.config['CRAWLER']['ALLOWED_DOMAINS'].split(',')]
        self.ignored_namespaces = [ns.strip() for ns in self.config['CRAWLER']['IGNORED_NAMESPACES'].split(',')]

        # Gewichtungen für alle Strategien
        self.link_weight = float(self.config['WEIGHTS']['LINK_WEIGHT'])
        self.parent_weight = float(self.config['WEIGHTS']['PARENT_WEIGHT'])
        self.title_weight = float(self.config['WEIGHTS']['TITLE_WEIGHT'])
        self.heading_weight = float(self.config['WEIGHTS']['HEADING_WEIGHT'])
        self.paragraph_weight = float(self.config['WEIGHTS']['PARAGRAPH_WEIGHT'])

        # NLTK deutsche Stoppwörter laden
        try:
            nltk.data.find('corpora/stopwords')
            self.stop_words = set(stopwords.words('german'))
        except LookupError:
            print("Warnung: NLTK Stoppwörter nicht gefunden.")
            print("Installation mit: nltk.download('stopwords')")
            self.stop_words = set()

        # Seed URLs
        self.start_urls = [url.strip() for url in self.config['CRAWLER']['SEED_URLS'].split(',')]

        # Frontier als Priority Queue (Max-Heap durch negative Scores)
        self.frontier = []
        self.visited_urls = set()
        self.url_scores = {}  # Cache für URL-Scores

        # Batch-Tracking für Evaluierung
        self.current_batch_number = 0
        self.batch_new_urls = []  # URLs die in diesem Batch zur Frontier hinzugefügt wurden
        self.batch_positions = []  # Positionen der neuen URLs in der Frontier

        # Statistiken
        self.stats = {
            'total_crawled': 0,
            'relevant_pages': 0,
            'irrelevant_pages': 0,
            'start_time': datetime.now(),
            'all_pages': [],  # Für Endstatistik
            'evaluation_log': [],  # Für Evaluierungsdaten
            'total_relevance_sum': 0.0,  # Für Durchschnittsberechnung
            'batch_frontier_positions': [],  # Durchschnittliche Positionen pro Batch
            'relevance_calculation_time': 0.0  # Zeit für Relevanzbewertung
        }

        # Verzeichnisse erstellen
        Path("exports").mkdir(parents=True, exist_ok=True)

        # Export-Datei initialisieren
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.export_file = f"exports/{self.name}_{self.timestamp}.json"

        self.print_config()

    def print_config(self):
        """Ausgabe der Konfiguration beim Start"""
        print(f"\n{'=' * 60}")
        print(f"CRAWLER START - {self.name}")
        print(f"Strategie: {self.__class__.__name__}")
        print(f"Seed URLs: {', '.join(self.start_urls)}")
        print(f"Batch-Größe: {self.batch_size}, Max. Seiten: {self.max_pages}")
        print(f"{'=' * 60}\n")

    def start_requests(self):
        """Initialisiert Crawler mit Seed-URLs"""
        for url in self.start_urls:
            # Seed-URLs mit hoher Priorität in Frontier
            self.add_to_frontier(url, 1.0)

        # Erste Batch verarbeiten
        yield from self.process_batch()

    def add_to_frontier(self, url, score):
        """Fügt URL zur Frontier hinzu und trackt Position"""
        if url not in self.visited_urls and self.is_valid_url(url):
            # Frontier-Größe begrenzen
            if len(self.frontier) >= self.frontier_max_size:
                # Entferne schlechteste URLs wenn Limit erreicht
                if score > -self.frontier[0][0]:  # Besserer Score als schlechtester
                    heapq.heapreplace(self.frontier, (-score, url))
                    self.url_scores[url] = score
                    # Tracke Position (approximiert)
                    position = self.get_url_position_in_frontier(score)
                    self.batch_positions.append(position)
            else:
                heapq.heappush(self.frontier, (-score, url))
                self.url_scores[url] = score
                # Tracke Position
                position = self.get_url_position_in_frontier(score)
                self.batch_positions.append(position)
                self.batch_new_urls.append(url)

    def get_url_position_in_frontier(self, score):
        """Berechnet die Position einer URL in der Frontier basierend auf ihrem Score"""
        position = 1
        for neg_score, _ in self.frontier:
            if -neg_score > score:
                position += 1
        return position

    def process_batch(self):
        """Verarbeitet nächste Batch aus Frontier"""
        batch = []

        # Erhöhe Batch-Nummer
        self.current_batch_number += 1

        # Prüfe Beendigungskriterien
        if self.check_termination():
            if not self.final_report_done:
                self.final_report_done = True
                self.print_final_report()
            raise CloseSpider('Beendigungskriterium erreicht')

        # Hole Top-N URLs aus Frontier
        for _ in range(min(self.batch_size, len(self.frontier))):
            if self.frontier:
                neg_score, url = heapq.heappop(self.frontier)
                batch.append((url, -neg_score))
                self.visited_urls.add(url)

        # Erstelle Requests für Batch
        for url, score in batch:
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                meta={'parent_score': score}
            )

        # Logge Batch-Evaluierung nach Verarbeitung
        if self.stats['total_crawled'] > 0:  # Nicht beim ersten Aufruf
            self.log_batch_evaluation()

    def parse(self, response):
        """Hauptparser für gecrawlte Seiten"""
        self.stats['total_crawled'] += 1

        # Textinhalte extrahieren
        soup = BeautifulSoup(response.text, 'html.parser')

        # Entferne Script und Style Tags
        for script in soup(["script", "style"]):
            script.decompose()

        title = soup.find('title').text if soup.find('title') else ''
        headings = ' '.join([h.text for h in soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])])
        paragraphs = ' '.join([p.text for p in soup.find_all('p')])

        # Zeitmessung für Relevanzbewertung
        calc_start = time.time()

        # Berechne Elterndokument-Relevanz (nur einmal pro Seite)
        parent_relevance = self.calculate_parent_relevance(title, headings, paragraphs)

        # Akkumuliere Berechnungszeit
        calc_time = time.time() - calc_start
        self.stats['relevance_calculation_time'] += calc_time

        # Speichere Seite und aktualisiere Statistiken
        if parent_relevance >= self.relevance_threshold:
            self.stats['relevant_pages'] += 1
            self.stats['all_pages'].append({
                'url': response.url,
                'score': parent_relevance,
                'title': title[:100]
            })
        else:
            self.stats['irrelevant_pages'] += 1

        # Akkumuliere Relevanz für Durchschnitt
        self.stats['total_relevance_sum'] += parent_relevance

        # Extrahiere und bewerte Links
        for link in soup.find_all('a', href=True):
            url = response.urljoin(link['href'])
            anchor_text = link.text.strip()

            if url not in self.visited_urls and self.is_valid_url(url):
                # Berechne Link-Relevanz (einmalig pro Link)
                link_relevance = self.calculate_combined_relevance(anchor_text, parent_relevance)
                self.add_to_frontier(url, link_relevance)

        # Periodisches Feedback
        self.print_progress()

        # Nächste Batch verarbeiten
        yield from self.process_batch()

    def preprocess_text(self, text):
        """Textvorverarbeitung: Case Folding und Stoppwort-Entfernung mit NLTK"""
        # Case Folding
        text = text.lower()

        # Entferne Sonderzeichen
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenisierung
        tokens = text.split()

        # Stoppwort-Entfernung mit NLTK
        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        else:
            # Fallback ohne Stoppwort-Entfernung
            tokens = [t for t in tokens if len(t) > 1]

        return ' '.join(tokens)

    def is_valid_url(self, url):
        """Prüft URL-Gültigkeit"""
        parsed = urlparse(url)

        # Domain-Filter - nur deutsche Wikipedia
        valid_domain = False
        for allowed in self.allowed_domains:
            if allowed in parsed.netloc:
                valid_domain = True
                break

        if not valid_domain:
            return False

        # URL-Dekodierung für Namespace-Filter
        decoded_path = unquote(parsed.path)

        # Namespace-Filter für Wikipedia
        for namespace in self.ignored_namespaces:
            if f'/{namespace}' in decoded_path or f'/wiki/{namespace}' in decoded_path:
                return False

        # Keine Anchors oder Queries
        if parsed.fragment or parsed.query:
            return False

        return True

    def check_termination(self):
        """Prüft Beendigungskriterien"""
        # Zeit-Limit
        runtime = datetime.now() - self.stats['start_time']
        if runtime > timedelta(minutes=self.max_runtime):
            return True

        # Seiten-Limit
        if self.stats['total_crawled'] >= self.max_pages:
            return True

        # Relevante Seiten Limit
        if self.stats['relevant_pages'] >= self.max_relevant:
            return True

        # Frontier leer
        if not self.frontier:
            return True

        return False

    def calculate_combined_relevance(self, anchor_text, parent_relevance):
        """Berechnet gewichtete Gesamtrelevanz"""
        anchor_score = self.calculate_text_relevance(anchor_text)
        return self.link_weight * anchor_score + self.parent_weight * parent_relevance

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """Berechnet gewichtete Relevanz des Elterndokuments"""
        # Berechne Scores für jeden Bereich
        title_score = self.calculate_text_relevance(title) if title else 0.0
        heading_score = self.calculate_text_relevance(headings) if headings else 0.0
        paragraph_score = self.calculate_text_relevance(paragraphs) if paragraphs else 0.0

        # Gewichtete Kombination
        weighted_score = (
                self.title_weight * title_score +
                self.heading_weight * heading_score +
                self.paragraph_weight * paragraph_score
        )

        return min(1.0, weighted_score)

    def calculate_text_relevance(self, text):
        """Abstrakte Methode - wird von Subklassen implementiert"""
        raise NotImplementedError("Subklassen müssen calculate_text_relevance implementieren")

    # Methoden für ML-basierte Strategien
    def load_or_train_model(self):
        """Lädt existierendes Modell oder trainiert neues"""
        if hasattr(self, 'model_path') and hasattr(self, 'vectorizer_path'):
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                with open(self.model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("Existierendes Modell geladen")
            else:
                # Lade Trainingsdaten
                with open(self.training_data_path, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)

                # Selektiere Labels je nach Strategie
                texts, labels = self.select_training_labels(training_data)

                # Trainiere Modell
                self.train_model(texts, labels)

    def select_training_labels(self, training_data):
        """Wird von Subklassen überschrieben für Label-Selektion"""
        raise NotImplementedError("ML-Strategien müssen select_training_labels implementieren")

    def train_model(self, texts, labels):
        """Wird von Subklassen überschrieben für Modell-Training"""
        raise NotImplementedError("ML-Strategien müssen train_model implementieren")

    def log_batch_evaluation(self):
        """Loggt Evaluierungsdaten nach jedem Batch"""
        runtime = (datetime.now() - self.stats['start_time']).total_seconds()
        harvest_rate = self.stats['relevant_pages'] / max(1, self.stats['total_crawled'])
        avg_relevance = self.stats['total_relevance_sum'] / max(1, self.stats['total_crawled'])

        # Berechne durchschnittliche Position der neuen URLs in der Frontier
        avg_position = 0
        if self.batch_positions:
            avg_position = sum(self.batch_positions) / len(self.batch_positions)

        evaluation_entry = {
            'batch_number': self.current_batch_number,
            'pages_visited': self.stats['total_crawled'],
            'relevant_pages_found': self.stats['relevant_pages'],
            'harvest_rate': harvest_rate,
            'average_relevance': avg_relevance,
            'execution_time_in_seconds': runtime,
            'avg_frontier_position': avg_position,
            'new_urls_added': len(self.batch_positions),
            'relevance_calculation_time': self.stats['relevance_calculation_time']
        }

        self.stats['evaluation_log'].append(evaluation_entry)

        # Reset für nächsten Batch
        self.batch_positions = []
        self.batch_new_urls = []

    def print_progress(self):
        """Gibt Fortschritt in Konsole aus"""
        if self.stats['total_crawled'] % 50 == 0:  # Alle 50 Seiten
            avg_relevance = self.stats['total_relevance_sum'] / max(1, self.stats['total_crawled'])
            harvest_rate = (self.stats['relevant_pages'] / max(1, self.stats['total_crawled'])) * 100
            print(f"[Batch {self.current_batch_number}] "
                  f"Gecrawlt: {self.stats['total_crawled']} | "
                  f"Relevant: {self.stats['relevant_pages']} | "
                  f"Harvest: {harvest_rate:.1f}% | "
                  f"Ø-Relevanz: {avg_relevance:.3f}")

    def print_final_report(self):
        """Gibt Abschlussbericht aus und speichert JSON mit Evaluierungsdaten"""
        # Letzte Batch-Evaluierung
        self.log_batch_evaluation()

        runtime = datetime.now() - self.stats['start_time']
        runtime_seconds = runtime.total_seconds()
        harvest_rate = (self.stats['relevant_pages'] / max(1, self.stats['total_crawled'])) * 100
        avg_relevance = self.stats['total_relevance_sum'] / max(1, self.stats['total_crawled'])

        print(f"\n{'=' * 60}")
        print(f"ABSCHLUSS - {self.name}")
        print(f"Laufzeit: {runtime}")
        print(f"Gecrawlt: {self.stats['total_crawled']} Seiten")
        print(f"Relevant: {self.stats['relevant_pages']} ({harvest_rate:.1f}%)")
        print(f"Ø-Relevanz: {avg_relevance:.4f}")
        print(f"Export: {self.export_file}")
        print(f"{'=' * 60}\n")

        # Sortiere alle Seiten nach Score
        sorted_pages = sorted(self.stats['all_pages'], key=lambda x: x['score'], reverse=True)

        # Top 5, Median 5, Bottom 5
        if len(sorted_pages) > 0:
            print("TOP 5 URLS:")
            for i, page in enumerate(sorted_pages[:5], 1):
                print(f"{i}. {page['score']:.4f} - {page['url']}")

            if len(sorted_pages) >= 10:
                print("\nMEDIAN 5 URLS:")
                middle = len(sorted_pages) // 2
                for i, page in enumerate(sorted_pages[middle - 2:middle + 3][:5], 1):
                    print(f"{i}. {page['score']:.4f} - {page['url']}")

                print("\nBOTTOM 5 URLS:")
                for i, page in enumerate(sorted_pages[-5:], 1):
                    print(f"{i}. {page['score']:.4f} - {page['url']}")

            print(f"\n{'=' * 60}\n")

        # Exportiere JSON mit Evaluierungsdaten
        relevant_pages = [p for p in sorted_pages if p['score'] >= self.relevance_threshold]

        export_data = {
            'summary': {
                'spider': self.name,
                'timestamp': self.timestamp,
                'total_execution_time_in_seconds': runtime_seconds,
                'total_pages_visited': self.stats['total_crawled'],
                'total_relevant_found': self.stats['relevant_pages'],
                'average_harvest_rate': harvest_rate / 100,
                'average_relevance': avg_relevance,
                'relevance_calculation_time': self.stats['relevance_calculation_time'],
                'evaluation_log': self.stats['evaluation_log'],
                'pages': relevant_pages
            }
        }

        with open(self.export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        # Erstelle Plots wenn aktiviert (nur einmal!)
        if self.config.getboolean('PLOTTING', 'CREATE_PLOTS', fallback=False):
            try:
                import subprocess
                subprocess.run([sys.executable, 'scripts/create_plots.py'], check=False)
                print("Grafiken wurden erstellt")
            except Exception as e:
                print(f"Fehler beim Erstellen der Grafiken: {e}")

    def create_plots(self):
        """Ruft externes Plot-Skript auf"""
        try:
            import subprocess
            subprocess.run([sys.executable, 'scripts/create_plots.py'], check=False)
            print("Grafiken wurden erstellt")
        except Exception as e:
            print(f"Fehler beim Erstellen der Grafiken: {e}")