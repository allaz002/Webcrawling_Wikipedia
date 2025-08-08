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


class BaseTopicalSpider(scrapy.Spider):
    """Basisklasse für alle Topical Crawling Strategien"""

    def __init__(self, *args, **kwargs):
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
        self.report_interval = int(self.config['CRAWLER']['REPORT_INTERVAL_SECONDS'])
        self.frontier_max_size = int(self.config['CRAWLER']['FRONTIER_MAX_SIZE'])

        # Domain und Namespace Filter aus Config
        self.allowed_domains = [d.strip() for d in self.config['CRAWLER']['ALLOWED_DOMAINS'].split(',')]
        self.ignored_namespaces = [ns.strip() for ns in self.config['CRAWLER']['IGNORED_NAMESPACES'].split(',')]

        # Reporting Konfiguration
        self.report_pages_interval = int(self.config['REPORTING']['REPORT_PAGES_INTERVAL'])
        self.report_time_interval = int(self.config['REPORTING']['REPORT_TIME_INTERVAL_SECONDS'])

        # Gewichtungen für Link-Relevanz
        self.link_weight = float(self.config['WEIGHTS']['LINK_WEIGHT'])
        self.parent_weight = float(self.config['WEIGHTS']['PARENT_WEIGHT'])

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

        # Flag für finalen Report
        self.final_report_written = False

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
            'last_report': datetime.now(),
            'all_pages': [],  # Für Endstatistik
            'evaluation_log': [],  # Für Evaluierungsdaten
            'total_relevance_sum': 0.0,  # Für Durchschnittsberechnung
            'batch_frontier_positions': [],  # Durchschnittliche Positionen pro Batch
            'relevance_calculation_time': 0.0  # Zeit für Relevanzbewertung
        }

        # Verzeichnisse erstellen
        Path("exports").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)

        # Report-Datei initialisieren (einmalig mit festem Timestamp)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_file = f"reports/{self.name}_{self.timestamp}.txt"
        self.export_file = f"exports/{self.name}_{self.timestamp}.json"

        self.print_config()

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

    def print_config(self):
        """Ausgabe der Konfiguration beim Start"""
        config_text = f"""
{'=' * 60}
CRAWLER KONFIGURATION - {self.name}
{'=' * 60}
Strategie: {self.__class__.__name__}
Seed URLs: {', '.join(self.start_urls)}
Batch-Größe: {self.batch_size}
Max. Seiten: {self.max_pages}
Max. relevante Seiten: {self.max_relevant}
Relevanz-Schwellenwert: {self.relevance_threshold}
Max. Laufzeit: {self.max_runtime} Minuten
Report-Intervall: {self.report_interval} Sekunden

GEWICHTUNGEN:
- Link-Gewicht: {self.link_weight}
- Elterndokument-Gewicht: {self.parent_weight}
{'=' * 60}
"""
        print(config_text)
        self.write_to_report(config_text)

    def write_to_report(self, text):
        """Schreibt Text in die Report-Datei"""
        with open(self.report_file, 'a', encoding='utf-8') as f:
            f.write(text + '\n')

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
                link_relevance = self.calculate_link_relevance(anchor_text, parent_relevance)
                self.add_to_frontier(url, link_relevance)

        # Periodische Reports
        self.print_progress_report()

        # Nächste Batch verarbeiten
        yield from self.process_batch()

    def calculate_link_relevance(self, anchor_text, parent_relevance):
        """Berechnet gewichtete Link-Relevanz"""
        anchor_score = self.calculate_text_relevance(anchor_text)
        return self.link_weight * anchor_score + self.parent_weight * parent_relevance

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """Abstrakte Methode - muss von Subklassen implementiert werden"""
        raise NotImplementedError("Subklassen müssen calculate_parent_relevance implementieren")

    def calculate_text_relevance(self, text):
        """Abstrakte Methode - wird von Subklassen implementiert"""
        raise NotImplementedError("Subklassen müssen calculate_text_relevance implementieren")

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
            tokens = [t for t in tokens if len(t) > 2]

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
        # Verhindere mehrfache Aufrufe nach Beendigung
        if self.final_report_written:
            return True

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

    def print_progress_report(self):
        """Gibt Zwischenbericht aus"""
        now = datetime.now()
        pages_since_last = self.stats['total_crawled'] % self.report_pages_interval
        time_since_last = (now - self.stats['last_report']).seconds

        # Report nach Seitenanzahl oder Zeit
        if pages_since_last == 0 or time_since_last >= self.report_time_interval:
            runtime = now - self.stats['start_time']
            harvest_rate = (self.stats['relevant_pages'] / max(1, self.stats['total_crawled'])) * 100

            report = f"""
--- ZWISCHENBERICHT [{now.strftime('%H:%M:%S')}] ---
Laufzeit: {runtime}
Harvest-Rate: {harvest_rate:.2f}%
Relevante Seiten: {self.stats['relevant_pages']}
Irrelevante Seiten: {self.stats['irrelevant_pages']}
Aktueller Batch: {self.current_batch_number}
"""
            print(report)
            self.write_to_report(report)
            self.stats['last_report'] = now

    def print_final_report(self):
        """Gibt Abschlussbericht aus und speichert JSON mit Evaluierungsdaten"""
        # Verhindere mehrfache Ausführung
        if self.final_report_written:
            return

        self.final_report_written = True

        # Letzte Batch-Evaluierung
        self.log_batch_evaluation()

        runtime = datetime.now() - self.stats['start_time']
        runtime_seconds = runtime.total_seconds()
        harvest_rate = (self.stats['relevant_pages'] / max(1, self.stats['total_crawled'])) * 100
        avg_relevance = self.stats['total_relevance_sum'] / max(1, self.stats['total_crawled'])

        # Sortiere alle Seiten nach Score
        sorted_pages = sorted(self.stats['all_pages'], key=lambda x: x['score'], reverse=True)

        # Top 10, Mittelfeld 10, Bottom 10
        top_10 = sorted_pages[:10] if len(sorted_pages) >= 10 else sorted_pages
        middle_start = max(0, len(sorted_pages) // 2 - 5)
        middle_10 = sorted_pages[middle_start:middle_start + 10] if len(sorted_pages) >= 20 else []
        bottom_10 = sorted_pages[-10:] if len(sorted_pages) >= 10 else sorted_pages[-len(sorted_pages):]

        report = f"""
{'=' * 60}
ABSCHLUSSBERICHT - {self.name}
{'=' * 60}
Spider-Name: {self.name}
Gesamtlaufzeit: {runtime}
Anzahl gecrawlter Seiten: {self.stats['total_crawled']}
Anzahl relevanter Seiten: {self.stats['relevant_pages']}
Anzahl irrelevanter Seiten: {self.stats['irrelevant_pages']}
Harvest-Rate: {harvest_rate:.2f}%
Durchschnittliche Relevanz: {avg_relevance:.4f}
Gesamte Relevanzbewertungszeit: {self.stats['relevance_calculation_time']:.2f} Sekunden

TOP 10 BESTE URLS:
"""
        for i, page in enumerate(top_10, 1):
            report += f"{i}. Score: {page['score']:.4f} - {page['url']}\n"

        if middle_10:
            report += "\nTOP 10 MITTELFELD URLS:\n"
            for i, page in enumerate(middle_10, 1):
                report += f"{i}. Score: {page['score']:.4f} - {page['url']}\n"

        if bottom_10:
            report += "\nTOP 10 SCHLECHTESTE URLS:\n"
            for i, page in enumerate(bottom_10, 1):
                report += f"{i}. Score: {page['score']:.4f} - {page['url']}\n"

        report += f"\n{'=' * 60}\n"

        print(report)
        self.write_to_report(report)

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

        print(f"Relevante Seiten und Evaluierungsdaten exportiert nach: {self.export_file}")

        # Erstelle Plots wenn aktiviert
        if self.config.getboolean('PLOTTING', 'CREATE_PLOTS', fallback=False):
            self.create_plots()

    def create_plots(self):
        """Ruft externes Plot-Skript auf"""
        try:
            import subprocess
            subprocess.run([sys.executable, 'create_plots.py'], check=False)
            print("Grafiken wurden erstellt")
        except Exception as e:
            print(f"Fehler beim Erstellen der Grafiken: {e}")