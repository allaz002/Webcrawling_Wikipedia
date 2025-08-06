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
import spacy


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

        # Gewichtungen
        self.link_weight = float(self.config['WEIGHTS']['LINK_WEIGHT'])
        self.parent_weight = float(self.config['WEIGHTS']['PARENT_WEIGHT'])
        self.title_weight = float(self.config['WEIGHTS']['TITLE_WEIGHT'])
        self.heading_weight = float(self.config['WEIGHTS']['HEADING_WEIGHT'])
        self.paragraph_weight = float(self.config['WEIGHTS']['PARAGRAPH_WEIGHT'])

        # SpaCy deutsches Modell für Stoppwörter laden
        try:
            self.nlp = spacy.load("de_core_news_sm")
        except:
            print("Warnung: SpaCy Modell 'de_core_news_sm' nicht gefunden.")
            print("Installation mit: python -m spacy download de_core_news_sm")
            self.nlp = None

        # Seed URLs
        self.start_urls = [url.strip() for url in self.config['CRAWLER']['SEED_URLS'].split(',')]

        # Frontier als Priority Queue (Max-Heap durch negative Scores)
        self.frontier = []
        self.visited_urls = set()
        self.url_scores = {}  # Cache für URL-Scores

        # Flag für finalen Report
        self.final_report_written = False

        # Statistiken
        self.stats = {
            'total_crawled': 0,
            'relevant_pages': 0,
            'irrelevant_pages': 0,
            'start_time': datetime.now(),
            'last_report': datetime.now(),
            'all_pages': []  # Für Endstatistik
        }

        # Verzeichnisse erstellen
        Path("exports").mkdir(exist_ok=True)
        Path("reports").mkdir(exist_ok=True)

        # Report-Datei initialisieren (einmalig mit festem Timestamp)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.report_file = f"reports/{self.name}_{self.timestamp}.txt"
        self.export_file = f"exports/{self.name}_{self.timestamp}.json"

        self.print_config()

    def print_config(self):
        """Ausgabe der Konfiguration beim Start"""
        config_text = f"""
{'='*60}
CRAWLER KONFIGURATION - {self.name}
{'='*60}
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
- Titel-Gewicht: {self.title_weight}
- Überschriften-Gewicht: {self.heading_weight}
- Paragraphen-Gewicht: {self.paragraph_weight}
{'='*60}
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
        """Fügt URL zur Frontier hinzu"""
        if url not in self.visited_urls and self.is_valid_url(url):
            # Frontier-Größe begrenzen
            if len(self.frontier) >= self.frontier_max_size:
                # Entferne schlechteste URLs wenn Limit erreicht
                if score > -self.frontier[0][0]:  # Besserer Score als schlechtester
                    heapq.heapreplace(self.frontier, (-score, url))
                    self.url_scores[url] = score
            else:
                heapq.heappush(self.frontier, (-score, url))
                self.url_scores[url] = score

    def process_batch(self):
        """Verarbeitet nächste Batch aus Frontier"""
        batch = []

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

        # Berechne Elterndokument-Relevanz (nur einmal pro Seite)
        parent_relevance = self.calculate_parent_relevance(title, headings, paragraphs)

        # Speichere Seite wenn relevant
        if parent_relevance >= self.relevance_threshold:
            self.stats['relevant_pages'] += 1
            self.stats['all_pages'].append({
                'url': response.url,
                'score': parent_relevance,
                'title': title[:100]
            })
        else:
            self.stats['irrelevant_pages'] += 1

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
        """Berechnet gewichtete Elterndokument-Relevanz"""
        title_score = self.calculate_text_relevance(title)
        heading_score = self.calculate_text_relevance(headings)
        paragraph_score = self.calculate_text_relevance(paragraphs)

        return (self.title_weight * title_score +
                self.heading_weight * heading_score +
                self.paragraph_weight * paragraph_score)

    def calculate_text_relevance(self, text):
        """Abstrakte Methode - wird von Subklassen implementiert"""
        raise NotImplementedError("Subklassen müssen calculate_text_relevance implementieren")

    def preprocess_text(self, text):
        """Textvorverarbeitung: Case Folding und Stoppwort-Entfernung"""
        # Case Folding
        text = text.lower()

        # Entferne Sonderzeichen
        text = re.sub(r'[^\w\s]', ' ', text)

        # Tokenisierung
        tokens = text.split()

        # Stoppwort-Entfernung mit SpaCy
        if self.nlp:
            # SpaCy deutsche Stoppwörter verwenden
            tokens = [t for t in tokens if t not in self.nlp.Defaults.stop_words and len(t) > 2]
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
"""
            print(report)
            self.write_to_report(report)
            self.stats['last_report'] = now

    def print_final_report(self):
        """Gibt Abschlussbericht aus"""
        # Verhindere mehrfache Ausführung
        if self.final_report_written:
            return

        self.final_report_written = True

        runtime = datetime.now() - self.stats['start_time']
        harvest_rate = (self.stats['relevant_pages'] / max(1, self.stats['total_crawled'])) * 100

        # Sortiere alle Seiten nach Score
        sorted_pages = sorted(self.stats['all_pages'], key=lambda x: x['score'], reverse=True)

        # Top 10, Mittelfeld 10, Bottom 10
        top_10 = sorted_pages[:10] if len(sorted_pages) >= 10 else sorted_pages
        middle_start = max(0, len(sorted_pages) // 2 - 5)
        middle_10 = sorted_pages[middle_start:middle_start+10] if len(sorted_pages) >= 20 else []
        bottom_10 = sorted_pages[-10:] if len(sorted_pages) >= 10 else sorted_pages[-len(sorted_pages):]

        report = f"""
{'='*60}
ABSCHLUSSBERICHT - {self.name}
{'='*60}
Spider-Name: {self.name}
Gesamtlaufzeit: {runtime}
Anzahl gecrawlter Seiten: {self.stats['total_crawled']}
Anzahl relevanter Seiten: {self.stats['relevant_pages']}
Anzahl irrelevanter Seiten: {self.stats['irrelevant_pages']}
Harvest-Rate: {harvest_rate:.2f}%

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

        report += f"\n{'='*60}\n"

        print(report)
        self.write_to_report(report)

        # Exportiere relevante Seiten als JSON (einmalig mit festem Dateinamen)
        relevant_pages = [p for p in sorted_pages if p['score'] >= self.relevance_threshold]

        with open(self.export_file, 'w', encoding='utf-8') as f:
            json.dump({
                'spider': self.name,
                'timestamp': self.timestamp,
                'total_relevant': len(relevant_pages),
                'pages': relevant_pages
            }, f, indent=2, ensure_ascii=False)

        print(f"Relevante Seiten exportiert nach: {self.export_file}")