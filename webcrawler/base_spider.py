import scrapy
from scrapy.exceptions import CloseSpider
import heapq
from datetime import datetime, timedelta
import json
from urllib.parse import urlparse, unquote, urlunparse
from pathlib import Path
import configparser
from bs4 import BeautifulSoup
import re
import nltk
from nltk.corpus import stopwords
import time
import pickle
import os
import psutil
import numpy as np


class BaseTopicalSpider(scrapy.Spider):
    """Basisklasse für alle Topical Crawling Strategien"""

    custom_settings = {
        'DUPEFILTER_CLASS': 'scrapy.dupefilters.BaseDupeFilter',  # Deaktiviert Scrapy's Filter
        'REQUEST_FINGERPRINTER_IMPLEMENTATION': '2.7'
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Konfiguration laden
        self.config = configparser.ConfigParser()
        self.config.read('crawler_config.ini')

        # Crawler-Parameter
        self.batch_size = int(self.config['CRAWLER']['BATCH_SIZE'])
        self.max_pages = int(self.config['CRAWLER']['MAX_PAGES'])
        self.max_relevant = int(self.config['CRAWLER']['MAX_RELEVANT_PAGES'])
        self.relevance_threshold = float(self.config['CRAWLER']['RELEVANCE_THRESHOLD'])
        self.max_runtime = int(self.config['CRAWLER']['MAX_RUNTIME_MINUTES'])
        self.frontier_max_size = int(self.config['CRAWLER']['FRONTIER_MAX_SIZE'])

        # Domain und Namespace Filter
        self.allowed_domains = [d.strip() for d in self.config['CRAWLER']['ALLOWED_DOMAINS'].split(',')]
        self.ignored_namespaces = [ns.strip() for ns in self.config['CRAWLER']['IGNORED_NAMESPACES'].split(',')]

        # Gewichtungen
        self.link_weight = float(self.config['WEIGHTS']['LINK_WEIGHT'])
        self.parent_weight = float(self.config['WEIGHTS']['PARENT_WEIGHT'])
        self.title_weight = float(self.config['WEIGHTS']['TITLE_WEIGHT'])
        self.heading_weight = float(self.config['WEIGHTS']['HEADING_WEIGHT'])
        self.paragraph_weight = float(self.config['WEIGHTS']['PARAGRAPH_WEIGHT'])

        # NLTK Stoppwörter
        try:
            nltk.data.find('corpora/stopwords')
            self.stop_words = set(stopwords.words('german'))
        except LookupError:
            self.stop_words = set()

        # Seed URLs
        self.start_urls = [url.strip() for url in self.config['CRAWLER']['SEED_URLS'].split(',')]

        self.frontier = []
        self.frontier_urls = set()
        self.visited_urls = set()

        # Tracking
        self.current_batch = 0
        self.visit_counter = 0
        self.final_report_done = False
        self.pending_requests = 0

        # Metriken
        self.doc_eval_times_ns = []
        self.doc_memory_deltas = []
        self.doc_memory_baselines = []
        self.all_evaluated_pages = []

        self.stats = {
            'total_crawled': 0,
            'relevant_pages': 0,
            'irrelevant_pages': 0,
            'start_time': datetime.now(),
            'total_relevance_sum': 0.0
        }

        # Export vorbereiten
        Path("exports").mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.export_file = f"exports/{self.name}_{self.timestamp}.json"

        self.print_config()

    def normalize_url(self, url):
        """Normalisiert URL für konsistente Duplikaterkennung"""
        parsed = urlparse(url)

        # Hostname normalisieren
        hostname = parsed.netloc.lower()
        if ':' in hostname:
            hostname = hostname.split(':')[0]

        # Wikipedia-spezifische Normalisierung
        if 'wikipedia.org' in hostname:
            # Entferne www, m, mobile Präfixe
            hostname = re.sub(r'^(www\.|m\.|mobile\.)', '', hostname)

        # Schema normalisieren
        scheme = 'https'

        # Path beibehalten
        path = parsed.path.rstrip('/')
        if not path:
            path = '/'

        # Normalisierte URL ohne Query/Fragment
        normalized = urlunparse((
            scheme,
            hostname,
            path,
            '',  # params
            '',  # query
            ''  # fragment
        ))

        return normalized

    def is_valid_url(self, url):
        """Prüft ob URL gecrawlt werden soll"""
        parsed = urlparse(url)

        # Fragment oder Query direkt ablehnen
        if parsed.fragment or parsed.query:
            return False

        # Strikte Domain-Prüfung (ohne Port)
        hostname = parsed.netloc.lower()
        if ':' in hostname:
            hostname = hostname.split(':')[0]

        domain_valid = False
        for domain in self.allowed_domains:
            if hostname == domain or hostname.endswith('.' + domain):
                domain_valid = True
                break

        if not domain_valid:
            return False

        # Namespace-Filter
        decoded_path = unquote(parsed.path)
        for namespace in self.ignored_namespaces:
            if f'/{namespace}' in decoded_path or f'/wiki/{namespace}' in decoded_path:
                return False

        return True

    def add_to_frontier(self, url, score):
        """Fügt URL zur Frontier hinzu mit korrektem Duplikat-Check"""
        # Erst validieren
        if not self.is_valid_url(url):
            return

        normalized = self.normalize_url(url)

        # Bereits besucht oder schon in Frontier?
        if normalized in self.visited_urls or normalized in self.frontier_urls:
            return

        # Frontier-Größe prüfen - effizienter mit heappushpop
        if len(self.frontier) >= self.frontier_max_size:
            new_neg = -score
            # schlechtestes Element finden (größter negativer Wert)
            worst_idx, (worst_neg, worst_url) = max(
                enumerate(self.frontier), key=lambda x: x[1][0]
            )
            if new_neg < worst_neg:  # nur ersetzen, wenn neuer besser ist
                self.frontier_urls.discard(self.normalize_url(worst_url))
                self.frontier[worst_idx] = (new_neg, url)
                self.frontier_urls.add(normalized)
                heapq.heapify(self.frontier)
        else:
            heapq.heappush(self.frontier, (-score, url))
            self.frontier_urls.add(normalized)

    def print_config(self):
        """Ausgabe der Konfiguration"""
        print(f"\n{'=' * 60}")
        print(f"CRAWLER START - {self.name}")
        print(f"Seed URLs: {', '.join(self.start_urls)}")
        print(f"Batch-Größe: {self.batch_size}, Max. Seiten: {self.max_pages}")
        print(f"{'=' * 60}\n")

    def start_requests(self):
        """Initialisierung mit Seed-URLs"""
        for url in self.start_urls:
            self.add_to_frontier(url, 1.0)

        # Erste Batch verarbeiten
        for request in self.process_batch():
            yield request

    def process_batch(self):
        """Verarbeitet nächste Batch aus Frontier"""
        self.current_batch += 1

        # Beendigungskriterien prüfen
        if self.check_termination():
            if not self.final_report_done:
                self.final_report_done = True
                self.print_final_report()
            raise CloseSpider('Beendigungskriterium erreicht')

        # Batch aus Frontier holen
        batch = []
        for _ in range(min(self.batch_size, len(self.frontier))):
            if self.frontier:
                neg_score, url = heapq.heappop(self.frontier)
                normalized = self.normalize_url(url)

                # Aus frontier_urls entfernen (noch nicht visited!)
                self.frontier_urls.discard(normalized)
                batch.append((url, -neg_score, normalized))

        # Requests erstellen
        for url, score, normalized in batch:
            self.pending_requests += 1
            yield scrapy.Request(
                url=url,
                callback=self.parse,
                errback=self.handle_error,
                meta={'parent_score': score, 'normalized_url': normalized},
                dont_filter=True
            )

    def handle_error(self, failure):
        """Behandelt fehlgeschlagene Requests"""
        self.pending_requests -= 1

        # Bei Fehler: URL nicht als besucht markieren
        # So kann sie später ggf. erneut versucht werden

        # Nächste Batch wenn alle Requests fertig
        if self.pending_requests == 0:
            for request in self.process_batch():
                yield request

    def parse(self, response):
        """Parser für gecrawlte Seiten"""
        self.pending_requests -= 1

        # Finale Duplikatsprüfung mit tatsächlicher Response-URL
        final_normalized = self.normalize_url(response.url)
        if final_normalized in self.visited_urls:
            # Bereits verarbeitet (z.B. durch Redirect)
            if self.pending_requests == 0:
                for request in self.process_batch():
                    yield request
            return

        # Jetzt als besucht markieren
        self.visited_urls.add(final_normalized)
        # Auch die ursprünglich angeforderte URL markieren
        self.visited_urls.add(response.meta['normalized_url'])

        # Canonical URL prüfen und als primäre URL behandeln
        canonical = response.xpath('//link[@rel="canonical"]/@href').get()
        if canonical:
            canonical_norm = self.normalize_url(response.urljoin(canonical))
            if canonical_norm != final_normalized:
                if canonical_norm in self.visited_urls:
                    # Diese Seite wurde unter Canonical-URL bereits verarbeitet
                    if self.pending_requests == 0:
                        for request in self.process_batch():
                            yield request
                    return
                # Canonical als primäre URL markieren
                self.visited_urls.add(canonical_norm)

        self.stats['total_crawled'] += 1
        self.visit_counter += 1

        # Effizientes Parsing mit XPath statt BeautifulSoup
        title = response.xpath('//title/text()').get('')
        headings = ' '.join(response.xpath(
            '//h1/text() | //h2/text() | //h3/text() | //h4/text() | //h5/text() | //h6/text()').getall())
        paragraphs = ' '.join(response.xpath('//p/text()').getall())

        # Dokumentbewertung mit Zeitmessung
        process = psutil.Process()
        mem_before = process.memory_info().rss / (1024 * 1024)
        self.doc_memory_baselines.append(mem_before)

        time_start = time.perf_counter_ns()
        parent_relevance = self.calculate_parent_relevance(title, headings, paragraphs)
        time_end = time.perf_counter_ns()

        self.doc_eval_times_ns.append(time_end - time_start)

        mem_after = process.memory_info().rss / (1024 * 1024)
        self.doc_memory_deltas.append(mem_after - mem_before)

        # Seite speichern
        self.all_evaluated_pages.append({
            'url': response.url,
            'score': parent_relevance,
            'title': title[:100],
            'visit_idx': self.visit_counter,
            'visit_ts': datetime.now().isoformat()
        })

        # Statistiken aktualisieren
        if parent_relevance >= self.relevance_threshold:
            self.stats['relevant_pages'] += 1
        else:
            self.stats['irrelevant_pages'] += 1

        self.stats['total_relevance_sum'] += parent_relevance

        # Links mit XPath extrahieren
        for link in response.xpath('//a[@href]'):
            href = link.xpath('@href').get()
            anchor_text = link.xpath('string()').get('').strip()
            url = response.urljoin(href)

            # Link-Relevanz berechnen
            anchor_score = self.calculate_text_relevance(anchor_text)
            link_relevance = self.link_weight * anchor_score + self.parent_weight * parent_relevance

            self.add_to_frontier(url, link_relevance)

        # Fortschritt ausgeben
        if self.stats['total_crawled'] % 50 == 0:
            self.print_progress()

        # Nächste Batch wenn alle Requests dieser Batch fertig
        if self.pending_requests == 0:
            for request in self.process_batch():
                yield request

    def preprocess_text(self, text):
        """Textvorverarbeitung"""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = text.split()

        if self.stop_words:
            tokens = [t for t in tokens if t not in self.stop_words and len(t) > 2]
        else:
            tokens = [t for t in tokens if len(t) > 1]

        return ' '.join(tokens)

    def check_termination(self):
        """Prüft Beendigungskriterien"""
        runtime = datetime.now() - self.stats['start_time']

        if runtime > timedelta(minutes=self.max_runtime):
            return True
        if self.stats['total_crawled'] >= self.max_pages:
            return True
        if self.stats['relevant_pages'] >= self.max_relevant:
            return True
        if not self.frontier and self.pending_requests == 0:
            return True

        return False

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """Berechnet gewichtete Relevanz des Elterndokuments"""
        title_score = self.calculate_text_relevance(title) if title else 0.0
        heading_score = self.calculate_text_relevance(headings) if headings else 0.0
        paragraph_score = self.calculate_text_relevance(paragraphs) if paragraphs else 0.0

        weighted_score = (
                self.title_weight * title_score +
                self.heading_weight * heading_score +
                self.paragraph_weight * paragraph_score
        )

        return min(1.0, weighted_score)

    def calculate_text_relevance(self, text):
        """Abstrakte Methode - von Subklassen implementiert"""
        raise NotImplementedError("Subklassen müssen calculate_text_relevance implementieren")

    def print_progress(self):
        """Fortschrittsanzeige"""
        avg_relevance = self.stats['total_relevance_sum'] / max(1, self.stats['total_crawled'])
        harvest_rate = (self.stats['relevant_pages'] / max(1, self.stats['total_crawled'])) * 100

        print(f"[Batch {self.current_batch}] "
              f"Gecrawlt: {self.stats['total_crawled']} | "
              f"Relevant: {self.stats['relevant_pages']} | "
              f"Harvest: {harvest_rate:.1f}% | "
              f"Ø-Relevanz: {avg_relevance:.3f}")

    def calculate_percentiles(self, data):
        """Berechnet Statistiken für Metriken"""
        if not data:
            return {'mean': 0.0, 'median': 0.0, 'p95': 0.0, 'max': 0.0}

        sorted_data = sorted(data)
        return {
            'mean': float(np.mean(data)),
            'median': np.median(sorted_data),
            'p95': sorted_data[min(int(len(sorted_data) * 0.95), len(sorted_data) - 1)],
            'max': max(sorted_data)
        }

    def print_final_report(self):
        """Abschlussbericht und JSON-Export"""
        runtime = datetime.now() - self.stats['start_time']
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

        # Sortierte Seiten
        sorted_pages = sorted(self.all_evaluated_pages, key=lambda x: x['score'], reverse=True)

        if sorted_pages:
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

        # Metriken berechnen
        doc_eval_stats = self.calculate_percentiles(self.doc_eval_times_ns)
        doc_memory_delta_stats = self.calculate_percentiles(self.doc_memory_deltas)
        doc_memory_baseline_stats = self.calculate_percentiles(self.doc_memory_baselines)

        # JSON exportieren
        export_data = {
            'summary': {
                'spider': self.name,
                'timestamp': self.timestamp,
                'total_execution_time_s': runtime.total_seconds(),
                'total_pages_visited': self.stats['total_crawled'],
                'total_relevant_found': self.stats['relevant_pages'],
                'average_harvest_rate': harvest_rate / 100,
                'average_relevance': avg_relevance,
                'doc_eval_time_ns': doc_eval_stats,
                'parent_calc_count': len(self.doc_eval_times_ns),
                'doc_memory_delta_mib': doc_memory_delta_stats,
                'doc_memory_baseline_mib': doc_memory_baseline_stats,
                'pages': sorted_pages,
                'raw_scores': [{'url': p['url'], 'score': p['score']} for p in sorted_pages]
            }
        }

        with open(self.export_file, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, ensure_ascii=False, indent=2)

        # Plots erstellen wenn aktiviert
        if self.config.getboolean('PLOTTING', 'CREATE_PLOTS', fallback=False):
            try:
                import subprocess
                import sys
                project_root = os.path.dirname(os.path.dirname(__file__))
                script_path = os.path.join(project_root, 'scripts', 'create_plots.py')
                subprocess.run([sys.executable, script_path], check=False, cwd=project_root)
                print("Grafiken wurden erstellt")
            except Exception as e:
                print(f"Fehler beim Erstellen der Grafiken: {e}")

    # ML-Methoden für Subklassen
    def load_or_train_model(self):
        """Lädt oder trainiert ML-Modell"""
        if hasattr(self, 'model_path') and hasattr(self, 'vectorizer_path'):
            if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
                with open(self.model_path, 'rb') as f:
                    self.classifier = pickle.load(f)
                with open(self.vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                print("Existierendes Modell geladen")
            else:
                with open(self.training_data_path, 'r', encoding='utf-8') as f:
                    training_data = json.load(f)
                texts, labels = self.select_training_labels(training_data)
                self.train_model(texts, labels)

    def select_training_labels(self, training_data):
        """Von ML-Strategien zu implementieren"""
        raise NotImplementedError("ML-Strategien müssen select_training_labels implementieren")

    def train_model(self, texts, labels):
        """Von ML-Strategien zu implementieren"""
        raise NotImplementedError("ML-Strategien müssen train_model implementieren")