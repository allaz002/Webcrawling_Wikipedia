"""
Evaluiert Crawling-Strategien gegen optimierte Ground-Truth aus Wikipedia-Kategorien
"""

import re
import json
import time
import requests
import configparser
from pathlib import Path
from urllib.parse import urlparse, unquote
from collections import deque, defaultdict


class WikiGroundTruthEvaluator:
    """Evaluiert Crawling-Strategien gegen Wikipedia-Kategorie-basierte Ground-Truth"""

    def __init__(self, config_file='evaluation_config.ini'):
        """Initialisiert Evaluator mit Konfiguration"""
        self.config = configparser.ConfigParser()
        self.config.optionxform = str  # Behalte UPPERCASE

        # Config-Pfad auflösen
        config_path = Path(config_file)
        if not config_path.is_absolute():
            config_path = Path(__file__).parent / config_file

        self.config.read(str(config_path), encoding='utf-8')

        # Konfiguration laden
        self.seed_categories = self._get_list('TOPIC', 'SEED_CATEGORIES')
        self.max_depth = self.config.getint('TOPIC', 'MAX_CATEGORY_DEPTH')
        self.exclude_patterns = [
            re.compile(p, re.I)
            for p in self._get_list('TOPIC', 'CATEGORY_EXCLUDE_REGEX')
        ]

        # Erweiterte Filterung für präzisere Ground-Truth
        self.min_articles_per_category = self.config.getint('TOPIC', 'MIN_ARTICLES_PER_CATEGORY', fallback=5)
        self.max_articles_per_category = self.config.getint('TOPIC', 'MAX_ARTICLES_PER_CATEGORY', fallback=500)
        self.exclude_category_keywords = self._get_list('TOPIC', 'EXCLUDE_CATEGORY_KEYWORDS')

        # Whitelist-Ansatz für Kernthemen (optional)
        self.core_articles_whitelist = self._get_list('TOPIC', 'CORE_ARTICLES_WHITELIST')

        # API-Konfiguration
        self.endpoint = self.config.get('API', 'ENDPOINT')
        self.max_rps = self.config.getfloat('API', 'MAX_REQUESTS_PER_SEC')

        # Pfade
        self.root = config_path.parent
        self.exports_dir = self.root / self.config.get('IO', 'EXPORTS_DIR')
        self.cache_path = self.root / self.config.get('IO', 'CACHE_PATH')
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        self.file_pattern = self.config.get('IO', 'FILE_PATTERN')

        # Steuerungsoptionen
        self.force_rebuild = self.config.getboolean('OPTIONS', 'FORCE_REBUILD_TOPIC', fallback=False)
        self.update_topic = self.config.getboolean('OPTIONS', 'UPDATE_TOPIC', fallback=False)
        self.clear_cache = self.config.getboolean('OPTIONS', 'CLEAR_CACHE', fallback=False)

        # Cache und API initialisieren
        self.cache = {} if self.clear_cache else self._load_cache()
        self.api = WikiAPI(self.endpoint, self.max_rps, self.cache)

        # Ground-Truth Pfad
        self.ground_truth_path = self.root / 'exports' / 'ground_truth.json'
        self.ground_truth_path.parent.mkdir(parents=True, exist_ok=True)

    def _get_list(self, section, key):
        """Extrahiert Liste aus Konfiguration (unterstützt Komma und Semikolon)"""
        raw = self.config.get(section, key, fallback='').strip()
        # Unterstütze sowohl Komma als auch Semikolon als Trenner
        # und mehrzeilige Einträge
        raw = raw.replace('\n', ',').replace(';', ',')
        items = []
        for item in raw.split(','):
            item = item.strip()
            if item:
                items.append(item)
        return items

    def _load_cache(self):
        """Lädt Cache aus Datei"""
        if self.cache_path.exists():
            try:
                return json.loads(self.cache_path.read_text(encoding='utf-8'))
            except:
                pass
        return {'categorymembers': {}, 'redirects': {}}

    def _save_cache(self):
        """Speichert Cache strukturiert"""
        # Duplikate entfernen und sortieren
        for key, value in list(self.cache.get('categorymembers', {}).items()):
            if isinstance(value, list):
                self.cache['categorymembers'][key] = sorted(set(value))

        # Identity-Redirects entfernen
        redirects = self.cache.get('redirects', {})
        self.cache['redirects'] = {k: v for k, v in redirects.items() if k != v}

        # Speichern
        self.cache_path.write_text(
            json.dumps(self.cache, ensure_ascii=False, indent=2, sort_keys=True),
            encoding='utf-8'
        )

    def _is_category_relevant(self, category_name, article_count):
        """Prüft ob Kategorie relevant ist mit deterministischen Regeln"""
        # Filter 1: Meta-Kategorien ausschließen (Regex)
        if any(pattern.search(category_name) for pattern in self.exclude_patterns):
            return False

        # Filter 2: Größenfilter (zu klein oder zu groß)
        if article_count < self.min_articles_per_category:
            return False
        if article_count > self.max_articles_per_category:
            return False

        # Filter 3: Ausschluss-Keywords (z.B. "Person", "Film", "Mythologie")
        if self.exclude_category_keywords:
            category_lower = category_name.lower()
            if any(kw.lower() in category_lower for kw in self.exclude_category_keywords):
                return False

        return True

    def build_ground_truth(self):
        """Baut optimierte Ground-Truth-Menge auf"""
        print("Baue Ground-Truth-Menge auf...")

        topic_titles = set()
        queue = deque()
        visited = set()
        category_stats = {}

        # Whitelist-Artikel direkt hinzufügen (falls konfiguriert)
        if self.core_articles_whitelist:
            for article_ref in self.core_articles_whitelist:
                # Normalisiere URLs oder Titel
                norm_title = normalize_wiki_title(article_ref)
                if norm_title:
                    topic_titles.add(norm_title)
            print(f"  Whitelist-Artikel hinzugefügt: {len(topic_titles)}")

        # Seeds in Queue
        print(f"  Seed-Kategorien: {len(self.seed_categories)}")
        for seed in self.seed_categories:
            seed = seed.strip()
            if seed and not any(p.search(seed) for p in self.exclude_patterns):
                queue.append((seed, 0))
                print(f"    • {seed}")

        if not queue:
            print("  WARNUNG: Keine gültigen Seed-Kategorien gefunden!")
            print(f"  Seed-Kategorien aus Config: {self.seed_categories}")
            print(f"  Exclude-Patterns: {[p.pattern for p in self.exclude_patterns]}")
            return topic_titles

        # Statistik für Debugging
        rejected_categories = {
            'too_few_articles': [],
            'too_many_articles': [],
            'excluded_keyword': [],
            'excluded_pattern': []
        }

        while queue:
            category, depth = queue.popleft()
            if category in visited:
                continue
            visited.add(category)

            # Fortschritt anzeigen
            if len(visited) % 10 == 0:
                print(f"  Verarbeitet: {len(visited)} Kategorien, {len(topic_titles)} Artikel")

            # Artikel der Kategorie holen
            articles = self.api.get_category_articles(category)
            article_count = len(articles)

            # Detaillierte Relevanz-Prüfung mit Grund-Tracking
            is_relevant = True
            rejection_reason = None

            # Prüfe Ausschlusskriterien der Reihe nach
            if any(p.search(category) for p in self.exclude_patterns):
                is_relevant = False
                rejection_reason = 'excluded_pattern'
            elif article_count < self.min_articles_per_category:
                is_relevant = False
                rejection_reason = 'too_few_articles'
            elif article_count > self.max_articles_per_category:
                is_relevant = False
                rejection_reason = 'too_many_articles'
            elif self.exclude_category_keywords:
                category_lower = category.lower()
                if any(kw.lower() in category_lower for kw in self.exclude_category_keywords):
                    is_relevant = False
                    rejection_reason = 'excluded_keyword'

            if is_relevant:
                # Redirects auflösen
                mapping = resolve_redirects_batch(self.api, articles)
                resolved = set(mapping.values())

                # Statistik sammeln
                category_stats[category] = {
                    'depth': depth,
                    'articles': article_count,
                    'added': len(resolved - topic_titles)
                }

                topic_titles.update(resolved)
            else:
                # Rejection tracking
                if rejection_reason:
                    rejected_categories[rejection_reason].append(category)

            # Unterkategorien hinzufügen (wenn Tiefe erlaubt)
            if depth < self.max_depth:
                for subcat in self.api.get_subcategories(category):
                    if not any(p.search(subcat) for p in self.exclude_patterns):
                        queue.append((subcat, depth + 1))

        # Statistik ausgeben
        print(f"\nGround-Truth aufgebaut:")
        print(f"  Kategorien verarbeitet: {len(visited)}")
        print(f"  Kategorien akzeptiert: {len(category_stats)}")
        print(f"  Artikel in Ground-Truth: {len(topic_titles)}")

        # Rejection-Gründe zeigen
        print("\n  Kategorien-Filterung:")
        for reason, cats in rejected_categories.items():
            if cats:
                print(f"    {reason}: {len(cats)} Kategorien")
                if len(cats) <= 3:
                    for cat in cats:
                        print(f"      - {cat}")

        # Top relevante Kategorien zeigen
        if category_stats:
            top_cats = sorted(category_stats.items(),
                            key=lambda x: x[1]['added'], reverse=True)[:5]
            print("\n  Top-Kategorien nach hinzugefügten Artikeln:")
            for cat, stats in top_cats:
                print(f"    • {cat}: {stats['added']} Artikel (Tiefe {stats['depth']})")

        return topic_titles

    def evaluate_strategies(self):
        """Evaluiert alle Crawling-Strategien gegen Ground-Truth"""
        # Ground-Truth laden oder erstellen
        if self.force_rebuild or self.update_topic or not self.ground_truth_path.exists():
            ground_truth = self.build_ground_truth()

            if self.update_topic and self.ground_truth_path.exists():
                # Mit vorheriger Version vereinigen
                previous = set(json.loads(self.ground_truth_path.read_text(encoding='utf-8')))
                ground_truth = previous | ground_truth

            # Speichern
            self.ground_truth_path.write_text(
                json.dumps(sorted(ground_truth), ensure_ascii=False, indent=2),
                encoding='utf-8'
            )
            self._save_cache()
        else:
            ground_truth = set(json.loads(self.ground_truth_path.read_text(encoding='utf-8')))

        # Crawling-Exporte laden
        exports = self._load_exports()

        # Evaluieren
        print("\nEvaluationsergebnisse:")
        print("-" * 50)

        results = {}
        for strategy in sorted(exports.keys()):
            metrics = self._calculate_metrics(exports[strategy], ground_truth)
            results[strategy] = metrics

            # Ausgabe
            print(f"\n{strategy}:")
            print(f"  Seiten gesamt:    {metrics['total']:>6}")
            print(f"  Relevant (TP):    {metrics['tp']:>6} ({metrics['precision']:.1%})")
            print(f"  Irrelevant (FP):  {metrics['fp']:>6}")
            print(f"  Verpasst (FN):    {metrics['fn']:>6}")
            print(f"  Precision:        {metrics['precision']:>6.4f}")
            print(f"  Recall:           {metrics['recall']:>6.4f}")

            if metrics['precision'] > 0 and metrics['recall'] > 0:
                f1 = 2 * (metrics['precision'] * metrics['recall']) / (metrics['precision'] + metrics['recall'])
                print(f"  F1-Score:         {f1:>6.4f}")

        return results

    def _load_exports(self):
        """Lädt Crawling-Exporte"""
        exports = defaultdict(list)

        for path in sorted(self.exports_dir.glob(self.file_pattern)):
            data = json.loads(path.read_text(encoding='utf-8'))
            strategy = data.get('summary', {}).get('spider', 'unknown')
            pages = data.get('summary', {}).get('pages', [])
            exports[strategy].extend(pages)

        return exports

    def _calculate_metrics(self, pages, ground_truth):
        """Berechnet Evaluationsmetriken"""
        tp = fp = fn = 0
        total = 0

        seen_titles = set()

        for page in pages:
            # Titel normalisieren
            url = page.get('url', '')
            title = page.get('title', '')
            norm_title = normalize_wiki_title(url) or normalize_wiki_title(title)

            if not norm_title:
                continue

            total += 1
            seen_titles.add(norm_title)

            # Evaluieren (nur basierend auf Ground-Truth-Zugehörigkeit)
            if norm_title in ground_truth:
                tp += 1
            else:
                fp += 1

        # False Negatives: Relevante Titel, die nicht gefunden wurden
        fn = len(ground_truth - seen_titles)

        # Metriken berechnen
        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        return {
            'total': total,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall
        }


class WikiAPI:
    """Wrapper für MediaWiki API mit Caching und Rate-Limiting"""

    def __init__(self, endpoint, max_rps, cache):
        self.endpoint = endpoint
        self.min_interval = 1.0 / max_rps if max_rps > 0 else 0
        self.last_request = 0.0
        self.cache = cache
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'BA-Evaluation/1.0 (University Research)'
        })

    def _throttle(self):
        """Rate-Limiting"""
        elapsed = time.time() - self.last_request
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self.last_request = time.time()

    def _request(self, params):
        """API-Request mit Retry-Logik"""
        for attempt in range(5):
            try:
                self._throttle()
                response = self.session.get(self.endpoint, params=params, timeout=30)
                response.raise_for_status()
                return response.json()
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(0.5 * (attempt + 1))

    def get_category_articles(self, category):
        """Holt alle Artikel einer Kategorie"""
        cache_key = f"ARTS::{category}"
        if cache_key in self.cache.get('categorymembers', {}):
            return self.cache['categorymembers'][cache_key]

        articles = []
        continuation = None

        while True:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Kategorie:{category}',
                'cmnamespace': '0',
                'cmlimit': '500',
                'format': 'json'
            }
            if continuation:
                params['cmcontinue'] = continuation

            data = self._request(params)

            for member in data.get('query', {}).get('categorymembers', []):
                if member.get('ns') == 0 and 'title' in member:
                    articles.append(member['title'])

            continuation = data.get('continue', {}).get('cmcontinue')
            if not continuation:
                break

        self.cache.setdefault('categorymembers', {})[cache_key] = articles
        return articles

    def get_subcategories(self, category):
        """Holt Unterkategorien"""
        subcategories = []
        continuation = None

        while True:
            params = {
                'action': 'query',
                'list': 'categorymembers',
                'cmtitle': f'Kategorie:{category}',
                'cmnamespace': '14',
                'cmlimit': '500',
                'format': 'json'
            }
            if continuation:
                params['cmcontinue'] = continuation

            data = self._request(params)

            for member in data.get('query', {}).get('categorymembers', []):
                if member.get('ns') == 14 and 'title' in member:
                    # Entferne "Kategorie:" Präfix
                    subcategories.append(member['title'].split(':', 1)[1])

            continuation = data.get('continue', {}).get('cmcontinue')
            if not continuation:
                break

        return subcategories


def normalize_wiki_title(input_string):
    """Normalisiert Wikipedia-URL oder Titel zu kanonischem Format"""
    if not input_string:
        return None

    if input_string.startswith('http'):
        parsed = urlparse(input_string)
        if parsed.netloc.lower() != 'de.wikipedia.org':
            return None
        if not parsed.path.startswith('/wiki/'):
            return None
        title = unquote(parsed.path[6:])  # "/wiki/" entfernen
    else:
        title = input_string

    # Underscores zu Spaces, trimmen
    title = title.replace('_', ' ').strip()
    if not title:
        return None

    # Erster Buchstabe groß
    return title[0].upper() + title[1:]


def resolve_redirects_batch(api, titles):
    """Löst Redirects für mehrere Titel auf"""
    titles = [t for t in titles if t]
    if not titles:
        return {}

    mapping = {}
    chunk_size = 50

    for i in range(0, len(titles), chunk_size):
        chunk = titles[i:i + chunk_size]

        data = api._request({
            'action': 'query',
            'titles': '|'.join(chunk),
            'redirects': '1',
            'format': 'json'
        })

        # Kanonische Titel
        pages = data.get('query', {}).get('pages', {})
        for page_data in pages.values():
            title = page_data.get('title')
            if title:
                mapping.setdefault(title, title)

        # Redirects
        for redirect in data.get('query', {}).get('redirects', []):
            mapping[redirect['from']] = redirect['to']

        # Fallback für nicht gefundene
        for title in chunk:
            mapping.setdefault(title, title)

    # Cache aktualisieren
    api.cache.setdefault('redirects', {}).update(
        {k: v for k, v in mapping.items() if k != v}
    )

    return mapping


if __name__ == '__main__':
    evaluator = WikiGroundTruthEvaluator('../evaluation_config.ini')
    evaluator.evaluate_strategies()