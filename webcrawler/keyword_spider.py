from webcrawler.base_spider import BaseTopicalSpider
from collections import Counter


class KeywordSpider(BaseTopicalSpider):
    """Keyword-basierte Crawling-Strategie"""

    name = 'keyword_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lade Keywords aus Konfiguration (nur Einzelwörter für Performance)
        keywords_raw = [kw.strip().lower() for kw in
                        self.config['KEYWORD']['KEYWORDS'].split(',')]
        # Filtere nur Einzelwort-Keywords
        self.keywords = [kw for kw in keywords_raw if ' ' not in kw]

        # Lade Multiplikatoren aus Konfiguration
        self.title_multiplier = int(self.config['KEYWORD']['TITLE_MULTIPLIER'])
        self.heading_multiplier = int(self.config['KEYWORD']['HEADING_MULTIPLIER'])
        self.paragraph_multiplier = int(self.config['KEYWORD']['PARAGRAPH_MULTIPLIER'])

        # Lade Normalisierungsfaktoren aus Konfiguration
        self.text_norm_factor = float(self.config['KEYWORD']['TEXT_NORM_FACTOR'])
        self.parent_norm_factor = float(self.config['KEYWORD']['PARENT_NORM_FACTOR'])

        print(f"Keywords (Einzelwörter): {', '.join(self.keywords)}")
        self.write_to_report(f"Keywords: {', '.join(self.keywords)}\n")

    def calculate_text_relevance(self, text):
        """
        Berechnet Relevanz basierend auf Keyword-Frequenz mit Python-Bordmitteln
        Höhere Anzahl von Keyword-Treffern = höherer Score
        """
        if not text:
            return 0.0

        # Textvorverarbeitung
        processed_text = self.preprocess_text(text)

        if not processed_text:
            return 0.0

        # Tokenisiere und zähle mit Counter
        words = processed_text.split()
        word_count = len(words)

        if word_count == 0:
            return 0.0

        # Zähle Keyword-Treffer effizient mit Counter
        word_counter = Counter(words)
        total_matches = sum(word_counter[keyword] for keyword in self.keywords)

        # Normalisiere Score (Treffer pro 100 Wörter)
        normalized_score = (total_matches / word_count) * 100

        # Konfigurierbare Normalisierung
        return min(1.0, normalized_score / self.text_norm_factor)

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """
        Optimierte Elterndokument-Bewertung mit Python-Bordmitteln
        Berechnet gewichtete Trefferanzahl / gewichtete Wortanzahl
        """
        total_weighted_matches = 0
        total_weighted_words = 0

        # Verarbeite Titel
        if title:
            processed_title = self.preprocess_text(title)
            if processed_title:
                words = processed_title.split()
                word_counter = Counter(words)
                matches = sum(word_counter[keyword] for keyword in self.keywords)
                word_count = len(words)
                total_weighted_matches += matches * self.title_multiplier
                total_weighted_words += word_count * self.title_multiplier

        # Verarbeite Überschriften
        if headings:
            processed_headings = self.preprocess_text(headings)
            if processed_headings:
                words = processed_headings.split()
                word_counter = Counter(words)
                matches = sum(word_counter[keyword] for keyword in self.keywords)
                word_count = len(words)
                total_weighted_matches += matches * self.heading_multiplier
                total_weighted_words += word_count * self.heading_multiplier

        # Verarbeite Paragraphen
        if paragraphs:
            processed_paragraphs = self.preprocess_text(paragraphs)
            if processed_paragraphs:
                words = processed_paragraphs.split()
                word_counter = Counter(words)
                matches = sum(word_counter[keyword] for keyword in self.keywords)
                word_count = len(words)
                total_weighted_matches += matches * self.paragraph_multiplier
                total_weighted_words += word_count * self.paragraph_multiplier

        # Berechne finalen Score
        if total_weighted_words == 0:
            return 0.0

        # Relevanz = gewichtete Treffer / gewichtete Wörter
        relevance_score = total_weighted_matches / total_weighted_words

        # Konfigurierbare Normalisierung
        return min(1.0, relevance_score * self.parent_norm_factor)