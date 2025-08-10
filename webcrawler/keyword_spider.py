from webcrawler.base_spider import BaseTopicalSpider


class KeywordSpider(BaseTopicalSpider):
    """
    Boolean Keyword Crawler für themenbasiertes Web-Crawling

    Verwendet eine vordefinierte Keyword-Liste und bewertet Texte basierend auf
    der Präsenz (nicht Frequenz) dieser Keywords. Jeder Textbereich hat
    angepasste Schwellwerte für eine dreistufige Bewertung.
    """

    name = 'keyword_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lade Keywords aus Konfiguration und konvertiere zu Set
        keywords_raw = [kw.strip().lower() for kw in
                        self.config['KEYWORD']['KEYWORDS'].split(',')]
        self.keywords = set(kw for kw in keywords_raw if kw and ' ' not in kw)

        # Gewichtungen aus WEIGHTS Sektion (von Basisklasse verwendet)
        self.title_weight = float(self.config['WEIGHTS']['TITLE_WEIGHT'])
        self.heading_weight = float(self.config['WEIGHTS']['HEADING_WEIGHT'])
        self.paragraph_weight = float(self.config['WEIGHTS']['PARAGRAPH_WEIGHT'])

        # Boolean Schwellwerte für verschiedene Textbereiche
        # Ankertext (für calculate_text_relevance)
        self.anchor_min = int(self.config['KEYWORD'].get('ANCHOR_MIN_KEYWORDS', '1'))
        self.anchor_max = int(self.config['KEYWORD'].get('ANCHOR_MAX_KEYWORDS', '2'))

        # Titel
        self.title_min = int(self.config['KEYWORD'].get('TITLE_MIN_KEYWORDS', '1'))
        self.title_max = int(self.config['KEYWORD'].get('TITLE_MAX_KEYWORDS', '2'))

        # Überschriften
        self.heading_min = int(self.config['KEYWORD'].get('HEADING_MIN_KEYWORDS', '1'))
        self.heading_max = int(self.config['KEYWORD'].get('HEADING_MAX_KEYWORDS', '2'))

        # Paragraphen
        self.paragraph_min = int(self.config['KEYWORD'].get('PARAGRAPH_MIN_KEYWORDS', '3'))
        self.paragraph_max = int(self.config['KEYWORD'].get('PARAGRAPH_MAX_KEYWORDS', '6'))

        # Ausgabe der Konfiguration
        print(f"\n{'=' * 60}")
        print(f"Boolean Keyword Spider initialisiert")
        print(f"{'=' * 60}")
        print(f"Keywords: {len(self.keywords)} Stück geladen")
        print(f"\nSchwellwerte für dreistufige Bewertung:")
        print(f"  Ankertext:    0 → 0.0 | {self.anchor_min}+ → 0.5 | {self.anchor_max}+ → 1.0")
        print(f"  Titel:        0 → 0.0 | {self.title_min}+ → 0.5 | {self.title_max}+ → 1.0")
        print(f"  Überschriften: 0 → 0.0 | {self.heading_min}+ → 0.5 | {self.heading_max}+ → 1.0")
        print(f"  Paragraphen:  0 → 0.0 | {self.paragraph_min}+ → 0.5 | {self.paragraph_max}+ → 1.0")
        print(f"{'=' * 60}\n")

        # Report schreiben
        self.write_to_report(f"Boolean-Modell mit {len(self.keywords)} Keywords\n")
        self.write_to_report(f"Schwellwerte: Anker({self.anchor_min}/{self.anchor_max}), "
                             f"Titel({self.title_min}/{self.title_max}), "
                             f"Headings({self.heading_min}/{self.heading_max}), "
                             f"Paragraphs({self.paragraph_min}/{self.paragraph_max})\n")

    def calculate_text_relevance(self, text):
        """
        Berechnet Relevanz für Ankertexte von Links

        Bewertungsstufen:
        - 0 Keywords → 0.0 (irrelevant)
        - 1+ Keywords → 0.5 (relevant)
        - 2+ Keywords → 1.0 (hochrelevant)

        Args:
            text: Ankertext des Links

        Returns:
            float: Relevanzscore {0.0, 0.5, 1.0}
        """
        return self._calculate_boolean_score(text, self.anchor_min, self.anchor_max)

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """
        Berechnet gewichtete Relevanz des Elterndokuments

        Kombiniert die Bewertungen von Titel, Überschriften und Paragraphen
        mit konfigurierbaren Gewichtungen.

        Args:
            title: Seitentitel
            headings: Alle Überschriften (h1-h6) kombiniert
            paragraphs: Alle Paragraphen kombiniert

        Returns:
            float: Gewichteter Relevanzscore [0.0, 1.0]
        """
        # Berechne Scores für jeden Bereich mit angepassten Schwellwerten
        title_score = self._calculate_boolean_score(
            title, self.title_min, self.title_max
        ) if title else 0.0

        heading_score = self._calculate_boolean_score(
            headings, self.heading_min, self.heading_max
        ) if headings else 0.0

        paragraph_score = self._calculate_boolean_score(
            paragraphs, self.paragraph_min, self.paragraph_max
        ) if paragraphs else 0.0

        # Gewichtete Kombination (Summe der Gewichte = 1.0)
        weighted_score = (
                self.title_weight * title_score +
                self.heading_weight * heading_score +
                self.paragraph_weight * paragraph_score
        )

        return min(1.0, weighted_score)

    def _calculate_boolean_score(self, text, min_threshold, max_threshold):
        """
        Interne Methode für dreistufige Boolean-Bewertung

        Zählt unique Keywords (Präsenz, nicht Frequenz) und ordnet
        basierend auf Schwellwerten einen diskreten Score zu.

        Args:
            text: Zu bewertender Text
            min_threshold: Schwellwert für Score 0.5
            max_threshold: Schwellwert für Score 1.0

        Returns:
            float: Score aus {0.0, 0.5, 1.0}
        """
        if not text:
            return 0.0

        # Textvorverarbeitung (Lowercase, Stoppwörter entfernen)
        processed_text = self.preprocess_text(text)
        if not processed_text:
            return 0.0

        # Finde unique Keywords im Text
        text_words = set(processed_text.split())
        found_keywords = text_words.intersection(self.keywords)
        keyword_count = len(found_keywords)

        # Dreistufige Bewertung
        if keyword_count >= max_threshold:
            return 1.0  # Hochrelevant
        elif keyword_count >= min_threshold:
            return 0.5  # Relevant
        else:
            return 0.0  # Irrelevant