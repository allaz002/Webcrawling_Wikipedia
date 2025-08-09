from webcrawler.base_spider import BaseTopicalSpider


class KeywordSpider(BaseTopicalSpider):
    """
    Klassischer Boolean Keyword Crawler
    Nutzt nur Keyword-Präsenz (nicht Frequenz) für binäre Relevanzentscheidungen
    """

    name = 'keyword_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lade Keywords aus Konfiguration
        keywords_raw = [kw.strip().lower() for kw in
                        self.config['KEYWORD']['KEYWORDS'].split(',')]
        # Als Set für O(1) Lookup-Performance
        self.keywords = set(kw for kw in keywords_raw if ' ' not in kw)

        # Gewichtungen aus WEIGHTS Sektion
        self.title_weight = float(self.config['WEIGHTS']['TITLE_WEIGHT'])
        self.heading_weight = float(self.config['WEIGHTS']['HEADING_WEIGHT'])
        self.paragraph_weight = float(self.config['WEIGHTS']['PARAGRAPH_WEIGHT'])

        # Boolean Schwellwerte - angepasst pro Textfeld
        # Titel: kurze Texte
        self.title_min = int(self.config['KEYWORD'].get('TITLE_MIN_KEYWORDS', 1))
        self.title_high = int(self.config['KEYWORD'].get('TITLE_HIGH_KEYWORDS', 2))

        # Überschriften: mittlere Länge
        self.heading_min = int(self.config['KEYWORD'].get('HEADING_MIN_KEYWORDS', 2))
        self.heading_high = int(self.config['KEYWORD'].get('HEADING_HIGH_KEYWORDS', 4))

        # Paragraphen: lange Texte
        self.paragraph_min = int(self.config['KEYWORD'].get('PARAGRAPH_MIN_KEYWORDS', 3))
        self.paragraph_high = int(self.config['KEYWORD'].get('PARAGRAPH_HIGH_KEYWORDS', 6))

        print(f"Boolean Keyword Spider mit {len(self.keywords)} Keywords")
        print(f"Schwellwerte - Titel: {self.title_min}/{self.title_high}, "
              f"Headings: {self.heading_min}/{self.heading_high}, "
              f"Paragraphs: {self.paragraph_min}/{self.paragraph_high}")
        self.write_to_report(f"Klassisches Boolean-Modell mit {len(self.keywords)} Keywords\n")
        self.write_to_report(f"Angepasste Schwellwerte pro Textfeld\n")

    def calculate_text_relevance(self, text):
        """
        Boolean-Modell: Zählt unique Keywords (Präsenz, nicht Frequenz)
        Nutzt Paragraph-Schwellwerte als Standard
        Returns: 0.0 (irrelevant), 0.5 (relevant), 1.0 (hochrelevant)
        """
        return self._calculate_boolean_relevance(text, self.paragraph_min, self.paragraph_high)

    def _calculate_boolean_relevance(self, text, min_threshold, high_threshold):
        """
        Interne Methode für Boolean-Bewertung mit variablen Schwellwerten
        """
        if not text:
            return 0.0

        processed_text = self.preprocess_text(text)
        if not processed_text:
            return 0.0

        # Unique Keywords im Text finden
        words = set(processed_text.split())
        keywords_found = words.intersection(self.keywords)
        unique_keyword_count = len(keywords_found)

        # Dreistufige Boolean-Entscheidung
        if unique_keyword_count >= high_threshold:
            return 1.0  # Hochrelevant
        elif unique_keyword_count >= min_threshold:
            return 0.5  # Relevant
        else:
            return 0.0  # Irrelevant

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """
        Berechnet gewichtete Relevanz mit angepassten Schwellwerten pro Textfeld
        Berücksichtigt die unterschiedlichen Textlängen realistisch
        """
        # Boolean-Bewertung mit feldspezifischen Schwellwerten
        title_relevance = self._calculate_boolean_relevance(
            title, self.title_min, self.title_high
        ) if title else 0.0

        heading_relevance = self._calculate_boolean_relevance(
            headings, self.heading_min, self.heading_high
        ) if headings else 0.0

        paragraph_relevance = self._calculate_boolean_relevance(
            paragraphs, self.paragraph_min, self.paragraph_high
        ) if paragraphs else 0.0

        # Gewichtete Summe (Gewichte summieren sich zu 1.0)
        weighted_relevance = (
                self.title_weight * title_relevance +
                self.heading_weight * heading_relevance +
                self.paragraph_weight * paragraph_relevance
        )

        return min(1.0, weighted_relevance)