from webcrawler.base_spider import BaseTopicalSpider


class KeywordSpider(BaseTopicalSpider):
    """
    Boolean Keyword Crawler für themenbasiertes Web-Crawling

    Verwendet eine vordefinierte Keyword-Liste und bewertet Texte basierend auf
    der Präsenz (nicht Frequenz) dieser Keywords mit binärer Bewertung.
    """

    name = 'keyword_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lade Keywords aus Konfiguration und konvertiere zu Set
        keywords_raw = [kw.strip().lower() for kw in
                        self.config['KEYWORD']['KEYWORDS'].split(',')]
        self.keywords = set(kw for kw in keywords_raw if kw and ' ' not in kw)

        # Boolean Schwellwerte für verschiedene Textbereiche
        self.anchor_min = int(self.config['KEYWORD'].get('ANCHOR_MIN_KEYWORDS', '1'))
        self.title_min = int(self.config['KEYWORD'].get('TITLE_MIN_KEYWORDS', '1'))
        self.heading_min = int(self.config['KEYWORD'].get('HEADING_MIN_KEYWORDS', '1'))
        self.paragraph_min = int(self.config['KEYWORD'].get('PARAGRAPH_MIN_KEYWORDS', '3'))

        print(f"Keyword Spider: {len(self.keywords)} Keywords geladen")

    def calculate_text_relevance(self, text):
        """
        Berechnet Relevanz für Texte (primär Ankertexte)

        Returns:
            float: 1.0 wenn Mindestanzahl Keywords erfüllt, sonst 0.0
        """
        if not text:
            return 0.0

        processed_text = self.preprocess_text(text)
        if not processed_text:
            return 0.0

        # Finde unique Keywords im Text
        text_words = set(processed_text.split())
        found_keywords = text_words.intersection(self.keywords)

        # Binäre Bewertung für Ankertexte
        return 1.0 if len(found_keywords) >= self.anchor_min else 0.0

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """
        Berechnet gewichtete Relevanz des Elterndokuments mit binärer Bewertung
        """
        # Binäre Bewertung für jeden Bereich
        title_score = 0.0
        if title:
            processed = self.preprocess_text(title)
            if processed:
                words = set(processed.split())
                keywords_found = len(words.intersection(self.keywords))
                title_score = 1.0 if keywords_found >= self.title_min else 0.0

        heading_score = 0.0
        if headings:
            processed = self.preprocess_text(headings)
            if processed:
                words = set(processed.split())
                keywords_found = len(words.intersection(self.keywords))
                heading_score = 1.0 if keywords_found >= self.heading_min else 0.0

        paragraph_score = 0.0
        if paragraphs:
            processed = self.preprocess_text(paragraphs)
            if processed:
                words = set(processed.split())
                keywords_found = len(words.intersection(self.keywords))
                paragraph_score = 1.0 if keywords_found >= self.paragraph_min else 0.0

        # Gewichtete Kombination
        weighted_score = (
                self.title_weight * title_score +
                self.heading_weight * heading_score +
                self.paragraph_weight * paragraph_score
        )

        return min(1.0, weighted_score)