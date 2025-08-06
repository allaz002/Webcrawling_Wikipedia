from base_spider import BaseTopicalSpider
import re


class KeywordSpider(BaseTopicalSpider):
    """Keyword-basierte Crawling-Strategie"""

    name = 'keyword_crawler'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Lade Keywords aus Konfiguration
        self.keywords = [kw.strip().lower() for kw in
                         self.config['KEYWORDS']['KEYWORDS'].split(',')]

        # Lade Multiplikatoren aus Konfiguration
        self.title_multiplier = int(self.config['MULTIPLIERS']['TITLE_MULTIPLIER'])
        self.heading_multiplier = int(self.config['MULTIPLIERS']['HEADING_MULTIPLIER'])
        self.paragraph_multiplier = int(self.config['MULTIPLIERS']['PARAGRAPH_MULTIPLIER'])

        print(f"Keywords: {', '.join(self.keywords)}")
        self.write_to_report(f"Keywords: {', '.join(self.keywords)}\n")

    def calculate_text_relevance(self, text):
        """
        Berechnet Relevanz basierend auf Keyword-Frequenz
        Höhere Anzahl von Keyword-Treffern = höherer Score
        """
        if not text:
            return 0.0

        # Textvorverarbeitung
        processed_text = self.preprocess_text(text)

        if not processed_text:
            return 0.0

        # Zähle Keyword-Treffer
        total_matches = 0
        text_words = processed_text.split()
        text_length = len(text_words)

        if text_length == 0:
            return 0.0

        for keyword in self.keywords:
            # Exakte Wort-Matches zählen
            keyword_words = keyword.split()

            if len(keyword_words) == 1:
                # Einzelnes Keyword
                total_matches += text_words.count(keyword)
            else:
                # Mehrwort-Keyword (Phrase)
                for i in range(len(text_words) - len(keyword_words) + 1):
                    if text_words[i:i + len(keyword_words)] == keyword_words:
                        total_matches += 1

        # Normalisiere Score (Treffer pro 100 Wörter)
        # Verhindert dass längere Texte automatisch höhere Scores bekommen
        normalized_score = (total_matches / text_length) * 100

        # Begrenzen auf Bereich [0, 1]
        # Annahme: >10 Treffer pro 100 Wörter = maximale Relevanz
        return min(1.0, normalized_score / 10)

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """
        Überschreibt Elterndokument-Bewertung für Keyword-Strategie
        Verwendet konfigurierbare Multiplikatoren
        """
        # Berechne individuelle Scores
        title_score = self.calculate_text_relevance(title)
        heading_score = self.calculate_text_relevance(headings)
        paragraph_score = self.calculate_text_relevance(paragraphs)

        # Gewichtete Summe mit konfigurierbaren Multiplikatoren
        weighted_score = (
                self.title_weight * title_score * self.title_multiplier +
                self.heading_weight * heading_score * self.heading_multiplier +
                self.paragraph_weight * paragraph_score * self.paragraph_multiplier
        )

        # Normalisiere auf [0, 1]
        total_weight = (
                self.title_weight * self.title_multiplier +
                self.heading_weight * self.heading_multiplier +
                self.paragraph_weight * self.paragraph_multiplier
        )

        return min(1.0, weighted_score / total_weight) if total_weight > 0 else 0.0