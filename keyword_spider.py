import math

from base_spider import BaseTopicalSpider
import re
from spacy.matcher import PhraseMatcher


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

        # Initialisiere SpaCy PhraseMatcher
        if self.nlp:
            self.matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
            # Füge Keywords als Patterns hinzu
            patterns = [self.nlp.make_doc(keyword) for keyword in self.keywords]
            self.matcher.add("KEYWORDS", patterns)
        else:
            self.matcher = None

        print(f"Keywords: {', '.join(self.keywords)}")
        self.write_to_report(f"Keywords: {', '.join(self.keywords)}\n")

    def calculate_text_relevance(self, text):
        """
        Berechnet Relevanz basierend auf Keyword-Frequenz mit SpaCy PhraseMatcher
        Höhere Anzahl von Keyword-Treffern = höherer Score
        """
        if not text or not self.matcher:
            return 0.0

        # Textvorverarbeitung
        processed_text = self.preprocess_text(text)

        if not processed_text:
            return 0.0

        # Erstelle SpaCy Doc
        doc = self.nlp(processed_text)

        # Finde alle Keyword-Matches
        matches = self.matcher(doc)
        total_matches = len(matches)

        # Berechne Wortanzahl
        word_count = len([token for token in doc if not token.is_space])

        if word_count == 0:
            return 0.0

        # Normalisiere Score (Treffer pro 100 Wörter)
        normalized_score = (total_matches / word_count) * 100

        # Begrenzen auf Bereich [0, 1]
        # Annahme: >10 Treffer pro 100 Wörter = maximale Relevanz
        return min(1.0, normalized_score / 10)

    def calculate_parent_relevance(self, title, headings, paragraphs):
        """
        Optimierte Elterndokument-Bewertung nur mit Multiplikatoren
        Berechnet gewichtete Trefferanzahl / gewichtete Wortanzahl
        """
        if not self.matcher:
            return 0.0

        total_weighted_matches = 0
        total_weighted_words = 0

        # Verarbeite Titel
        if title:
            processed_title = self.preprocess_text(title)
            if processed_title:
                doc = self.nlp(processed_title)
                matches = len(self.matcher(doc))
                words = len([token for token in doc if not token.is_space])
                total_weighted_matches += matches * self.title_multiplier
                total_weighted_words += words * self.title_multiplier

        # Verarbeite Überschriften
        if headings:
            processed_headings = self.preprocess_text(headings)
            if processed_headings:
                doc = self.nlp(processed_headings)
                matches = len(self.matcher(doc))
                words = len([token for token in doc if not token.is_space])
                total_weighted_matches += matches * self.heading_multiplier
                total_weighted_words += words * self.heading_multiplier

        # Verarbeite Paragraphen
        if paragraphs:
            processed_paragraphs = self.preprocess_text(paragraphs)
            if processed_paragraphs:
                doc = self.nlp(processed_paragraphs)
                matches = len(self.matcher(doc))
                words = len([token for token in doc if not token.is_space])
                total_weighted_matches += matches * self.paragraph_multiplier
                total_weighted_words += words * self.paragraph_multiplier

        # Berechne finalen Score
        if total_weighted_words == 0:
            return 0.0

        # Relevanz = gewichtete Treffer / gewichtete Wörter
        relevance_score = total_weighted_matches / total_weighted_words

        # Normalisiere auf [0, 1] - Annahme: >0.1 (10% Keywords) = maximale Relevanz
        return min(1.0, math.log(1 + relevance_score * 100) / math.log(11))