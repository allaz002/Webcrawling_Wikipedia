#!/usr/bin/env python
"""
Hauptskript zum Starten der Topical Crawler
Verwendung: python run_crawler.py [keyword|vectorspace|naivebayes]
"""

import sys
import os
from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from webcrawler.keyword_spider import KeywordSpider
from webcrawler.vectorspace_spider import VectorSpaceSpider
from webcrawler.naivebayes_spider import NaiveBayesSpider

def print_usage():
    """Zeigt die Verwendung"""
    print("""
    Verwendung:
    
    python run_crawler.py <strategie>
    
    Strategien:
    - keyword     : Keyword-basierte Strategie
    - vectorspace : Vektorraum-Modell mit Cosinus-Ähnlichkeit
    - naivebayes  : Naive Bayes Klassifikation
    
    Beispiel:
    python run_crawler.py keyword
    
    Konfiguration erfolgt über crawler_config.ini
    """)


def main():
    """Hauptfunktion"""
    
    # Prüfe Argumente
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
        
    strategy = sys.argv[1].lower()
    
    # Wähle Spider basierend auf Strategie
    spider_map = {
        'keyword': KeywordSpider,
        'vectorspace': VectorSpaceSpider,
        'naivebayes': NaiveBayesSpider
    }
    
    if strategy not in spider_map:
        print(f"Fehler: Unbekannte Strategie '{strategy}'")
        print_usage()
        sys.exit(1)
        
    # Prüfe ob Konfigurationsdatei existiert
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'crawler_config.ini')

    if not os.path.exists(config_path):
        print("Fehler: crawler_config.ini nicht gefunden!")
        print("Bitte erstellen Sie die Konfigurationsdatei.")
        sys.exit(1)
        
    # Initialisiere Scrapy Process
    settings = get_project_settings()
    process = CrawlerProcess(settings)
    
    # Starte gewählten Spider
    spider_class = spider_map[strategy]
    print(f"\nStarte {strategy.upper()} Crawler...\n")
    
    process.crawl(spider_class)
    process.start()
    
    print(f"\nCrawling abgeschlossen. Siehe reports/ und exports/ für Ergebnisse.\n")


if __name__ == '__main__':
    main()