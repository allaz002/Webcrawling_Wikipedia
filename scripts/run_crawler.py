
"""
Startet die topical Crawler
Verwendung: python run_crawler.py [keyword|vectorspace|naivebayes]
"""

import sys
import os

# Projektverzeichnis finden
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from scrapy.crawler import CrawlerProcess
from scrapy.utils.project import get_project_settings
from webcrawler.keyword_spider import KeywordSpider
from webcrawler.vectorspace_spider import VectorSpaceSpider
from webcrawler.naivebayes_spider import NaiveBayesSpider

def print_usage():
    """Zeigt Verwendung"""
    print("""Verwendung: python run_crawler.py [keyword|vectorspace|naivebayes]""")

def main():
    """Hauptfunktion"""
    
    # Argumente prüfen
    if len(sys.argv) != 2:
        print_usage()
        sys.exit(1)
        
    strategy = sys.argv[1].lower()
    
    # Strategie mappen
    spider_map = {
        'keyword': KeywordSpider,
        'vectorspace': VectorSpaceSpider,
        'naivebayes': NaiveBayesSpider
    }
    
    if strategy not in spider_map:
        print(f"Fehler: Unbekannte Strategie '{strategy}'")
        print_usage()
        sys.exit(1)
        
    # Existenzprüfung Konfigurationsdatei
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'crawler_config.ini')

    if not os.path.exists(config_path):
        print("Fehler: crawler_config.ini nicht gefunden!")
        sys.exit(1)
        
    # Scrapy-Prozess initialisieren
    settings = get_project_settings()
    process = CrawlerProcess(settings)

    # Spider starten
    spider_class = spider_map[strategy]
    print(f"\nStarte {strategy.upper()} Crawler...\n")
    
    process.crawl(spider_class)
    process.start()
    
    print(f"\nBeende {strategy.upper()} Crawler...\n")


if __name__ == '__main__':
    main()