#!/usr/bin/env python
"""
Skript zur Visualisierung der Crawling-Ergebnisse
Erstellt Grafiken für wissenschaftliche Analysen
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib_venn as venn
import numpy as np
import configparser
from datetime import datetime
from pathlib import Path


class CrawlerPlotter:
    """Erstellt Visualisierungen für Crawler-Ergebnisse"""

    def __init__(self, config_file='crawler_config.ini'):
        """Initialisiert Plotter mit Konfiguration"""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        # Lade Konfiguration
        self.create_plots = self.config.getboolean('PLOTTING', 'CREATE_PLOTS', fallback=False)
        self.output_dir = self.config['PLOTTING']['OUTPUT_DIRECTORY']

        if not self.create_plots:
            print("Plotting ist deaktiviert in der Konfiguration")
            return

        # Erstelle Output-Verzeichnis
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Definiere konsistente Farben für Strategien
        self.colors = {
            'keyword': '#FF6B6B',  # Rot
            'vectorspace': '#4ECDC4',  # Türkis
            'naivebayes': '#45B7D1'  # Blau
        }

        # Matplotlib-Einstellungen für wissenschaftliche Darstellung
        plt.rcParams['figure.figsize'] = (10, 6)
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'

        # Timestamp für Dateinamen
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Sammle Daten
        self.data = self.load_all_data()

    def load_all_data(self):
        """Lädt alle JSON-Dateien aus dem exports Verzeichnis"""
        data = {}

        # Finde alle JSON-Dateien
        json_files = glob.glob('exports/*_crawler_*.json')

        for file_path in json_files:
            # Extrahiere Strategie-Namen aus Dateiname
            filename = os.path.basename(file_path)
            if 'keyword' in filename:
                strategy = 'keyword'
            elif 'vectorspace' in filename:
                strategy = 'vectorspace'
            elif 'naivebayes' in filename:
                strategy = 'naivebayes'
            else:
                continue

            # Lade JSON-Daten
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    json_data = json.load(f)
                    if 'summary' in json_data:
                        data[strategy] = json_data['summary']
                    else:
                        print(f"Warnung: Keine summary in {file_path}")
            except Exception as e:
                print(f"Fehler beim Laden von {file_path}: {e}")

        return data

    def plot_harvest_rate(self):
        """Erstellt Liniendiagramm der Harvest-Rate über besuchte Seiten"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy, strategy_data in self.data.items():
            if 'evaluation_log' not in strategy_data:
                continue

            eval_log = strategy_data['evaluation_log']
            if not eval_log:
                continue

            # Extrahiere Daten
            pages = [entry['pages_visited'] for entry in eval_log]
            harvest_rates = [entry['harvest_rate'] for entry in eval_log]

            # Plotte Linie
            ax.plot(pages, harvest_rates,
                    color=self.colors[strategy],
                    linewidth=2,
                    marker='o',
                    markersize=5,
                    label=f'{strategy.capitalize()} Spider',
                    alpha=0.8)

        # Formatierung
        ax.set_xlabel('Anzahl besuchter Seiten', fontsize=12)
        ax.set_ylabel('Harvest-Rate', fontsize=12)
        ax.set_title('Entwicklung der Harvest-Rate über den Crawl-Verlauf', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.set_ylim(0, max(ax.get_ylim()[1], 0.5))

        # Speichern
        filename = f"{self.output_dir}/harvest_rate_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Harvest-Rate Grafik gespeichert: {filename}")

    def plot_relevant_pages_time(self):
        """Erstellt Liniendiagramm der relevanten Seiten über Zeit"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy, strategy_data in self.data.items():
            if 'evaluation_log' not in strategy_data:
                continue

            eval_log = strategy_data['evaluation_log']
            if not eval_log:
                continue

            # Extrahiere Daten
            times = [entry['execution_time_in_seconds'] / 60 for entry in eval_log]  # In Minuten
            relevant = [entry['relevant_pages_found'] for entry in eval_log]

            # Plotte Linie
            ax.plot(times, relevant,
                    color=self.colors[strategy],
                    linewidth=2,
                    marker='s',
                    markersize=5,
                    label=f'{strategy.capitalize()} Spider',
                    alpha=0.8)

        # Formatierung
        ax.set_xlabel('Zeit (Minuten)', fontsize=12)
        ax.set_ylabel('Anzahl relevanter Seiten', fontsize=12)
        ax.set_title('Anzahl gefundener relevanter Seiten über Zeit', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.set_ylim(bottom=0)

        # Speichern
        filename = f"{self.output_dir}/relevant_pages_time_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Relevante Seiten über Zeit Grafik gespeichert: {filename}")

    def plot_average_relevance(self):
        """Erstellt Liniendiagramm der durchschnittlichen Relevanz"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy, strategy_data in self.data.items():
            if 'evaluation_log' not in strategy_data:
                continue

            eval_log = strategy_data['evaluation_log']
            if not eval_log:
                continue

            # Extrahiere Daten
            pages = [entry['pages_visited'] for entry in eval_log]
            avg_relevance = [entry['average_relevance'] for entry in eval_log]

            # Plotte Linie
            ax.plot(pages, avg_relevance,
                    color=self.colors[strategy],
                    linewidth=2,
                    marker='^',
                    markersize=5,
                    label=f'{strategy.capitalize()} Spider',
                    alpha=0.8)

        # Formatierung
        ax.set_xlabel('Anzahl besuchter Seiten', fontsize=12)
        ax.set_ylabel('Durchschnittliche Relevanz', fontsize=12)
        ax.set_title('Entwicklung der durchschnittlichen Relevanz', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.set_ylim(0, 1)

        # Relevanz-Bereiche markieren
        ax.axhspan(0, 0.25, alpha=0.1, color='red', label='Irrelevant')
        ax.axhspan(0.25, 0.5, alpha=0.1, color='orange', label='Mäßig relevant')
        ax.axhspan(0.5, 0.75, alpha=0.1, color='yellow', label='Stark relevant')
        ax.axhspan(0.75, 1.0, alpha=0.1, color='green', label='Sehr relevant')

        # Speichern
        filename = f"{self.output_dir}/average_relevance_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Durchschnittliche Relevanz Grafik gespeichert: {filename}")

    def plot_overlap_venn(self):
        """Erstellt Venn-Diagramm für Überlappung der gefundenen Seiten"""
        # Sammle URLs für jede Strategie
        urls_by_strategy = {}

        for strategy, strategy_data in self.data.items():
            if 'pages' not in strategy_data:
                continue

            urls_by_strategy[strategy] = set([page['url'] for page in strategy_data['pages']])

        if len(urls_by_strategy) < 2:
            print("Nicht genug Daten für Overlap-Diagramm")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        if len(urls_by_strategy) == 3:
            # 3-Wege Venn-Diagramm
            sets = [urls_by_strategy.get('keyword', set()),
                    urls_by_strategy.get('vectorspace', set()),
                    urls_by_strategy.get('naivebayes', set())]

            labels = ['Keyword', 'VectorSpace', 'NaiveBayes']

            # Erstelle Venn-Diagramm
            v = venn.venn3(sets, labels, ax=ax)

            # Färbe Kreise
            if v.get_patch_by_id('100'):
                v.get_patch_by_id('100').set_color(self.colors['keyword'])
                v.get_patch_by_id('100').set_alpha(0.5)
            if v.get_patch_by_id('010'):
                v.get_patch_by_id('010').set_color(self.colors['vectorspace'])
                v.get_patch_by_id('010').set_alpha(0.5)
            if v.get_patch_by_id('001'):
                v.get_patch_by_id('001').set_color(self.colors['naivebayes'])
                v.get_patch_by_id('001').set_alpha(0.5)

        elif len(urls_by_strategy) == 2:
            # 2-Wege Venn-Diagramm
            strategies = list(urls_by_strategy.keys())
            sets = [urls_by_strategy[strategies[0]], urls_by_strategy[strategies[1]]]
            labels = [s.capitalize() for s in strategies]

            v = venn.venn2(sets, labels, ax=ax)

            # Färbe Kreise
            if v.get_patch_by_id('10'):
                v.get_patch_by_id('10').set_color(self.colors[strategies[0]])
                v.get_patch_by_id('10').set_alpha(0.5)
            if v.get_patch_by_id('01'):
                v.get_patch_by_id('01').set_color(self.colors[strategies[1]])
                v.get_patch_by_id('01').set_alpha(0.5)

        # Berechne Overlap-Metriken
        all_urls = set()
        for urls in urls_by_strategy.values():
            all_urls.update(urls)

        # Titel mit Statistiken
        ax.set_title('Überlappung der gefundenen relevanten Seiten\n' +
                     f'Gesamt: {len(all_urls)} eindeutige URLs',
                     fontsize=14, fontweight='bold')

        # Speichern
        filename = f"{self.output_dir}/overlap_venn_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Overlap Venn-Diagramm gespeichert: {filename}")

    def create_all_plots(self):
        """Erstellt alle Grafiken"""
        if not self.create_plots:
            return

        if not self.data:
            print("Keine Daten zum Plotten gefunden")
            return

        print("\nErstelle Visualisierungen...")

        try:
            self.plot_harvest_rate()
        except Exception as e:
            print(f"Fehler bei Harvest-Rate Plot: {e}")

        try:
            self.plot_relevant_pages_time()
        except Exception as e:
            print(f"Fehler bei Relevant Pages Plot: {e}")

        try:
            self.plot_average_relevance()
        except Exception as e:
            print(f"Fehler bei Average Relevance Plot: {e}")

        try:
            self.plot_overlap_venn()
        except Exception as e:
            print(f"Fehler bei Overlap Venn Plot: {e}")

        print(f"\nAlle Grafiken wurden im Verzeichnis '{self.output_dir}' gespeichert")


def main():
    """Hauptfunktion"""
    plotter = CrawlerPlotter()
    plotter.create_all_plots()


if __name__ == '__main__':
    main()