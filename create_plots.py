
"""
Skript zur Visualisierung der Crawling-Ergebnisse
Erstellt Grafiken für wissenschaftliche Analysen
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib_venn as venn
import matplotlib.patches as mpatches
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
        """Erstellt Liniendiagramm der Harvest-Rate über relevante Seiten"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy, strategy_data in self.data.items():
            if 'evaluation_log' not in strategy_data:
                continue

            eval_log = strategy_data['evaluation_log']
            if not eval_log:
                continue

            # Extrahiere Daten - X-Achse ist jetzt relevant_pages_found
            relevant_pages = [entry['relevant_pages_found'] for entry in eval_log]
            harvest_rates = [entry['harvest_rate'] for entry in eval_log]

            # Plotte Linie
            ax.plot(relevant_pages, harvest_rates,
                    color=self.colors[strategy],
                    linewidth=2,
                    marker='o',
                    markersize=5,
                    label=f'{strategy.capitalize()} Spider',
                    alpha=0.8)

        # Formatierung
        ax.set_xlabel('Anzahl gefundener relevanter Seiten', fontsize=12)
        ax.set_ylabel('Harvest-Rate', fontsize=12)
        ax.set_title('Entwicklung der Harvest-Rate über gefundene relevante Seiten', fontsize=14, fontweight='bold')
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
        """Erstellt Liniendiagramm der durchschnittlichen Relevanz über relevante Seiten"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy, strategy_data in self.data.items():
            if 'evaluation_log' not in strategy_data:
                continue

            eval_log = strategy_data['evaluation_log']
            if not eval_log:
                continue

            # Extrahiere Daten - X-Achse ist jetzt relevant_pages_found
            relevant_pages = [entry['relevant_pages_found'] for entry in eval_log]
            avg_relevance = [entry['average_relevance'] for entry in eval_log]

            # Plotte Linie
            ax.plot(relevant_pages, avg_relevance,
                    color=self.colors[strategy],
                    linewidth=2,
                    marker='^',
                    markersize=5,
                    label=f'{strategy.capitalize()} Spider',
                    alpha=0.8)

        # Formatierung
        ax.set_xlabel('Anzahl gefundener relevanter Seiten', fontsize=12)
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

    def plot_frontier_positions(self):
        """Erstellt Liniendiagramm der durchschnittlichen Frontier-Positionen pro Batch"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy, strategy_data in self.data.items():
            if 'evaluation_log' not in strategy_data:
                continue

            eval_log = strategy_data['evaluation_log']
            if not eval_log:
                continue

            # Extrahiere Daten
            batch_numbers = [entry.get('batch_number', i) for i, entry in enumerate(eval_log)]
            avg_positions = [entry.get('avg_frontier_position', 0) for entry in eval_log]

            # Filtere Nullwerte
            valid_data = [(b, p) for b, p in zip(batch_numbers, avg_positions) if p > 0]
            if valid_data:
                batch_numbers, avg_positions = zip(*valid_data)

                # Plotte Linie
                ax.plot(batch_numbers, avg_positions,
                        color=self.colors[strategy],
                        linewidth=2,
                        marker='D',
                        markersize=5,
                        label=f'{strategy.capitalize()} Spider',
                        alpha=0.8)

        # Formatierung
        ax.set_xlabel('Batch-Nummer', fontsize=12)
        ax.set_ylabel('Durchschnittliche Position in Frontier', fontsize=12)
        ax.set_title('Durchschnittliche Platzierung neuer URLs in der Frontier', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(loc='best', frameon=True, shadow=True)
        ax.set_ylim(bottom=0)

        # Speichern
        filename = f"{self.output_dir}/frontier_positions_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Frontier-Positionen Grafik gespeichert: {filename}")

    def plot_evaluation_time(self):
        """Erstellt Balkendiagramm der Bewertungszeit pro Strategie"""
        fig, ax = plt.subplots(figsize=(10, 6))

        strategies = []
        times = []
        colors_list = []

        # Sammle Zeiten für feste Anzahl von Seiten
        target_pages = 100  # Normalisiere auf 100 Seiten

        for strategy, strategy_data in self.data.items():
            if 'relevance_calculation_time' in strategy_data:
                total_time = strategy_data['relevance_calculation_time']
                total_pages = strategy_data.get('total_pages_visited', 1)

                # Normalisiere auf target_pages
                normalized_time = (total_time / total_pages) * target_pages

                strategies.append(strategy.capitalize())
                times.append(normalized_time)
                colors_list.append(self.colors[strategy])

        if strategies:
            # Erstelle Balkendiagramm
            bars = ax.bar(strategies, times, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

            # Füge Werte über den Balken hinzu
            for bar, time in zip(bars, times):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{time:.2f}s',
                        ha='center', va='bottom', fontweight='bold')

            # Formatierung
            ax.set_xlabel('Crawling-Strategie', fontsize=12)
            ax.set_ylabel(f'Zeit für Bewertung von {target_pages} Seiten (Sekunden)', fontsize=12)
            ax.set_title('Vergleich der Bewertungsgeschwindigkeit', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Speichern
        filename = f"{self.output_dir}/evaluation_time_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bewertungszeit Grafik gespeichert: {filename}")

    def create_comparison_table(self):
        """Erstellt Vergleichstabelle der Top/Mittel/Bottom Seiten"""
        fig, ax = plt.subplots(figsize=(14, 10))
        ax.axis('tight')
        ax.axis('off')

        # Sammle Daten für jede Strategie
        table_data = []
        headers = ['Kategorie'] + [s.capitalize() for s in self.data.keys()]

        # Funktion zum Extrahieren von Titeln
        def get_titles(pages, indices):
            titles = []
            for i in indices:
                if i < len(pages):
                    title = pages[i].get('title', 'Kein Titel')[:50]  # Max 50 Zeichen
                    titles.append(title)
                else:
                    titles.append('-')
            return titles

        # Sortiere Seiten für jede Strategie
        sorted_pages_by_strategy = {}
        for strategy, strategy_data in self.data.items():
            if 'pages' in strategy_data:
                sorted_pages = sorted(strategy_data['pages'],
                                      key=lambda x: x['score'],
                                      reverse=True)
                sorted_pages_by_strategy[strategy] = sorted_pages

        # Top 5
        row_data = ['TOP 5 (Beste)']
        for strategy in self.data.keys():
            if strategy in sorted_pages_by_strategy:
                pages = sorted_pages_by_strategy[strategy]
                titles = get_titles(pages, range(5))
                row_data.append('\n'.join(titles))
            else:
                row_data.append('-')
        table_data.append(row_data)

        # Mittlere 5
        row_data = ['MITTE 5']
        for strategy in self.data.keys():
            if strategy in sorted_pages_by_strategy:
                pages = sorted_pages_by_strategy[strategy]
                middle_start = max(0, len(pages) // 2 - 2)
                titles = get_titles(pages, range(middle_start, middle_start + 5))
                row_data.append('\n'.join(titles))
            else:
                row_data.append('-')
        table_data.append(row_data)

        # Bottom 5
        row_data = ['BOTTOM 5 (Schlechteste)']
        for strategy in self.data.keys():
            if strategy in sorted_pages_by_strategy:
                pages = sorted_pages_by_strategy[strategy]
                start_idx = max(0, len(pages) - 5)
                titles = get_titles(pages, range(start_idx, len(pages)))
                row_data.append('\n'.join(titles))
            else:
                row_data.append('-')
        table_data.append(row_data)

        # Erstelle Tabelle
        table = ax.table(cellText=table_data,
                         colLabels=headers,
                         cellLoc='left',
                         loc='center',
                         colWidths=[0.15] + [0.85 / len(self.data.keys())] * len(self.data.keys()))

        # Formatierung
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 3)

        # Farben für Zeilen
        colors_rows = {
            0: '#90EE90',  # Hellgrün für Top 5
            1: '#FFFFE0',  # Hellgelb für Mitte
            2: '#FFB6C1'  # Hellrot für Bottom 5
        }

        # Setze Farben und Rahmen
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
                cell.set_linewidth(2)
            else:
                row_idx = row - 1
                if row_idx in colors_rows:
                    cell.set_facecolor(colors_rows[row_idx])
                cell.set_linewidth(1)

                # Dicke Linien zwischen Kategorien
                if row in [1, 2, 3]:
                    cell.set_linewidth(2)

        # Titel
        plt.title('Vergleich der Crawling-Strategien: Top/Mittel/Bottom Artikel',
                  fontsize=16, fontweight='bold', pad=20)

        # Speichern
        filename = f"{self.output_dir}/comparison_table_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Vergleichstabelle gespeichert: {filename}")

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
            self.plot_frontier_positions()
        except Exception as e:
            print(f"Fehler bei Frontier Positions Plot: {e}")

        try:
            self.plot_evaluation_time()
        except Exception as e:
            print(f"Fehler bei Evaluation Time Plot: {e}")

        try:
            self.create_comparison_table()
        except Exception as e:
            print(f"Fehler bei Comparison Table: {e}")

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
    main()  # !/usr/bin/env python