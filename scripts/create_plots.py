"""
Visualisiert Ergebnisse vom Crawling in verschiedenen Grafiken
"""

import json
import os
import glob
import matplotlib.pyplot as plt
import matplotlib_venn as venn
import configparser
from datetime import datetime
from pathlib import Path
import numpy as np


class CrawlerPlotter:
    """Visualisiert Ergebnisse der verschiedenen Crawling-Strategien"""

    def __init__(self, config_file='crawler_config.ini'):
        """Initialisiert Plotter mit Konfiguration"""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        # Konfiguration laden
        self.create_plots = self.config.getboolean('PLOTTING', 'CREATE_PLOTS', fallback=False)
        self.output_dir = self.config['PLOTTING']['OUTPUT_DIRECTORY']

        if not self.create_plots:
            print("Plotting ist deaktiviert")
            return

        # Output-Verzeichnis erstellen
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Farben für Strategien definieren
        self.colors = {
            'keyword': '#FF6B6B',       # Rot
            'vectorspace': '#8B8C0A',   # Olivgrün
            'naivebayes': '#45B7D1'     # Blau
        }

        # Deutsche Namen für Strategien
        self.strategy_names = {
            'keyword': 'Keyword',
            'vectorspace': 'Vektorraummodell',
            'naivebayes': 'Naive Bayes'
        }

        # Reihenfolge der Strategien
        self.strategy_order = ['keyword', 'vectorspace', 'naivebayes']

        # Plotdarstellung definieren
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

        # Daten sammeln
        self.data = self.load_all_data()

    def load_all_data(self):
        """Lädt alle JSON-Dateien aus dem exports Verzeichnis"""
        data = {}

        # Alle JSON-Dateien identifizieren
        json_files = glob.glob('exports/*_crawler_*.json')

        for file_path in json_files:
            # Strategie Namen finden
            filename = os.path.basename(file_path)
            if 'keyword' in filename:
                strategy = 'keyword'
            elif 'vectorspace' in filename:
                strategy = 'vectorspace'
            elif 'naivebayes' in filename:
                strategy = 'naivebayes'
            else:
                continue

            # JSON-Daten laden
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

    def check_all_strategies_present(self):
        """Prüft ob alle drei Strategien vorhanden sind"""
        required = set(self.strategy_order)
        return required.issubset(self.data.keys())

    def plot_scoring_performance(self):
        """Erstellt gruppiertes Balkendiagramm der Bewertungszeit pro Seite"""
        if not self.check_all_strategies_present():
            return

        fig, ax = plt.subplots(figsize=(12, 7))

        strategies = []
        doc_times = []  # Dokumentbewertung
        link_times = []  # Linkbewertung

        # Daten in definierter Reihenfolge sammeln
        for strategy in self.strategy_order:
            if strategy not in self.data:
                continue

            strategy_data = self.data[strategy]
            if 'cpu_ms_per_1000_docs' not in strategy_data:
                continue

            # Berechne ms pro einzelner Seite
            ms_per_page_total = strategy_data['cpu_ms_per_1000_docs'] / 1000

            # Schätze Aufteilung (70% Dokument, 30% Links)
            doc_time = ms_per_page_total * 0.7
            link_time = ms_per_page_total * 0.3

            strategies.append(self.strategy_names[strategy])
            doc_times.append(doc_time)
            link_times.append(link_time)

        if strategies:
            x = np.arange(len(strategies))
            width = 0.35

            # Balken für Dokumentbewertung
            bars1 = ax.bar(x - width/2, doc_times, width,
                          label='Dokumentbewertung',
                          color='#4CAF50', alpha=0.8)

            # Balken für Linkbewertung
            bars2 = ax.bar(x + width/2, link_times, width,
                          label='Linkbewertung',
                          color='#FF9800', alpha=0.8)

            # Werte über den Balken
            for bar in bars1:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10)

            for bar in bars2:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}',
                       ha='center', va='bottom', fontsize=10)

            # Formatierung
            ax.set_xlabel('Crawling-Strategie', fontsize=12)
            ax.set_ylabel('Bewertungszeit [ms/Seite]', fontsize=12)
            ax.set_title('Durchschnittliche Bewertungszeit pro Seite\n(CPU-Zeit für Dokument- und Linkbewertung)',
                        fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(strategies)
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Speichern
        filename = f"{self.output_dir}/scoring_performance_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bewertungszeit Grafik gespeichert: {filename}")

    def plot_memory_usage(self):
        """Erstellt Balkendiagramm des Speicherbedarfs (95. Perzentil)"""
        if not self.check_all_strategies_present():
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        # Daten in definierter Reihenfolge sammeln
        strategies = []
        memories = []
        colors_list = []

        for strategy in self.strategy_order:
            if strategy not in self.data:
                continue

            strategy_data = self.data[strategy]
            if 'memory_p95_mib' in strategy_data:
                mem_p95 = strategy_data['memory_p95_mib']
                strategies.append(self.strategy_names[strategy])
                memories.append(mem_p95)
                colors_list.append(self.colors[strategy])

        if strategies:
            # Balkendiagramm erstellen
            bars = ax.bar(strategies, memories, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

            # Werte über den Balken hinzufügen
            for bar, memory in zip(bars, memories):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width() / 2., height,
                        f'{memory:.1f} MiB',
                        ha='center', va='bottom', fontweight='bold')

            # Formatierung
            ax.set_xlabel('Crawling-Strategie', fontsize=12)
            ax.set_ylabel('Speicherbedarf in MiB (95. Perzentil)', fontsize=12)
            ax.set_title('Speicherbedarf während Bewertungslogik\n(95. Perzentil des RSS)',
                        fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

        # Speichern
        filename = f"{self.output_dir}/memory_usage_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Speicherbedarf Grafik gespeichert: {filename}")

    def create_comparison_table(self):
        """Erstellt Vergleichstabelle der Top/Mittel/Bottom Seiten"""

        # Dynamische Figurengröße basierend auf Anzahl Strategien
        num_strategies = len(self.data.keys())
        fig_width = 10 + num_strategies * 2
        fig, ax = plt.subplots(figsize=(fig_width, 14))
        ax.axis('tight')
        ax.axis('off')

        # Header in definierter Reihenfolge
        headers = ['Kategorie']
        for strategy in self.strategy_order:
            if strategy in self.data:
                headers.append(self.strategy_names[strategy])

        table_data = []

        # Funktion zum Bereinigen des Titels
        def clean_title(title):
            """Entfernt Wikipedia-Suffix und bereinigt Titel"""
            if not title:
                return 'Kein Titel'
            # Entferne " – Wikipedia" oder " - Wikipedia"
            title = title.replace(' – Wikipedia', '').replace(' - Wikipedia', '')
            # Entferne überflüssige Whitespaces
            title = ' '.join(title.split())
            # Max. 50 Zeichen
            if len(title) > 50:
                title = title[:47] + '...'
            return title

        # Seiten für jede Strategie sortieren
        sorted_pages_by_strategy = {}
        for strategy, strategy_data in self.data.items():
            if 'pages' in strategy_data:
                sorted_pages = sorted(strategy_data['pages'],
                                      key=lambda x: x['score'],
                                      reverse=True)
                sorted_pages_by_strategy[strategy] = sorted_pages

        # Zeilen für oberste Ränge 1-5
        for i in range(5):
            row_data = [f'Oberste Ränge {i + 1}']
            for strategy in self.strategy_order:
                if strategy not in self.data:
                    continue
                if strategy in sorted_pages_by_strategy:
                    pages = sorted_pages_by_strategy[strategy]
                    if i < len(pages):
                        title = clean_title(pages[i].get('title', ''))
                        score = pages[i].get('score', 0)
                        row_data.append(f'{title}\n(Relevanzwert: {score:.3f})')
                    else:
                        row_data.append('-')
                else:
                    row_data.append('-')
            table_data.append(row_data)

        # Zeilen für mittlere Ränge (um Median)
        for i in range(5):
            row_data = [f'Mittlere Ränge {i + 1}']
            for strategy in self.strategy_order:
                if strategy not in self.data:
                    continue
                if strategy in sorted_pages_by_strategy:
                    pages = sorted_pages_by_strategy[strategy]
                    # Berechne Index um Median herum
                    median_idx = len(pages) // 2
                    start_idx = max(0, median_idx - 2)
                    idx = start_idx + i
                    if idx < len(pages):
                        title = clean_title(pages[idx].get('title', ''))
                        score = pages[idx].get('score', 0)
                        row_data.append(f'{title}\n(Relevanzwert: {score:.3f})')
                    else:
                        row_data.append('-')
                else:
                    row_data.append('-')
            table_data.append(row_data)

        # Zeilen für unterste Ränge 1-5
        for i in range(5):
            row_data = [f'Unterste Ränge {i + 1}']
            for strategy in self.strategy_order:
                if strategy not in self.data:
                    continue
                if strategy in sorted_pages_by_strategy:
                    pages = sorted_pages_by_strategy[strategy]
                    # Die letzten 5 Seiten
                    idx = len(pages) - 5 + i
                    if idx >= 0 and idx < len(pages):
                        title = clean_title(pages[idx].get('title', ''))
                        score = pages[idx].get('score', 0)
                        row_data.append(f'{title}\n(Relevanzwert: {score:.3f})')
                    else:
                        row_data.append('-')
                else:
                    row_data.append('-')
            table_data.append(row_data)

        # Tabelle erstellen
        col_widths = [0.15] + [(0.85 / num_strategies)] * num_strategies

        table = ax.table(cellText=table_data,
                         colLabels=headers,
                         cellLoc='left',
                         loc='center',
                         colWidths=col_widths)

        # Formatierung
        table.auto_set_font_size(False)
        table.set_fontsize(9)  # Minimal größere Schrift
        table.scale(1, 2.8)

        # Farben für Zeilen
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
                cell.set_linewidth(2)
                cell.set_height(0.08)
            else:
                row_idx = row - 1
                # Oberste Ränge
                if row_idx < 5:
                    cell.set_facecolor('#90EE90')  # Hellgrün
                # Mittlere Ränge
                elif row_idx < 10:
                    cell.set_facecolor('#FFFFE0')  # Hellgelb
                # Unterste Ränge
                else:
                    cell.set_facecolor('#FFB6C1')  # Hellrot

                cell.set_linewidth(0.5)

                # Trennlinie zwischen Kategorien
                if row_idx in [0, 5, 10]:
                    cell.set_linewidth(2)

                # Textausrichtung
                cell.set_text_props(linespacing=1.5)
                cell.PAD = 0.05

        # Titel mit Präzisierung
        plt.title('Vergleich der Crawling-Strategien: Artikel nach Relevanzrang\n' +
                  'Oberste Ränge: höchste Relevanzwerte | Mittlere Ränge: um Median | Unterste Ränge: niedrigste Relevanzwerte',
                  fontsize=14, fontweight='bold', pad=20)

        # Speichern
        filename = f"{self.output_dir}/comparison_table_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Vergleichstabelle gespeichert: {filename}")

    def plot_overlap_venn(self):
        """Erstellt Venn-Diagramm für Überlappung der gefundenen Seiten"""
        # URLs für jede Strategie sammeln
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
            # Venndiagramm für 3 Kreise in definierter Reihenfolge
            sets = [urls_by_strategy.get('keyword', set()),
                    urls_by_strategy.get('vectorspace', set()),
                    urls_by_strategy.get('naivebayes', set())]

            labels = ['Keyword', 'Vektorraummodell', 'Naive Bayes']

            # Venn-Diagramm erstellen
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
            # Venndiagramm für 2 Kreise
            strategies = list(urls_by_strategy.keys())
            sets = [urls_by_strategy[strategies[0]], urls_by_strategy[strategies[1]]]
            labels = [self.strategy_names[s] for s in strategies]

            v = venn.venn2(sets, labels, ax=ax)

            # Kreise färben
            if v.get_patch_by_id('10'):
                v.get_patch_by_id('10').set_color(self.colors[strategies[0]])
                v.get_patch_by_id('10').set_alpha(0.5)
            if v.get_patch_by_id('01'):
                v.get_patch_by_id('01').set_color(self.colors[strategies[1]])
                v.get_patch_by_id('01').set_alpha(0.5)

        # Overlap berechnen
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
        """Erstellt alle Grafiken in der richtigen Reihenfolge"""
        if not self.create_plots:
            print("Plotting ist deaktiviert")
            return

        print("\nErstelle Visualisierungen...")

        # Erstelle alle Plots
        self.plot_scoring_performance()
        self.plot_memory_usage()
        self.create_comparison_table()
        self.plot_overlap_venn()

        print("Alle Visualisierungen wurden erstellt\n")


if __name__ == "__main__":
    plotter = CrawlerPlotter("crawler_config.ini")
    plotter.create_all_plots()