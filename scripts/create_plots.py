
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

    def plot_harvest_rate(self):
        """Erstellt Liniendiagramm der Harvest-Rate über relevante Seiten"""
        fig, ax = plt.subplots(figsize=(10, 6))

        for strategy, strategy_data in self.data.items():
            if 'evaluation_log' not in strategy_data:
                continue

            eval_log = strategy_data['evaluation_log']
            if not eval_log:
                continue

            # Daten extrahieren
            relevant_pages = [entry['relevant_pages_found'] for entry in eval_log]
            harvest_rates = [entry['harvest_rate'] for entry in eval_log]

            # Linie plotten
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

            # Daten extrahieren
            relevant_pages = [entry['relevant_pages_found'] for entry in eval_log]
            avg_relevance = [entry['average_relevance'] for entry in eval_log]

            # Linie plotten
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
        ax.axhspan(0.5, 0.75, alpha=0.1, color='yellow', label='Sehr relevant')
        ax.axhspan(0.75, 1.0, alpha=0.1, color='green', label='Vollkommen Relevant')

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

            # Daten extrahieren
            batch_numbers = [entry.get('batch_number', i) for i, entry in enumerate(eval_log)]
            avg_positions = [entry.get('avg_frontier_position', 0) for entry in eval_log]

            # Nullwerte filtern
            valid_data = [(b, p) for b, p in zip(batch_numbers, avg_positions) if p > 0]
            if valid_data:
                batch_numbers, avg_positions = zip(*valid_data)

                # Batches aggregieren
                aggregation_window = 5
                aggregated_batches = []
                aggregated_positions = []

                for i in range(0, len(batch_numbers), aggregation_window):
                    window_batches = batch_numbers[i:i + aggregation_window]
                    window_positions = avg_positions[i:i + aggregation_window]

                    if window_batches:
                        # Mittlerer Batch-Wert für X-Achse
                        avg_batch = sum(window_batches) / len(window_batches)
                        # Durchschnittliche Position für Y-Achse
                        avg_pos = sum(window_positions) / len(window_positions)

                        aggregated_batches.append(avg_batch)
                        aggregated_positions.append(avg_pos)

                # Linie mit Interpolation plotten
                if len(aggregated_batches) > 1:
                    # Cubic interpolation
                    from scipy.interpolate import interp1d
                    import numpy as np

                    # Interpolierte Kurve erstellen
                    try:
                        # Cubic interpolation wenn genug Punkte
                        if len(aggregated_batches) > 3:
                            f = interp1d(aggregated_batches, aggregated_positions,
                                         kind='cubic', fill_value='extrapolate')
                        else:
                            # Linear bei wenigen Punkten
                            f = interp1d(aggregated_batches, aggregated_positions,
                                         kind='linear', fill_value='extrapolate')

                        # Feinere X-Achse für glatte Kurve erstellen
                        x_smooth = np.linspace(min(aggregated_batches),
                                               max(aggregated_batches), 100)
                        y_smooth = f(x_smooth)

                        # Glatte Linie plotten
                        ax.plot(x_smooth, y_smooth,
                                color=self.colors[strategy],
                                linewidth=2,
                                label=f'{strategy.capitalize()} Spider',
                                alpha=0.8)

                        # Aggregierte Datenpunkte makieren
                        ax.scatter(aggregated_batches, aggregated_positions,
                                   color=self.colors[strategy],
                                   s=30, alpha=0.6, zorder=5)

                    except:
                        # Fallback zu einfacher Linie bei Interpolationsfehler
                        ax.plot(aggregated_batches, aggregated_positions,
                                color=self.colors[strategy],
                                linewidth=2,
                                marker='D',
                                markersize=5,
                                label=f'{strategy.capitalize()} Spider',
                                alpha=0.8)
                else:
                    # Nur ein Datenpunkt
                    ax.scatter(aggregated_batches, aggregated_positions,
                               color=self.colors[strategy],
                               s=50,
                               label=f'{strategy.capitalize()} Spider',
                               alpha=0.8)

        # Formatierung
        ax.set_xlabel('Batch-Nummer (5er-Aggregation)', fontsize=12)
        ax.set_ylabel('Durchschnittliche Position in Frontier', fontsize=12)
        ax.set_title('Durchschnittliche Platzierung neuer URLs in der Frontier\n(Aggregiert über 5 Batches)',
                     fontsize=14, fontweight='bold')
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

        # Zeiten für feste Anzahl von Seiten sammeln
        target_pages = 100 # Normalisiert auf 100 Seiten

        for strategy, strategy_data in self.data.items():
            if 'relevance_calculation_time' in strategy_data:
                total_time = strategy_data['relevance_calculation_time']
                total_pages = strategy_data.get('total_pages_visited', 1)

                # Auf target_pages normalisieren
                normalized_time = (total_time / total_pages) * target_pages

                strategies.append(strategy.capitalize())
                times.append(normalized_time)
                colors_list.append(self.colors[strategy])

        if strategies:
            # Balkendiagramm erstellen
            bars = ax.bar(strategies, times, color=colors_list, alpha=0.8, edgecolor='black', linewidth=1.5)

            # Werte über den Balken hinzufügen
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

        # Dynamische Figurengröße basierend auf Anzahl Strategien
        num_strategies = len(self.data.keys())
        fig_width = 10 + num_strategies * 2
        fig, ax = plt.subplots(figsize=(fig_width, 12))
        ax.axis('tight')
        ax.axis('off')

        # Daten für jede Strategie sammeln
        table_data = []
        headers = ['Kategorie'] + [s.capitalize() for s in self.data.keys()]

        # Funktion zum Extrahieren von Titeln
        def get_titles(pages, indices):
            titles = []
            for i in indices:
                if i < len(pages):
                    # Sauberen Titel extrahieren
                    title = pages[i].get('title', 'Kein Titel')
                    # Entferne überflüssige Whitespaces
                    title = ' '.join(title.split())
                    # Max. 60 zeichen aufnehmen
                    if len(title) > 60:
                        title = title[:57] + '...'
                    titles.append(title)
                else:
                    titles.append('-')
            return titles

        # Seiten für jede Strategie sortieren
        sorted_pages_by_strategy = {}
        for strategy, strategy_data in self.data.items():
            if 'pages' in strategy_data:
                sorted_pages = sorted(strategy_data['pages'],
                                      key=lambda x: x['score'],
                                      reverse=True)
                sorted_pages_by_strategy[strategy] = sorted_pages

        # Zeilen für Top 5 erstellen
        for i in range(5):
            row_data = [f'Top {i + 1}']
            for strategy in self.data.keys():
                if strategy in sorted_pages_by_strategy:
                    pages = sorted_pages_by_strategy[strategy]
                    if i < len(pages):
                        title = pages[i].get('title', 'Kein Titel')
                        title = ' '.join(title.split())[:60]
                        score = pages[i].get('score', 0)
                        row_data.append(f'{title}\n(Score: {score:.3f})')
                    else:
                        row_data.append('-')
                else:
                    row_data.append('-')
            table_data.append(row_data)

        # Zeilen für mittlere 5 erstellen
        for i in range(5):
            row_data = [f'Mitte {i + 1}']
            for strategy in self.data.keys():
                if strategy in sorted_pages_by_strategy:
                    pages = sorted_pages_by_strategy[strategy]
                    middle_start = max(0, len(pages) // 2 - 2)
                    idx = middle_start + i
                    if idx < len(pages):
                        title = pages[idx].get('title', 'Kein Titel')
                        title = ' '.join(title.split())[:60]
                        score = pages[idx].get('score', 0)
                        row_data.append(f'{title}\n(Score: {score:.3f})')
                    else:
                        row_data.append('-')
                else:
                    row_data.append('-')
            table_data.append(row_data)

        # Zeilen für Bottom 5 erstellen
        for i in range(5):
            row_data = [f'Bottom {i + 1}']
            for strategy in self.data.keys():
                if strategy in sorted_pages_by_strategy:
                    pages = sorted_pages_by_strategy[strategy]
                    start_idx = max(0, len(pages) - 5)
                    idx = start_idx + i
                    if idx < len(pages):
                        title = pages[idx].get('title', 'Kein Titel')
                        title = ' '.join(title.split())[:60]
                        score = pages[idx].get('score', 0)
                        row_data.append(f'{title}\n(Score: {score:.3f})')
                    else:
                        row_data.append('-')
                else:
                    row_data.append('-')
            table_data.append(row_data)

        # Tabelle erstellen
        col_widths = [0.12] + [(0.88 / num_strategies)] * num_strategies

        table = ax.table(cellText=table_data,
                         colLabels=headers,
                         cellLoc='left',
                         loc='center',
                         colWidths=col_widths)

        # Formatierung
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2.5)

        # Farben für Zeilen
        for (row, col), cell in table.get_celld().items():
            if row == 0:  # Header
                cell.set_facecolor('#4CAF50')
                cell.set_text_props(weight='bold', color='white')
                cell.set_linewidth(2)
                cell.set_height(0.08)
            else:
                row_idx = row - 1
                # Top 5
                if row_idx < 5:
                    cell.set_facecolor('#90EE90') # Hellgrün
                # Mitte 5
                elif row_idx < 10:
                    cell.set_facecolor('#FFFFE0') # Hellgelb
                # Bottom 5
                else:
                    cell.set_facecolor('#FFB6C1') # Hellrot

                cell.set_linewidth(0.5)

                # Trennlinie zwischen Kategorien
                if row_idx in [0, 5, 10]:
                    cell.set_linewidth(2)

                # Textausrichtung
                cell.set_text_props(linespacing=1.5)
                cell.PAD = 0.05

        # Titel
        plt.title('Vergleich der Crawling-Strategien: Top/Mittel/Bottom Artikel\n' +
                  '(Sortiert nach Relevanz-Score)',
                  fontsize=16, fontweight='bold', pad=20)

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
            # Venndiagramm für 3 Kreise
            sets = [urls_by_strategy.get('keyword', set()),
                    urls_by_strategy.get('vectorspace', set()),
                    urls_by_strategy.get('naivebayes', set())]

            labels = ['Keyword', 'VectorSpace', 'NaiveBayes']

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
            labels = [s.capitalize() for s in strategies]

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

        print(f"\nGrafiken im Verzeichnis '{self.output_dir}' gespeichert")

def main():
    """Hauptfunktion"""
    plotter = CrawlerPlotter()
    plotter.create_all_plots()


if __name__ == '__main__':
    main()