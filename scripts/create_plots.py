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
import matplotlib.lines as mlines
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV



class CrawlerPlotter:
    """Visualisiert Ergebnisse der verschiedenen Crawling-Strategien"""

    def __init__(self, config_file='crawler_config.ini'):
        """Initialisiert Plotter mit Konfiguration"""
        self.config = configparser.ConfigParser()
        self.config.read(config_file)

        # Konfiguration laden
        self.create_plots = self.config.getboolean('PLOTTING', 'CREATE_PLOTS', fallback=False)
        self.output_dir = self.config['PLOTTING']['OUTPUT_DIRECTORY']

        # Neue Konfigurationsparameter
        self.time_base_pages = int(self.config.get('PLOTTING', 'time_base_pages', fallback='100'))
        self.venn_top_percent = int(self.config.get('PLOTTING', 'venn_top_percent', fallback='20'))
        self.num_tables = self.config.getint('PLOTTING', 'NUM_TABLES')
        self.table_pcts = [int(x.strip()) for x in self.config.get('PLOTTING', 'TABLE_PCTS').split(',')]
        self.table_ns = [int(x.strip()) for x in self.config.get('PLOTTING', 'TABLE_NS').split(',')]

        if not self.create_plots:
            print("Plotting ist deaktiviert")
            return

        # Output-Verzeichnis erstellen
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        # Farben für Strategien definieren
        self.colors = {
            'keyword': '#FF6B6B',
            'vectorspace': '#8B8C0A',
            'naivebayes': '#45B7D1'
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
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['legend.fontsize'] = 10
        plt.rcParams['figure.dpi'] = 100
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['savefig.bbox'] = 'tight'
        plt.rcParams['axes.labelpad'] = 14

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
        """Erstellt Balkendiagramm der Dokumentbewertungszeit"""
        if not self.check_all_strategies_present():
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        strategies = []
        doc_times_ms = []

        for strategy in self.strategy_order:
            if strategy not in self.data:
                continue

            strategy_data = self.data[strategy]

            if 'doc_eval_time_ns' in strategy_data and 'parent_calc_count' in strategy_data:
                parent_count = strategy_data['parent_calc_count']
                if parent_count > 0:
                    doc_time_ns = strategy_data['doc_eval_time_ns']['mean']
                    doc_time_ms = doc_time_ns / 1_000_000
                    doc_time_scaled = doc_time_ms * self.time_base_pages
                else:
                    doc_time_scaled = 0
            else:
                doc_time_scaled = 0

            strategies.append(self.strategy_names[strategy])
            doc_times_ms.append(doc_time_scaled)

        if strategies:
            x = np.arange(len(strategies))

            bars = []
            for i, strategy in enumerate(self.strategy_order):
                if strategy not in self.data:
                    continue

                bar = ax.bar(x[i], doc_times_ms[i], width=0.6,
                             color=self.colors[strategy], alpha=0.8,
                             edgecolor='black', linewidth=1.5)
                bars.append(bar)

                if doc_times_ms[i] > 0:
                    ax.text(x[i], doc_times_ms[i],
                            f'{doc_times_ms[i]:.2f}',
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

            ax.set_xlabel('Crawling-Strategie', fontsize=11)
            ax.set_ylabel(f'Bewertungszeit [ms pro {self.time_base_pages} Seiten]', fontsize=11)
            ax.set_title(f'Durchschnittliche Dokumentbewertungsdauer',
                         fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(strategies)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

            plt.figtext(
                0.5, -0.05,
                'Hinweis: Werte beziehen sich ausschließlich auf die Dokumentbewertung.',
                ha='center', fontsize=10, style='italic'
            )

        filename = f"{self.output_dir}/scoring_performance_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Bewertungszeit Grafik gespeichert: {filename}")

    def plot_memory_usage(self):
        """Erstellt Balkendiagramm des zusätzlichen Speicherbedarfs"""
        if not self.check_all_strategies_present():
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        strategies = []
        doc_memories = []
        baselines = []

        for strategy in self.strategy_order:
            if strategy not in self.data:
                continue

            strategy_data = self.data[strategy]

            doc_mem_p95 = 0
            baseline_p95 = 0

            if 'doc_memory_delta_mib' in strategy_data:
                doc_mem_p95 = strategy_data['doc_memory_delta_mib'].get('p95', 0)

            if 'doc_memory_baseline_mib' in strategy_data:
                baseline_p95 = strategy_data['doc_memory_baseline_mib'].get('p95', 0)

            strategies.append(self.strategy_names[strategy])
            doc_memories.append(doc_mem_p95)
            baselines.append(baseline_p95)

        max_memory = max(doc_memories) if doc_memories else 0
        use_kib = max_memory < 1.0

        if use_kib:
            doc_memories = [m * 1024 for m in doc_memories]
            baselines = [b * 1024 for b in baselines]
            unit = 'KiB'
        else:
            unit = 'MiB'

        if strategies:
            x = np.arange(len(strategies))

            for i, strategy in enumerate(self.strategy_order):
                if strategy not in self.data:
                    continue

                bar = ax.bar(x[i], doc_memories[i], width=0.6,
                             color=self.colors[strategy], alpha=0.8,
                             edgecolor='black', linewidth=1.5)

                if doc_memories[i] > 0:
                    ax.text(x[i], doc_memories[i],
                            f'{doc_memories[i]:.2f}',
                            ha='center', va='bottom', fontsize=11, fontweight='bold')

            ax.set_xlabel('Crawling-Strategie', fontsize=11)
            ax.set_ylabel(f'Zusätzlicher Speicherbedarf (p95) [{unit}]', fontsize=11)
            ax.set_title('Zusätzlicher Speicherbedarf durch die Dokumentbewertung',
                         fontsize=14, fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels(strategies)
            ax.grid(True, alpha=0.3, linestyle='--', axis='y')

            baseline_text = f'Basis-Speicherbedarf (p95) in {unit}: '
            for i, (strat_name, baseline) in enumerate(zip(strategies, baselines)):
                if i > 0:
                    baseline_text += ', '
                baseline_text += f'{strat_name} = {baseline:.1f}'

            combined_text = baseline_text + '\nHinweis: Werte beziehen sich ausschließlich auf die Dokumentbewertung.'
            plt.figtext(0.5, -0.05, combined_text, ha='center', fontsize=10, style='italic')

        filename = f"{self.output_dir}/memory_usage_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Speicherbedarf Grafik gespeichert: {filename}")

    def create_quantile_tables(self):
        """Erstellt mehrere Tabellen ab konfigurierten Perzentilen (aufsteigende Definition)"""

        # Spaltenüberschriften
        headers = ['Kategorie']
        for strategy in self.strategy_order:
            if strategy in self.data:
                headers.append(self.strategy_names[strategy])

        # Titelbereinigung
        def clean_title(title):
            """Entfernt Wikipedia-Suffix und kürzt lange Titel"""
            if not title:
                return 'Kein Titel'
            title = title.replace(' – Wikipedia', '').replace(' - Wikipedia', '')
            title = ' '.join(title.split())
            return (title[:47] + '.') if len(title) > 50 else title

        # pro Strategie nach Score (absteigend) sortieren
        sorted_pages_by_strategy = {}
        for strategy, sdata in self.data.items():
            if 'pages' in sdata:
                sorted_pages_by_strategy[strategy] = sorted(
                    sdata['pages'], key=lambda x: x['score'], reverse=True
                )

        # Tabellen erzeugen
        for t in range(self.num_tables):
            pct = self.table_pcts[t]
            count = self.table_ns[t]

            # Zeilen aufbauen
            table_data = []
            for i in range(count):
                # wissenschaftliches Label (Quantil/Perzentil)
                row = [f'Rang {i + 1}']
                for strategy in self.strategy_order:
                    if strategy not in sorted_pages_by_strategy:
                        continue
                    pages = sorted_pages_by_strategy[strategy]
                    n = len(pages)
                    # Mapping: Perzentil (aufsteigend) -> absteigender Rang
                    # Q100 => Index 0 (Maximum), Q0 => Index n-1 (Minimum)
                    start_idx = (n - 1) - int((pct / 100.0) * (n - 1))
                    idx = start_idx + i
                    if 0 <= idx < n:
                        title = clean_title(pages[idx].get('title', ''))
                        score = pages[idx].get('score', 0.0)
                        row.append(f'{title}\n(Relevanzwert: {score:.3f})')
                    else:
                        row.append('-')
                table_data.append(row)

            # Figure/Axes
            num_strategies = len(headers) - 1
            fig_width = 10 + num_strategies * 2
            fig_height = 6 + max(1, count) * 0.4
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))
            ax.axis('tight')
            ax.axis('off')

            # Tabelle (ohne Hintergrundfarben)
            col_widths = [0.15] + [(0.85 / num_strategies)] * num_strategies
            table = ax.table(cellText=table_data,
                             colLabels=headers,
                             cellLoc='left',
                             loc='center',
                             colWidths=col_widths)

            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1, 2.5)

            # Header fett, sonst Standard
            for (r, c), cell in table.get_celld().items():
                if r == 0:
                    cell.set_text_props(weight='bold', size=11)
                    cell.set_linewidth(1.0)
                else:
                    cell.set_linewidth(0.5)
                    cell.set_text_props(linespacing=1.5)
                    cell.PAD = 0.05

            # Titel
            ax.text(0.5, 0.75, f'Relevanzrangliste, Start bei {pct}. Perzentil (Q={pct/100:.2f})',
                    transform=ax.transAxes, ha='center', va='top',
                    fontsize=12, fontweight='bold')

            # Datei pro Tabelle
            filename = f"{self.output_dir}/comparison_table_p{pct}_{self.timestamp}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.05)
            plt.close()
            print(f"Tabelle gespeichert: {filename}")

    def plot_overlap_venn(self):
        """Erstellt Venn-Diagramm für Überlappung der Top-K Seiten"""
        # URLs für jede Strategie sammeln (Top-K basierend auf Score)
        urls_by_strategy = {}

        for strategy, strategy_data in self.data.items():
            if 'pages' not in strategy_data:
                continue

            # Sortiere alle bewerteten Seiten nach Score
            all_pages = sorted(strategy_data['pages'],
                               key=lambda x: x['score'],
                               reverse=True)

            # Berechne wie viele Seiten die Top-K% ausmachen
            num_pages = len(all_pages)
            top_k_count = max(1, int(num_pages * self.venn_top_percent / 100))

            # Nimm nur die Top-K Seiten
            top_pages = all_pages[:top_k_count]
            urls_by_strategy[strategy] = set([page['url'] for page in top_pages])

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

            # Berechne Überlappungsanteile für alle Kombinationen
            if len(sets) == 3:
                only_kw = len(sets[0] - sets[1] - sets[2])
                only_vs = len(sets[1] - sets[0] - sets[2])
                only_nb = len(sets[2] - sets[0] - sets[1])
                kw_vs = len(sets[0] & sets[1] - sets[2])
                kw_nb = len(sets[0] & sets[2] - sets[1])
                vs_nb = len(sets[1] & sets[2] - sets[0])
                all_three = len(sets[0] & sets[1] & sets[2])

                # Füge Prozentangaben zu den Labels hinzu
                for label_id, count in [('100', only_kw), ('010', only_vs), ('001', only_nb),
                                        ('110', kw_vs), ('101', kw_nb), ('011', vs_nb),
                                        ('111', all_three)]:
                    label = v.get_label_by_id(label_id)
                    if label and count > 0:
                        total = len(sets[0] | sets[1] | sets[2])
                        percent = (count / total) * 100 if total > 0 else 0
                        label.set_text(f'{count}\n({percent:.1f}%)')

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
        ax.set_title(f'Überlappung der {self.venn_top_percent}% höchstbewerteten Seiten\n' +
                     f'Gesamt: {len(all_urls)} eindeutige URLs',
                     fontsize=14, fontweight='bold')

        # Speichern
        filename = f"{self.output_dir}/overlap_venn_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Overlap Venn-Diagramm gespeichert: {filename}")

    def plot_relevance_trend(self):
        """Erstellt Liniendiagramm der normalisierten Relevanztrends über die Zeit"""
        if not self.check_all_strategies_present():
            return

        # Konfiguration laden
        window_size = int(self.config.get('PLOTTING', 'MOVING_AVG_WINDOW', fallback='50'))
        max_pages = int(self.config.get('PLOTTING', 'MAX_PAGES', fallback='1000'))

        fig, ax = plt.subplots(figsize=(12, 7))

        # Listen für Plot-Elemente und Labels
        lines = []
        labels = []
        trend_line = None
        trend_label = None

        # Daten für jede Strategie verarbeiten
        for strategy in self.strategy_order:
            if strategy not in self.data:
                continue

            strategy_data = self.data[strategy]
            if 'pages' not in strategy_data:
                continue

            pages = strategy_data['pages']
            if not pages:
                continue

            # Nach visit_idx sortieren für zeitliche Reihenfolge
            sorted_pages = sorted(pages, key=lambda x: x.get('visit_idx', 0))

            # Alle Scores für Perzentilberechnung extrahieren
            all_scores = [p['score'] for p in sorted_pages]

            # Perzentilränge mit korrekter Midrank-Behandlung berechnen
            def calculate_percentile_rank(score, all_values):
                """Berechnet Perzentilrang mit fairem Midrank für Ties"""
                less_than = sum(1 for v in all_values if v < score)
                equal_to = sum(1 for v in all_values if v == score)
                # Fairer Midrank ohne Bias
                percentile_rank = (less_than + 0.5 * equal_to) / len(all_values)
                return percentile_rank

            # Perzentilränge für alle Seiten berechnen
            percentile_ranks = [calculate_percentile_rank(score, all_scores)
                                for score in all_scores]

            # Erst nach Perzentilberechnung auf max_pages kürzen
            percentile_ranks = percentile_ranks[:max_pages]
            sorted_pages = sorted_pages[:max_pages]

            # Trailing Moving Average berechnen
            moving_avg = []
            for i in range(len(percentile_ranks)):
                # Nur vergangene und aktuelle Werte verwenden (trailing window)
                start = max(0, i - window_size + 1)
                end = i + 1

                window_values = percentile_ranks[start:end]
                moving_avg.append(np.mean(window_values))

            # X-Achse: Anzahl besuchter Seiten
            x_values = list(range(1, len(percentile_ranks) + 1))

            # Hauptlinie plotten (Moving Average)
            line, = ax.plot(x_values, moving_avg,
                            color=self.colors[strategy],
                            linewidth=2.5,
                            alpha=0.9)
            lines.append(line)
            labels.append(f'{self.strategy_names[strategy]}')

            # Trendlinie auf ungeglätteten Daten berechnen
            if len(x_values) > 1:
                # Lineare Regression auf Original-Perzentilrängen
                z = np.polyfit(x_values, percentile_ranks, 1)
                p = np.poly1d(z)

                # Trendlinie plotten (gestrichelt, dünner, durchsichtig)
                tl, = ax.plot(x_values, p(x_values),
                              linestyle='--',
                              color=self.colors[strategy],
                              linewidth=1.2,
                              alpha=0.5)

                # Nur einmal einen grauen Proxy für die Legende erzeugen
                if trend_line is None:
                    trend_line = mlines.Line2D([], [], linestyle='--', color='gray',
                                               linewidth=1.2, alpha=0.5)
                    trend_label = 'Lineare Trendlinie'

        # Achsenbeschriftung und Titel
        ax.set_xlabel('Anzahl besuchter Seiten', fontsize=12)
        ax.set_ylabel('Normalisierter Relevanzwert (Perzentil-Rang)', fontsize=12)
        ax.set_title('Vergleich der Relevanztrends über den Crawling-Verlauf',
                     fontsize=14, fontweight='bold')

        # Y-Achse Bereich festlegen
        ax.set_ylim([0, 1])

        # X-Achse begrenzen
        if max_pages:
            ax.set_xlim([0, max_pages])

        # Gitter
        ax.grid(True, alpha=0.3, linestyle='--')

        # Legende mit Trendlinie am Ende
        if trend_line is not None:
            lines.append(trend_line)
            labels.append(trend_label)

        ax.legend(lines, labels, loc='best', framealpha=0.9)

        # Fußnote mit technischen Details
        plt.figtext(0.5, -0.05,
                    f'Glättung: Trailing Moving Average (Fenster = {window_size}). '
                    f'Gestrichelt: lineare Trends auf den Originaldaten.',
                    ha='center', fontsize=10, style='italic')

        # Speichern
        filename = f"{self.output_dir}/relevance_trend_{self.timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Relevanztrend Grafik gespeichert: {filename}")

    def create_all_plots(self):
        """Erstellt alle Grafiken in der richtigen Reihenfolge"""
        if not self.create_plots:
            print("Plotting ist deaktiviert")
            return

        print("\nErstelle Visualisierungen...")

        # Erstelle alle Plots inklusive der neuen kalibrierten Relevanz
        self.plot_scoring_performance()
        self.plot_memory_usage()
        self.create_quantile_tables()
        self.plot_overlap_venn()
        self.plot_relevance_trend()
        print("Alle Visualisierungen wurden erstellt\n")


if __name__ == "__main__":
    plotter = CrawlerPlotter("crawler_config.ini")
    plotter.create_all_plots()