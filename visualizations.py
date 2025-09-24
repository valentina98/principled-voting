import os
import json
import matplotlib.pyplot as plt
from IPython.display import display
from collections import defaultdict
import numpy as np

class Visualizations:
    
    @staticmethod
    def plot_learning_curve_supervised(location: str):
        results_file = os.path.join(location, 'results.json')
        if not os.path.exists(results_file):
            raise FileNotFoundError(f'No results.json found at {results_file}')

        with open(results_file, 'r') as f:
            results_data = json.load(f)

        learning_curve = results_data.get('learning_curve', {})
        if not learning_curve:
            raise ValueError("No '' found in results.json")

        architecture = results_data.get('architecture', 'Unknown')
        voting_rule = results_data.get('voting_rule', 'Unknown')
        max_num_alternatives = results_data.get('max_num_alternatives', '?')
        max_num_voters = results_data.get('max_num_voters', '?')
        learning_rate = results_data.get('learning_rate', 0)
        
        steps = sorted(int(s) for s in learning_curve.keys())

        train_loss = [learning_curve[str(s)].get('train_loss_avg', float('nan')) for s in steps]
        dev_loss = [learning_curve[str(s)].get('dev_loss', float('nan')) for s in steps]
        exact_match = [learning_curve[str(s)].get('dev_exact_match_acc', float('nan')) for s in steps]
        hamming = [learning_curve[str(s)].get('dev_hamming_acc', float('nan')) for s in steps]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax2 = ax.twinx()  # secondary axis so losses can be log-scaled independently of accuracies

        # Fill under accuracies
        ax.fill_between(steps, hamming, 0, color='tab:blue', alpha=0.3, label='Hamming Accuracy', zorder=1)
        ax.fill_between(steps, exact_match, 0, color='tab:green', alpha=0.3, label='Exact Match Accuracy', zorder=1)

        # Accuracy lines
        ax.plot(steps, hamming, color='tab:blue', linewidth=1.5, zorder=2)
        ax.plot(steps, exact_match, color='tab:green', linewidth=1.5, zorder=2)

        # Losses
        ax2.plot(steps, train_loss, label='Avg Training Loss', color='orange', linestyle='--', linewidth=2, zorder=3)
        ax2.plot(steps, dev_loss, label='Dev Loss', color='red', linewidth=2, zorder=4)

        # Labels and styling
        ax.set_xlabel('Training Steps', fontsize=18)
        ax.set_ylabel('Accuracies', fontsize=18)
        ax2.set_ylabel('Losses (Log Scale)', color='red', fontsize=18)
        ax.set_title(
            f'Training {architecture} on {voting_rule} (alt.={max_num_alternatives}, voters={max_num_voters}, lr={learning_rate:.0e})',
            fontsize=20, pad=10
        )
        # Combine legends from both axes
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=16)

        ax.grid(True)
        ax.set_xlim(left=min(steps), right=max(steps))
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_yscale('log')  # loss typically varies across orders of magnitude
        ax2.tick_params(axis='y', labelsize=14, colors='red')  # red log scale for clarity
        fig.tight_layout()

        plot_path = os.path.join(location, 'learning_curve.png')
        plt.savefig(plot_path, dpi=200)
        print(f'[visualizations] Learning curve saved to {plot_path}')

        display(plt.gcf())  # force notebook to render
        plt.close()


    @staticmethod
    def plot_learning_curve_unsupervised(location: str):
        results_file = os.path.join(location, 'results.json')
        if not os.path.exists(results_file):
            # raise FileNotFoundError(f'No results.json found at {results_file}')
            print(f'[visualizations] No results.json found at {results_file}')
            return

        with open(results_file, 'r') as f:
            results_data = json.load(f)

        learning_curve = results_data.get('learning_curve', {})
        if not learning_curve:
            # raise ValueError('No 'learning_curve' found in results.json')
            print(f"[visualizations WARNING] No 'learning_curve' found in {results_file}")
            return

        architecture = results_data.get('architecture', 'Unknown')
        max_num_alternatives = results_data.get('max_num_alternatives', '?')
        max_num_voters = results_data.get('max_num_voters', '?')
        learning_rate = results_data.get('learning_rate', 0)

        steps = sorted(int(s) for s in learning_curve.keys())

        dev_loss = [learning_curve[str(s)].get('dev_loss', float('nan')) for s in steps]
        train_loss = [learning_curve[str(s)].get('train_loss_avg', float('nan')) for s in steps]

        example_step = learning_curve[str(steps[0])]
        axioms = list(example_step.get('axiom_satisfaction', {}).keys())

        axiom_curves = {
            ax: [learning_curve[str(s)]['axiom_satisfaction'].get(ax, {}).get('cond_satisfaction', float('nan')) for s in steps]
            for ax in axioms
        }

        admissibility_curves = {
            'all_adm_winner': [
                learning_curve[str(s)].get('admissibility', {}).get('all_adm_winner', float('nan')) for s in steps
            ]
        }

        # Check for dev_loss_terms
        example_dev_loss_terms = example_step.get('dev_loss_terms', {})
        dev_loss_terms_curves = {}
        if isinstance(example_dev_loss_terms, dict) and example_dev_loss_terms:
            dev_loss_terms_curves = {
                ax: [learning_curve[str(s)].get('dev_loss_terms', {}).get(ax, float('nan')) for s in steps]
                for ax in example_dev_loss_terms.keys()
            }

        # Create axes
        # fig, ax = plt.subplots(figsize=(10, 6))
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax2 = ax.twinx()

        # Admissibility
        all_adm_curve = admissibility_curves['all_adm_winner']
        ax.plot(steps, all_adm_curve, color='tab:blue', linewidth=1.5, label='All Admissible Winners', zorder=4)

        # Axioms
        greens = ['#006400', '#32CD32', '#66CDAA', '#228B22', '#98FB98', '#90EE90']
        for idx, (ax_name, curve) in enumerate(axiom_curves.items()):
            color = greens[idx % len(greens)]
            ax.plot(steps, curve, label=f'Axiom: {ax_name}', color=color, linewidth=1.5, zorder=5)

        # Train loss
        ax2.plot(steps, train_loss, label='Avg Training Loss', color='orange', linestyle='--', linewidth=2, zorder=2)
        # Dev loss
        ax2.plot(steps, dev_loss, label='Dev Loss', color='red', linewidth=2, zorder=3)
        # Per-axiom dev losses
        reds = ['#8B0000', '#FF6347', '#B22222', '#FF7F7F', '#DC143C', '#FF4500']
        for idx, (ax_name, curve) in enumerate(dev_loss_terms_curves.items()):
            color = reds[idx % len(reds)]
            ax2.plot(steps, curve, label=f'Dev Loss: {ax_name}', color=color, linestyle=':', linewidth=2, zorder=6)
        
        # Labels and styling
        ax.set_xlabel('Training Steps', fontsize=18)  #, labelpad=20
        ax.set_ylabel('Satisfaction', fontsize=18)
        ax2.set_ylabel('Losses (Log Scale)', color='red', fontsize=18)
        plt.title(
            f'Training {architecture} on Axioms (alt.={max_num_alternatives}, voters={max_num_voters}, lr={learning_rate:.0e})',
            fontsize=20, pad=10
        )

        ax.grid(True)
        ax.set_xlim(left=min(steps), right=max(steps))
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax2.set_yscale('log')  # loss typically varies across orders of magnitude
        ax2.tick_params(axis='y', which='both', labelsize=14, colors='red')  # loss scale in red
        
        # Combine legends from both axes ax1 and ax2
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        # place the legend below the plot, with 3 columns
        ax.legend(lines1 + lines2, labels1 + labels2,
                loc='upper center', bbox_to_anchor=(0.5, -0.15),
                ncol=3, fontsize=14, frameon=True,
                columnspacing=0.8,  # space between columns
                )
        fig.tight_layout(rect=[0, 0.05, 1, 1])  # leave extra room at the bottom

        plot_path = os.path.join(location, 'learning_curve.png')
        plt.savefig(plot_path, dpi=200)
        print(f'[PlotLearning] Learning curve saved to {plot_path}')

        display(plt.gcf())  # force notebook to render
        plt.close()

        
    @staticmethod
    def plot_table_supervised(locations: list):
        '''
        Generates a LaTeX-ready table comparing architectures across
        Borda, Plurality, and Copeland, showing Test Loss, Exact Match,
        and Hamming accuracy per voting rule.
        '''
        results = defaultdict(dict)

        for loc in locations:
            results_file = os.path.join(loc, 'results.json')
            if not os.path.exists(results_file):
                print(f'[visualizations WARNING] Skipping {loc}: No results.json found.')
                continue

            with open(results_file, 'r') as f:
                data = json.load(f)

            arch = data.get('architecture', 'Unknown')
            rule = data.get('voting_rule', 'Unknown')
            test_results = data.get('test_results', {})

            metrics = {
                'test_loss': test_results.get('test_loss', float('nan')),
                'test_exact_match_acc': test_results.get('test_exact_match_acc', float('nan')),
                'test_hamming_acc': test_results.get('test_hamming_acc', float('nan')),
            }
            results[rule][arch] = metrics

        # Collect all architectures for consistent columns
        all_architectures = sorted({arch for rules in results.values() for arch in rules.keys()})

        # Build LaTeX table
        table = '\\begin{tabular}{l' + 'c' * len(all_architectures) + '}\n'
        table += '\\toprule\n'
        header = ['Metric'] + all_architectures
        table += ' & '.join(header) + ' \\\\\n'
        table += '\\midrule\n'

        for rule in ['Borda', 'Plurality', 'Copeland']:
            if rule not in results:
                continue
            table += f'\\multicolumn{{{len(all_architectures)+1}}}{{l}}{{\\textbf{{{rule}}}}} \\\\\n'
            for metric_key, metric_label in [
                ('test_loss', 'Test Loss'),
                ('test_exact_match_acc', 'Exact Match Acc.'),
                ('test_hamming_acc', 'Hamming Acc.')
            ]:
                row = [metric_label]
                for arch in all_architectures:
                    metric_val = results[rule].get(arch, {}).get(metric_key, float('nan'))
                    if 'Acc' in metric_label:
                        cell = f'{metric_val*100:.1f}\\%' if not (metric_val != metric_val) else '-'
                    else:
                        cell = f'{metric_val:.4f}' if not (metric_val != metric_val) else '-'
                    row.append(cell)
                table += ' & '.join(row) + ' \\\\\n'
            table += '\\addlinespace\n'

        table += '\\bottomrule\n'
        table += '\\end{tabular}'

        print('\n LaTeX Results Table: \n')
        print(table)

    @staticmethod
    def plot_table_unsupervised(locations_dict: dict[str, str]):

        # Collect data
        records = []
        all_rules = set()
        all_axioms = set()
        setups = []

        for setup_label, loc in locations_dict.items():
            with open(os.path.join(loc, 'results.json'), 'r') as f:
                results = json.load(f)

            voters = results.get('max_num_voters', 'N/A')
            alts = results.get('max_num_alternatives', 'N/A')
            setups.append(setup_label)

            axioms = results.get('axiom_satisfaction_model', {})
            similarities = results.get('rule_similarity', {})
            all_axioms.update(axioms.keys())
            all_rules.update(similarities.keys())

            timing = results.get('timing_log', {})
            training_time = timing.get('training_total', 0.0) / 60.0
            evaluation_time = (
                timing.get('test_loss_evaluation', 0.0) +
                timing.get('admissibility_evaluation_test', 0.0) +
                timing.get('axiom_satisfaction_evaluation_test', 0.0) +
                timing.get('rule_similarity_evaluation_test', 0.0)
            ) / 60.0

            records.append({
                'voters': voters,
                'alts': alts,
                'adm': results.get('admissibility', {}).get('all_adm_winner', float('nan')),
                'axioms': {k: v['cond_satisfaction'] * 100 for k, v in axioms.items()},
                'training_time': training_time,
                'evaluation_time': evaluation_time,
                'similarities': {k: v['identity_accu'] * 100 for k, v in similarities.items()},
            })

        all_axioms = sorted(all_axioms)
        all_rules = sorted(all_rules)

        # Table header
        print('\\begin{tabular}{' + 'l' + 'c' * len(setups) + '}')
        print('\\toprule')
        header = ['\\textbf{Metric}'] + [f'\\textbf{{{col}}}' for col in setups]
        print(' & '.join(header) + ' \\\\')
        print('\\midrule')

        # Basic rows
        rows = [
            ('Max \# Voters', [str(r['voters']) for r in records]),
            ('Max \# Alternatives', [str(r['alts']) for r in records]),
            ('All admissible winner', [f"{r['adm']:.2f}" for r in records]),
            ('Training Time (min)', [f"{r['training_time']:.2f}" for r in records]),
            ('Evaluation Time (min)', [f"{r['evaluation_time']:.2f}" for r in records]),
        ]

        # Axiom rows
        for ax in all_axioms:
            rows.append((
                f'Axiom: {ax}',
                [f"{r['axioms'].get(ax, float('nan')):.2f}" for r in records]
            ))

        # Rule similarity rows
        rows.append((
            '\\textbf{Rule Similarity}',  # placeholder header row
            [''] * len(records)  # empty cells for alignment
        ))
        for rule in all_rules:
            rows.append((
                rule,  # clean rule name only
                [f"{r['similarities'].get(rule, float('nan')):.2f}" for r in records]
            ))

        # Render rows
        for row in rows:
            print(' & '.join([row[0]] + row[1]) + ' \\\\')

        print('\\bottomrule')
        print('\\end{tabular}')


    