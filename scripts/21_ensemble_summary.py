"""
Step 21: Ensemble Summary Report
Aggregates per-antibiotic ensemble results into a cross-antibiotic summary
report with comparative visualizations.
"""

import json
import sys
import os
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.output_validation import OutputValidator


def load_all_ensemble_results(input_files):
    """Load ensemble analysis results for all antibiotics."""
    results = {}
    for filepath in input_files:
        filepath = Path(filepath)
        if not filepath.exists():
            print(f"Warning: {filepath} not found, skipping")
            continue
        try:
            with open(filepath) as f:
                data = json.load(f)
            antibiotic = data.get('antibiotic', filepath.stem.replace('_ensemble_analysis', ''))
            results[antibiotic] = data
            print(f"Loaded results for {antibiotic}")
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load {filepath}: {e}")
    return results


def build_summary(all_results):
    """Build cross-antibiotic summary statistics."""
    summary = {
        'antibiotics_analysed': list(all_results.keys()),
        'n_antibiotics': len(all_results),
        'per_antibiotic': {},
        'overall': {}
    }

    best_f1_scores = []
    best_methods = []
    improvements = []
    individual_best_f1s = []

    for abx, data in all_results.items():
        entry = {
            'individual_performance': data.get('individual_performance', {}),
            'n_ensemble_methods': len(data.get('ensemble_methods', {})),
        }

        if 'best_ensemble' in data:
            best = data['best_ensemble']
            perf = best.get('performance', {})
            entry['best_ensemble_method'] = best.get('method_name', 'unknown')
            entry['best_ensemble_f1'] = perf.get('f1', 0)
            entry['best_ensemble_accuracy'] = perf.get('accuracy', 0)
            entry['best_ensemble_auroc'] = perf.get('auroc', 0)

            best_f1_scores.append(perf.get('f1', 0))
            best_methods.append(best.get('method_name', 'unknown'))

            imp = best.get('improvement_over_best_individual', {})
            if imp:
                entry['f1_improvement'] = imp.get('f1_improvement', 0)
                improvements.append(imp.get('f1_improvement', 0))

        # Best individual model for this antibiotic
        ind_perf = data.get('individual_performance', {})
        if ind_perf:
            best_ind = max(ind_perf.items(), key=lambda x: x[1].get('f1', 0))
            entry['best_individual_model'] = best_ind[0]
            entry['best_individual_f1'] = best_ind[1].get('f1', 0)
            individual_best_f1s.append(best_ind[1].get('f1', 0))

        # Success criteria
        entry['success_criteria'] = data.get('success_criteria', {})

        summary['per_antibiotic'][abx] = entry

    # Overall statistics
    if best_f1_scores:
        summary['overall'] = {
            'mean_ensemble_f1': float(np.mean(best_f1_scores)),
            'std_ensemble_f1': float(np.std(best_f1_scores)),
            'min_ensemble_f1': float(np.min(best_f1_scores)),
            'max_ensemble_f1': float(np.max(best_f1_scores)),
            'mean_improvement_over_individual': float(np.mean(improvements)) if improvements else 0,
            'mean_individual_best_f1': float(np.mean(individual_best_f1s)) if individual_best_f1s else 0,
            'most_common_best_method': max(set(best_methods), key=best_methods.count) if best_methods else 'none',
        }

    return summary


def plot_cross_antibiotic_comparison(summary, output_dir):
    """Create cross-antibiotic visualization of ensemble vs individual performance."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    antibiotics = []
    ensemble_f1s = []
    individual_f1s = []
    model_names = ['xgboost', 'lightgbm', 'cnn', 'sequence_cnn', 'dnabert']

    for abx, data in summary['per_antibiotic'].items():
        antibiotics.append(abx.title())
        ensemble_f1s.append(data.get('best_ensemble_f1', 0))
        individual_f1s.append(data.get('best_individual_f1', 0))

    if not antibiotics:
        print("No data available for plotting")
        return

    # --- Plot 1: Ensemble vs Best Individual ---
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(antibiotics))
    width = 0.35

    bars1 = ax.bar(x - width / 2, individual_f1s, width, label='Best Individual Model',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width / 2, ensemble_f1s, width, label='Best Ensemble',
                   color='darkorange', alpha=0.8)

    ax.set_ylabel('F1 Score')
    ax.set_title('Ensemble vs Best Individual Model Performance')
    ax.set_xticks(x)
    ax.set_xticklabels(antibiotics, rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'cross_antibiotic_ensemble_vs_individual.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f"Saved: cross_antibiotic_ensemble_vs_individual.png")

    # --- Plot 2: Per-model heatmap ---
    all_model_f1 = {model: [] for model in model_names}
    for abx_data in summary['per_antibiotic'].values():
        ind_perf = abx_data.get('individual_performance', {})
        for model in model_names:
            f1 = ind_perf.get(model, {}).get('f1', np.nan)
            all_model_f1[model].append(f1)

    if any(not all(np.isnan(v) for v in vals) for vals in all_model_f1.values()):
        fig, ax = plt.subplots(figsize=(10, 5))
        matrix = np.array([all_model_f1[m] for m in model_names])
        im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

        ax.set_xticks(range(len(antibiotics)))
        ax.set_xticklabels(antibiotics, rotation=15)
        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels([m.replace('_', ' ').title() for m in model_names])

        # Annotate cells
        for i in range(len(model_names)):
            for j in range(len(antibiotics)):
                val = matrix[i, j]
                if not np.isnan(val):
                    ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                            color='black' if 0.3 < val < 0.8 else 'white', fontsize=9)

        plt.colorbar(im, label='F1 Score')
        ax.set_title('Model Performance Across Antibiotics')
        plt.tight_layout()
        plt.savefig(output_dir / 'model_performance_heatmap.png', dpi=150,
                    bbox_inches='tight')
        plt.close()
        print(f"Saved: model_performance_heatmap.png")

    # --- Plot 3: Improvement waterfall ---
    if any('f1_improvement' in d for d in summary['per_antibiotic'].values()):
        fig, ax = plt.subplots(figsize=(8, 5))
        imp_values = [summary['per_antibiotic'][abx.lower()].get('f1_improvement', 0)
                      for abx in [a.lower() for a in antibiotics]]
        # Re-get with proper case
        imp_values = []
        abx_labels = []
        for abx, data in summary['per_antibiotic'].items():
            imp = data.get('f1_improvement', 0)
            imp_values.append(imp)
            abx_labels.append(abx.title())

        colors = ['forestgreen' if v > 0 else 'firebrick' for v in imp_values]
        ax.bar(abx_labels, imp_values, color=colors, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_ylabel('F1 Improvement')
        ax.set_title('Ensemble Improvement Over Best Individual Model')

        for i, v in enumerate(imp_values):
            ax.text(i, v + 0.002 if v >= 0 else v - 0.008,
                    f'{v:+.3f}', ha='center', va='bottom' if v >= 0 else 'top', fontsize=9)

        plt.tight_layout()
        plt.savefig(output_dir / 'ensemble_improvement.png', dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: ensemble_improvement.png")


def main():
    """Main entry point."""
    antibiotic_files = snakemake.input
    output_summary = snakemake.output.summary
    output_plots_dir = snakemake.output.plots

    print("=" * 60)
    print("Ensemble Summary Report")
    print("=" * 60)

    # Load all per-antibiotic ensemble results
    all_results = load_all_ensemble_results(antibiotic_files)

    if not all_results:
        print("ERROR: No ensemble results found!")
        summary = {'error': 'No ensemble results available'}
    else:
        # Build summary
        summary = build_summary(all_results)

        # Print key findings
        overall = summary.get('overall', {})
        print(f"\nAnalysed {summary['n_antibiotics']} antibiotics: {summary['antibiotics_analysed']}")
        if overall:
            print(f"Mean ensemble F1: {overall['mean_ensemble_f1']:.3f} "
                  f"(+/- {overall['std_ensemble_f1']:.3f})")
            print(f"Mean improvement over best individual: "
                  f"{overall['mean_improvement_over_individual']:+.3f}")
            print(f"Most common best method: {overall['most_common_best_method']}")

        # Generate plots
        plot_cross_antibiotic_comparison(summary, output_plots_dir)

    # Save summary JSON
    validator = OutputValidator()
    validator.save_standardized_results(summary, output_summary, 'ensemble_summary')
    print(f"\nSummary saved to {output_summary}")


if __name__ == "__main__":
    main()
