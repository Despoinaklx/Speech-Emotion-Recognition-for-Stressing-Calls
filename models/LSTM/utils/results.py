"""
Saving and printing LOSO results.
"""
import pandas as pd
from datetime import datetime
import os
import numpy as np


def save_results_to_files(results_dict, results_dir):
    """Write per-fold metrics to a timestamped CSV."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        fold_metrics = []
        for i, fold_result in enumerate(results_dict['fold_results']):
            fold_metrics.append({
                'fold': i + 1,
                'validation_speaker': fold_result['validation_speaker'],
                'final_accuracy': fold_result['final_accuracy'],
                'final_f1_macro': fold_result['final_f1_macro'],
                'best_epoch': fold_result['best_epoch'],
                'total_epochs': fold_result['total_epochs'],
                'balance_score': fold_result.get('balance_score', 'N/A')
            })

        df_metrics = pd.DataFrame(fold_metrics)
        csv_path = os.path.join(results_dir, f'fold_metrics_8fold_loso_{timestamp}.csv')
        df_metrics.to_csv(csv_path, index=False, encoding='utf-8')
        print(f"CSV saved to: {csv_path}")

    except Exception as e:
        print(f"Error saving CSV: {e}")


def create_results_dict(config, dataset_info, fold_results, overall_metrics, classification_report):
    """Collect all results in one dict."""
    return {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'max_epochs': config.MAX_EPOCHS,
            'patience': config.PATIENCE,
            'learning_rate': config.LEARNING_RATE,
            'batch_size': config.BATCH_SIZE,
            'device': str(config.device),
            'methodology': '8-Fold Stratified LOSO with GroupShuffleSplit'
        },
        'dataset_info': dataset_info,
        'overall_accuracy': overall_metrics['accuracy'],
        'overall_f1_macro': overall_metrics['f1_macro'],
        'overall_f1_weighted': overall_metrics['f1_weighted'],
        'average_balance_score': np.mean([r['balance_score'] for r in fold_results]),
        'classification_report': classification_report,
        'fold_results': fold_results,
        'predictions': {
            'y_true': overall_metrics['y_true'],
            'y_pred': overall_metrics['y_pred']
        }
    }


def print_final_results(overall_metrics, fold_results, train_speakers, test_speakers):
    """Print the final LOSO summary."""
    print('\n' + '=' * 70)
    print('8-FOLD STRATIFIED LOSO CROSS-VALIDATION RESULTS')
    print('=' * 70)

    accuracy = overall_metrics['accuracy']
    f1_macro = overall_metrics['f1_macro']
    f1_weighted = overall_metrics['f1_weighted']

    print(f'8-Fold Stratified LOSO Accuracy: {accuracy:.4f}')
    print(f'8-Fold Stratified LOSO F1-Macro: {f1_macro:.4f}')
    print(f'8-Fold Stratified LOSO F1-Weighted: {f1_weighted:.4f}')

    balance_scores = [result['balance_score'] for result in fold_results]
    avg_balance_score = np.mean(balance_scores)
    print(f'Average Balance Score: {avg_balance_score:.4f} (lower = better)')

    print(f'\nTraining speakers used: {train_speakers} (8 speakers)')
    print(f'Held-out test speakers: {test_speakers} (2 speakers)')

    print('\n8-Fold LOSO Classification Report:')
    print(overall_metrics['classification_report'])

    return avg_balance_score
