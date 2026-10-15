"""
Utilities for LOSO LSTM training.
"""

from .data import (
    get_speaker_emotion_profile,
    calculate_fold_balance_score,
    print_dataset_info,
)

from .split import (
    find_best_speaker_split,
    load_fixed_speaker_split,
    stratified_loso_order
)

from .training import (
    prepare_fold_data,
    setup_model_and_optimizer,
    create_criterion,
    train_fold,
    validate_model
)

from .visualization import (
    plot_training_curves,
    plot_confusion_matrix,
    plot_fold_performance
)

from .results import (
    save_results_to_files,
    create_results_dict,
    print_final_results
)

__all__ = [
    # data
    'get_speaker_emotion_profile',
    'calculate_fold_balance_score',
    'print_dataset_info',

    # split
    'find_best_speaker_split',
    'load_fixed_speaker_split',
    'stratified_loso_order',

    # training
    'prepare_fold_data',
    'setup_model_and_optimizer',
    'create_criterion',
    'train_fold',
    'validate_model',

    # visualization
    'plot_training_curves',
    'plot_confusion_matrix',
    'plot_fold_performance',

    # results
    'save_results_to_files',
    'create_results_dict',
    'print_final_results'
]
