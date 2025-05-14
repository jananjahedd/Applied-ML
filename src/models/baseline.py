import numpy as np
import os
import glob
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import shuffle
from collections import defaultdict


ALL_FEATURE_NAMES = [
    'Fpz-Cz_delta_RelP', 'Pz-Oz_delta_RelP', 'Fpz-Cz_theta_RelP', 'Pz-Oz_theta_RelP',
    'Fpz-Cz_alpha_RelP', 'Pz-Oz_alpha_RelP', 'Fpz-Cz_sigma_RelP', 'Pz-Oz_sigma_RelP',
    'Fpz-Cz_beta_RelP', 'Pz-Oz_beta_RelP', 'horizontal_Var', 'submental_Mean'
]

EEG_FEATURE_NAMES = [
    'Fpz-Cz_delta_RelP', 'Pz-Oz_delta_RelP', 'Fpz-Cz_theta_RelP', 'Pz-Oz_theta_RelP',
    'Fpz-Cz_alpha_RelP', 'Pz-Oz_alpha_RelP', 'Fpz-Cz_sigma_RelP', 'Pz-Oz_sigma_RelP',
    'Fpz-Cz_beta_RelP', 'Pz-Oz_beta_RelP'
]

SPLITS_DIR = "/Users/jananjahed/Desktop/ML_applied/Applied-ML/data_splits/sleep-cassette"


def load_split_data(npz_file_path):
    """
    Loads data and fusion configuration from a single .npz split file.
    """
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        X_train = data['X_train']
        y_train = data['y_train']
        X_val = data['X_val']
        y_val = data['y_val']
        X_test = data['X_test']
        y_test = data['y_test']

        fusion_config = "unknown_config"
        if 'fusion_configuration' in data:
            fusion_config = str(data['fusion_configuration'].item()) if data['fusion_configuration'].ndim == 0 else str(data['fusion_configuration'])
        else:
            basename = os.path.basename(npz_file_path)
            parts = basename.replace(".npz", "").split('_')
            if len(parts) > 4:
                fusion_config = "_".join(parts[5:])

        file_feature_names = None
        if 'feature_names' in data:
            loaded_features = data['feature_names']
            if isinstance(loaded_features, np.ndarray) and loaded_features.ndim == 0 and isinstance(loaded_features.item(), list):
                file_feature_names = loaded_features.item()
            elif isinstance(loaded_features, np.ndarray):
                file_feature_names = list(loaded_features)
            elif isinstance(loaded_features, list):
                file_feature_names = loaded_features
            else:
                print(f"Warning: 'feature_names' in {npz_file_path} is of type {type(loaded_features)}. Trying list conversion.")
                try:
                    file_feature_names = list(loaded_features)
                except TypeError:
                    print("Error: Could not convert 'feature_names'. Fallback needed.")
                    file_feature_names = ALL_FEATURE_NAMES if X_train.shape[1] == len(ALL_FEATURE_NAMES) else None
        else:
            print(f"Warning: 'feature_names' not in {npz_file_path}. Fallback needed.")
            file_feature_names = ALL_FEATURE_NAMES if X_train.shape[1] == len(ALL_FEATURE_NAMES) else None

        if X_train.size == 0 or y_train.size == 0:
            print(f"Warning: X_train or y_train is empty in {npz_file_path}.")
            return None, None, None, None, None, None, None, None, True

        return X_train, y_train, X_val, y_val, X_test, y_test, file_feature_names, fusion_config, False
    except Exception as e:
        print(f"Error loading {npz_file_path}: {e}")
        return None, None, None, None, None, None, None, None, True


def select_features(X, features_in_file, desired_features):
    if X is None or X.size == 0:
        return np.array([])
    if features_in_file is None:
        print(f"'features_in_file' is None. Cannot select. Shape of X: {X.shape}")
        if X.shape[1] == len(desired_features):
            print("Assuming X columns match desired features due to name list missing.")
            return X
        return np.array([])

    try:
        if not isinstance(features_in_file, list):
            features_in_file = list(features_in_file)

        indices_to_select = []
        for desired_feat_name in desired_features:
            try:
                indices_to_select.append(features_in_file.index(desired_feat_name))
            except ValueError:
                pass

        if not indices_to_select:
            return np.array([])
        return X[:, indices_to_select]
    except Exception as e:
        print(f"An unexpected error occurred during feature selection: {e}")
        return np.array([])


def main_lr():
    split_files = glob.glob(os.path.join(SPLITS_DIR, "split_*.npz"))
    if not split_files:
        print(f"No .npz split files found in {SPLITS_DIR}.")
        return

    print(f"Found {len(split_files)} split files to process for Logistic Regression.")

    results_by_config = defaultdict(lambda: {'accuracy': [], 'f1_macro': [],
                                             'roc_auc_ovr': [], 'reports': [],
                                             'conf_matrices': [], 'count': 0})
    master_label_set = set()


    for i, split_file in enumerate(sorted(split_files)): 
        print(f"\n--- Processing Split File {i+1}/{len(split_files)}: {os.path.basename(split_file)} ---")

        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, file_feature_names, fusion_config, is_empty = load_split_data(split_file)

        if is_empty:
            print(f"Skipping file {split_file}.")
            continue

        if file_feature_names is None and X_train_raw is not None :
            if X_train_raw.shape[1] != len(EEG_FEATURE_NAMES):
                print(f"Feature names undetermined and X_train shape {X_train_raw.shape} not matching default EEG features for {split_file}. Skipping.")
                continue
            else:
                file_feature_names = EEG_FEATURE_NAMES

        y_train, y_val, y_test = y_train.astype(int), y_val.astype(int),
        y_test.astype(int)
        master_label_set.update(y_train)
        master_label_set.update(y_val)
        master_label_set.update(y_test)

        current_config_results = results_by_config[fusion_config]
        current_config_results['count'] += 1

        print(f"Processing for Configuration: {fusion_config}")
        print("Training Logistic Regression Baseline (EEG Features only)...")

        X_train_lr = select_features(X_train_raw, file_feature_names,
                                     EEG_FEATURE_NAMES)
        X_val_lr = select_features(X_val_raw, file_feature_names,
                                   EEG_FEATURE_NAMES)
        X_test_lr = select_features(X_test_raw, file_feature_names,
                                    EEG_FEATURE_NAMES)

        if X_train_lr.size == 0 or X_test_lr.size == 0 :
            print(f"Skipping LR for this fold (config: {fusion_config}): Empty feature arrays after EEG selection.")
            current_config_results['accuracy'].append(np.nan)
            current_config_results['f1_macro'].append(np.nan)
            current_config_results['roc_auc_ovr'].append(np.nan)
            continue

        lr_pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logreg', LogisticRegression(solver='liblinear', multi_class='ovr', random_state=42, class_weight='balanced', max_iter=1000))
        ])
        lr_param_grid = {'logreg__C': [1e-3, 1e-2, 0.1, 1, 10, 100]}

        best_lr = None
        X_hp_train_lr = X_train_lr
        y_hp_train_lr = y_train
        if X_val_lr.size > 0 and y_val.size > 0:
            X_hp_train_lr = np.vstack((X_train_lr, X_val_lr))
            y_hp_train_lr = np.concatenate((y_train, y_val))

        X_hp_train_lr, y_hp_train_lr = shuffle(X_hp_train_lr, y_hp_train_lr,
                                               random_state=42)

        if X_hp_train_lr.shape[0] < 5 or len(np.unique(y_hp_train_lr)) < 2 :
            print(f"Warning: Combined train+val for LR is too small ("
                  f"{X_hp_train_lr.shape[0]} samples, "
                  f"{len(np.unique(y_hp_train_lr))} classes).")
            try:
                lr_pipeline.fit(X_hp_train_lr, y_hp_train_lr)
                best_lr = lr_pipeline
            except Exception as e:
                print(f"Error fitting LR directly: {e}")
        else:
            min_samples_class = np.min(np.bincount(y_hp_train_lr)) if len(y_hp_train_lr) > 0 else 0
            n_splits_cv = min(5, min_samples_class)
            if n_splits_cv < 2:
                print(f"Cannot perform {n_splits_cv}-fold CV for LR.")
                try:
                    lr_pipeline.fit(X_hp_train_lr, y_hp_train_lr)
                    best_lr = lr_pipeline
                except Exception as e:
                    print(f"Error fitting LR directly: {e}")
            else:
                gs = GridSearchCV(lr_pipeline, lr_param_grid, cv=n_splits_cv,
                                  scoring='f1_macro',
                                  n_jobs=-1,
                                  error_score='raise')
                try:
                    gs.fit(X_hp_train_lr, y_hp_train_lr)
                    best_lr = gs.best_estimator_
                    print(f"Best LR Params: {gs.best_params_}")
                except Exception as e:
                    print(f"Error in GridSearchCV for LR: {e}")
                    best_lr = None

        if best_lr and X_test_lr.size > 0:
            y_pred = best_lr.predict(X_test_lr)
            current_config_results['accuracy'].append(accuracy_score(y_test,
                                                                     y_pred))
            current_config_results['f1_macro'].append(f1_score(
                y_test, y_pred, average='macro', zero_division=0))

            sorted_labels = sorted(list(master_label_set))
            try:
                y_proba = best_lr.predict_proba(X_test_lr)
                if len(np.unique(y_test)) > 1 and y_proba.shape[1] >= len(sorted_labels):
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr',
                                            average='macro',
                                            labels=sorted_labels)
                    current_config_results['roc_auc_ovr'].append(roc_auc)
                else:
                    current_config_results['roc_auc_ovr'].append(np.nan)
            except Exception as e:
                current_config_results['roc_auc_ovr'].append(np.nan)
                print(f"LR ROC AUC error: {e}")

            print(f"LR Test Performance (Config: {fusion_config}, Fold: {os.path.basename(split_file)}):")
            print(classification_report(y_test, y_pred, zero_division=0,
                                        labels=sorted_labels,
                                        target_names=[f"Class {l}" for l in sorted_labels]))
        else:
            current_config_results['accuracy'].append(np.nan)
            current_config_results['f1_macro'].append(np.nan)
            current_config_results['roc_auc_ovr'].append(np.nan)

    print("\n\n--- Overall Results for Logistic Regression Baseline ---")
    for config_name, results in results_by_config.items():
        if results['count'] > 0 and any(not np.isnan(x) for x in results['accuracy']):
            print(f"\nConfiguration: {config_name} (Processed {results['count']} files)")
            print(f"  Mean Accuracy: {np.nanmean(results['accuracy']):.4f} +/- {np.nanstd(results['accuracy']):.4f}")
            print(f"  Mean Macro F1-score: {np.nanmean(results['f1_macro']):.4f} +/- {np.nanstd(results['f1_macro']):.4f}")
            print(f"  Mean ROC AUC (OVR Macro): {np.nanmean(results['roc_auc_ovr']):.4f} +/- {np.nanstd(results['roc_auc_ovr']):.4f}")
        else:
            print(f"\nNo valid results: {config_name}")


if __name__ == '__main__':
    main_lr()
