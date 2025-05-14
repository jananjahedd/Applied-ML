import numpy as np
import os
import glob
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline as SklearnPipeline 
from sklearn.metrics import classification_report, accuracy_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.utils import shuffle
from collections import defaultdict

try:
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import SMOTE
    imblearn_available = True
    print("INFO: imblearn library found. SMOTE is available.")
except ImportError:
    print("WARNING: imblearn library not found. SMOTE will not be used. "
          "To use SMOTE, please install imbalanced-learn")
    ImbPipeline = SklearnPipeline
    SMOTE = None
    imblearn_available = False

ALL_FEATURE_NAMES = [
    'Fpz-Cz_delta_RelP', 'Pz-Oz_delta_RelP', 'Fpz-Cz_theta_RelP',
    'Pz-Oz_theta_RelP', 'Fpz-Cz_alpha_RelP', 'Pz-Oz_alpha_RelP',
    'Fpz-Cz_sigma_RelP', 'Pz-Oz_sigma_RelP', 'Fpz-Cz_beta_RelP',
    'Pz-Oz_beta_RelP', 'horizontal_Var', 'submental_Mean'
]
SPLITS_DIR = "/Users/jananjahed/Desktop/ML_applied/Applied-ML/data_splits/sleep-cassette"


def load_split_data(npz_file_path):
    try:
        data = np.load(npz_file_path, allow_pickle=True)
        X_train, y_train, X_val, y_val, X_test, y_test = data['X_train'],
        data['y_train'], data['X_val'], data['y_val'], data['X_test'],
        data['y_test']

        fusion_config = "unknown_config"
        if 'fusion_configuration' in data:
            item = data['fusion_configuration'].item() if data[
                'fusion_configuration'].ndim == 0 else data[
                    'fusion_configuration']
            fusion_config = str(item)
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
                try: 
                    file_feature_names = list(loaded_features)
                except TypeError: 
                    file_feature_names = ALL_FEATURE_NAMES if X_train.shape[1] == len(ALL_FEATURE_NAMES) else None
        else:
            file_feature_names = ALL_FEATURE_NAMES if X_train.shape[1] == len(ALL_FEATURE_NAMES) else None

        if X_train.size == 0 or y_train.size == 0:
            print(f"Warning: X_train or y_train is empty in {npz_file_path}.")
            return None, None, None, None, None, None, None, None, True
        return X_train, y_train, X_val, y_val, X_test, y_test, file_feature_names, fusion_config, False
    except Exception as e:
        print(f"Error loading {npz_file_path}: {e}")
        return None, None, None, None, None, None, None, None, True


def main_svm():
    split_files = glob.glob(os.path.join(SPLITS_DIR, "split_*.npz"))
    if not split_files:
        print(f"No .npz split files found in {SPLITS_DIR}.")
        return
    print(f"Found {len(split_files)} split files to process for SVM.")

    results_by_config = defaultdict(lambda: {'accuracy': [], 'f1_macro': [],
                                             'roc_auc_ovr': [], 'count': 0})
    master_label_set = set()

    for i, split_file in enumerate(sorted(split_files)):
        print(f"\n--- Processing Split File {i+1}/{len(split_files)}: {os.path.basename(split_file)} ---")
        X_train_raw, y_train, X_val_raw, y_val, X_test_raw, y_test, file_feature_names, fusion_config, is_empty = load_split_data(split_file)

        if is_empty:
            print(f"Skipping file {split_file}.")
            continue
        if file_feature_names is None:
            print(f"Feature names undetermined for {split_file}")
            continue

        y_train, y_val, y_test = y_train.astype(int), y_val.astype(int),
        y_test.astype(int)
        master_label_set.update(y_train)
        master_label_set.update(y_val)
        master_label_set.update(y_test)

        current_config_results = results_by_config[fusion_config]
        current_config_results['count'] += 1
        print(f"Processing for Configuration: {fusion_config}")
        print("Training SVM Model (All features from this file's config)...")

        X_train_svm, X_val_svm, X_test_svm = X_train_raw, X_val_raw, X_test_raw
        if X_train_svm.size == 0 or X_test_svm.size == 0:
            print(f"Skipping SVM for this fold {fusion_config}.")
            current_config_results['accuracy'].append(np.nan)
            current_config_results['f1_macro'].append(np.nan)
            current_config_results['roc_auc_ovr'].append(np.nan)
            continue

        n_features = X_train_svm.shape[1]
        use_pca = n_features > 10

        svm_pipeline_steps = [('scaler', StandardScaler())]
        if use_pca:
            num_pca_fit_samples = X_train_svm.shape[0] + (X_val_svm.shape[0] if X_val_svm.size > 0 else 0)
            max_pca_comps = min(num_pca_fit_samples, n_features)
            pca_n_components = 0.95
            if max_pca_comps <= 1:
                use_pca = False
            if use_pca:
                svm_pipeline_steps.append(('pca', PCA(n_components=pca_n_components, random_state=42)))

        if imblearn_available and SMOTE is not None:
            svm_pipeline_steps.append(('smote', SMOTE(random_state=42)))
        svm_pipeline_steps.append(('svm', SVC(kernel='rbf', probability=True,
                                              random_state=42,
                                              class_weight='balanced')))

        CurrentPipeline = ImbPipeline if 'smote' in dict(svm_pipeline_steps) else SklearnPipeline
        svm_pipeline = CurrentPipeline(svm_pipeline_steps)

        svm_param_grid = {'svm__C': [0.1, 1, 10, 50], 'svm__gamma': [1e-4,
                                                                     1e-3,
                                                                     1e-2,
                                                                     0.1,
                                                                     'scale']}

        X_hp_train_svm = X_train_svm
        y_hp_train_svm = y_train
        if X_val_svm.size > 0 and y_val.size > 0:
            X_hp_train_svm = np.vstack((X_train_svm, X_val_svm))
            y_hp_train_svm = np.concatenate((y_train, y_val))
        X_hp_train_svm, y_hp_train_svm = shuffle(X_hp_train_svm,
                                                 y_hp_train_svm,
                                                 random_state=42)

        best_svm = None
        if X_hp_train_svm.shape[0] < 5 or len(np.unique(y_hp_train_svm)) < 2:
            print(f"Combined train+val for SVM too small. Fitting directly.")
            try:
                svm_pipeline.fit(X_hp_train_svm, y_hp_train_svm)
                best_svm = svm_pipeline
            except Exception as e:
                print(f"Error fitting SVM directly: {e}")
        else:
            min_samples_class = np.min(np.bincount(y_hp_train_svm)) if len(y_hp_train_svm)>0 else 0
            if 'smote' in svm_pipeline.named_steps and SMOTE is not None:
                k_val = min(5, min_samples_class - 1) if min_samples_class > 1 else 1
                if k_val < 1:
                    print("Removing SMOTE as k_neighbors too small.")
                    svm_pipeline_steps_no_smote = [s for s in svm_pipeline_steps if s[0]!='smote']
                    svm_pipeline = SklearnPipeline(svm_pipeline_steps_no_smote)
                else:
                    svm_pipeline.set_params(smote__k_neighbors=k_val)

            n_splits_cv = min(5, min_samples_class)
            if n_splits_cv < 2:
                print(f"Warning: Cannot perform {n_splits_cv}-fold CV for SVM. Fitting directly.")
                try:
                    svm_pipeline.fit(X_hp_train_svm, y_hp_train_svm)
                    best_svm = svm_pipeline
                except Exception as e: 
                    print(f"Error fitting SVM directly: {e}")
            else:
                gs = GridSearchCV(svm_pipeline, svm_param_grid, cv=n_splits_cv,
                                  scoring='f1_macro', n_jobs=-1,
                                  error_score='raise')
                try:
                    gs.fit(X_hp_train_svm, y_hp_train_svm)
                    best_svm = gs.best_estimator_
                    print(f"Best SVM Params: {gs.best_params_}")
                except Exception as e: 
                    print(f"Error in GridSearchCV for SVM: {e}")
                    best_svm = None

        if best_svm and X_test_svm.size > 0:
            y_pred = best_svm.predict(X_test_svm)
            current_config_results['accuracy'].append(accuracy_score(y_test,
                                                                     y_pred))
            current_config_results['f1_macro'].append(f1_score(y_test, y_pred,
                                                               average='macro',
                                                               zero_division=0))

            sorted_labels = sorted(list(master_label_set))
            try:
                y_proba = best_svm.predict_proba(X_test_svm)
                if len(np.unique(y_test)) > 1 and y_proba.shape[1] >= len(sorted_labels):
                    roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr',
                                            average='macro',
                                            labels=sorted_labels)
                    current_config_results['roc_auc_ovr'].append(roc_auc)
                else:
                    current_config_results['roc_auc_ovr'].append(np.nan)
            except Exception as e:
                current_config_results['roc_auc_ovr'].append(np.nan)
                print(f"SVM ROC AUC error: {e}")

            print(f"SVM Test Performance (Config: {fusion_config}, Fold: {os.path.basename(split_file)}):")
            print(classification_report(y_test, y_pred, zero_division=0,
                                        labels=sorted_labels,
                                        target_names=[f"Class {l}" for l in sorted_labels]))
        else:
            current_config_results['accuracy'].append(np.nan)
            current_config_results['f1_macro'].append(np.nan)
            current_config_results['roc_auc_ovr'].append(np.nan)

    print("\n\n--- Overall Results for SVM Model ---")
    for config_name, results in results_by_config.items():
        if results['count'] > 0 and any(not np.isnan(x) for x in results['accuracy']):
            print(f"\nConfiguration: {config_name} (Processed {results['count']} files)")
            print(f"  Mean Accuracy: {np.nanmean(results['accuracy']):.4f} +/- {np.nanstd(results['accuracy']):.4f}")
            print(f"  Mean Macro F1-score: {np.nanmean(results['f1_macro']):.4f} +/- {np.nanstd(results['f1_macro']):.4f}")
            print(f"  Mean ROC AUC (OVR Macro): {np.nanmean(results['roc_auc_ovr']):.4f} +/- {np.nanstd(results['roc_auc_ovr']):.4f}")
        else:
            print(f"\nNo valid results for SVM for configuration: {config_name}")


if __name__ == '__main__':
    main_svm()
