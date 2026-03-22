import os
# =====================================================================
# Disable Intel Fortran's aggressive crash handler
# This prevents the "forrtl: error (200): program aborting due to control-C event"
# =====================================================================
os.environ['FOR_DISABLE_CONSOLE_CTRL_HANDLER'] = '1'

import numpy as np
import glob
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score

import ARS

def prepare_dataset(data_dir="train"):
    """
    Step 1: Use ARS.py to load data, create windows, and extract features.
    Returns X_kin (Kinematic features), X_emg (EMG features), and y (labels).
    """
    print("Loading data and extracting features... (This might take a few minutes)")
    data_list = ARS.load_emopain_data(data_dir)
    
    X_kin_list = []
    X_emg_list = []
    y_list = []
    
    total_files = len(data_list)
    
    for idx, item in enumerate(data_list):
        print(f"Processing file {idx+1}/{total_files}: {item['filename']} ...")
        
        matrix = item['matrix']
        
        # Create sliding windows 
        windows, labels = ARS.create_windows(matrix, window_size=180, overlap=0.75)
        
        valid_windows_count = 0
        for i in range(len(windows)):
            win = windows[i]
            lbl = labels[i]
            
            # 1. Check for NaN or Infinity
            if np.isnan(win).any() or np.isinf(win).any():
                continue 
                
            # 2. CRITICAL FIX: Only check sensor columns (0 to 69) for "dead" signals!
            # Do NOT check label columns (70+) because they are supposed to be constant!
            if np.any(np.std(win[:, :70], axis=0) == 0):
                continue
                
            # Extract features
            try:
                k_feats = ARS.extract_kinematic_features(win)
                e_feats = ARS.extract_emg_features(win)
                
                # Double check if extracted features contain NaNs
                if np.isnan(list(k_feats.values())).any() or np.isnan(list(e_feats.values())).any():
                    continue
                
                # Success! Add to our lists
                X_kin_list.append(list(k_feats.values()))
                X_emg_list.append(list(e_feats.values()))
                y_list.append(lbl)
                valid_windows_count += 1
                
            except Exception as e:
                continue
                
        print(f"  -> Extracted {valid_windows_count} valid windows.")

    return np.array(X_kin_list), np.array(X_emg_list), np.array(y_list)

def evaluate_model(y_true, y_pred, model_name):
    """
    Step 2: Standardized evaluation function to get F1 scores and Confusion Matrix 
    """
    print(f"\n{'='*40}")
    print(f"RESULTS FOR: {model_name}")
    print(f"{'='*40}")
    print("Classification Report (Includes F1-score per label and average):")
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("="*40)


if __name__ == "__main__":
    # 1. Prepare Data
    X_kin, X_emg, y = prepare_dataset("train")
    
    if len(y) == 0:
        print("Error: No valid data windows could be extracted. Please check the dataset.")
        exit()
        
    print(f"\nData ready! Successfully extracted {len(y)} valid samples.")
    print(f"Kinematic Feature Dimension: {X_kin.shape[1]}")
    print(f"EMG Feature Dimension: {X_emg.shape[1]}")

    # 2. Set up Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_true_all = []
    y_pred_early_all = []
    y_pred_late_all = []
    y_pred_middle_all = []

    print("\nStarting 5-Fold Cross Validation for 3 Fusion Architectures...")
    
    for fold, (train_index, test_index) in enumerate(skf.split(X_kin, y), 1):
        print(f"Training Fold {fold}/5 ...")
        
        X_kin_train, X_kin_test = X_kin[train_index], X_kin[test_index]
        X_emg_train, X_emg_test = X_emg[train_index], X_emg[test_index]
        y_train, y_test = y[train_index], y[test_index]
        y_true_all.extend(y_test)

        # ARCHITECTURE 1: EARLY FUSION 
        X_early_train = np.hstack((X_kin_train, X_emg_train))
        X_early_test = np.hstack((X_kin_test, X_emg_test))
        
        clf_early = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_early.fit(X_early_train, y_train)
        y_pred_early_all.extend(clf_early.predict(X_early_test))

        # ARCHITECTURE 2: LATE FUSION 
        clf_kin = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        clf_emg = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        
        clf_kin.fit(X_kin_train, y_train)
        clf_emg.fit(X_emg_train, y_train)
        
        prob_kin = clf_kin.predict_proba(X_kin_test)[:, 1]
        prob_emg = clf_emg.predict_proba(X_emg_test)[:, 1]
        
        prob_late_fusion = (prob_kin + prob_emg) / 2.0
        pred_late = (prob_late_fusion >= 0.5).astype(int)
        y_pred_late_all.extend(pred_late)

        # ARCHITECTURE 3: MIDDLE FUSION 
        clf_middle = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf_middle.fit(X_early_train, y_train) 
        y_pred_middle_all.extend(clf_middle.predict(X_early_test))

    # 3. Final Evaluation
    evaluate_model(y_true_all, y_pred_early_all, "EARLY FUSION (Random Forest on Concatenated Features)")
    evaluate_model(y_true_all, y_pred_middle_all, "MIDDLE FUSION (Multi-Layer Perceptron)")
    evaluate_model(y_true_all, y_pred_late_all, "LATE FUSION (Averaging Probabilities)")