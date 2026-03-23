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
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

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

def plot_model_performance(y_true, results_dict):
    """
    Generates and saves individual figures for each Confusion Matrix 
    and a standalone F1-Score comparison bar chart.
    """
    sns.set_theme(style="whitegrid")
    class_names = ['Healthy', 'Pain'] 
    
    # 1. Generate Individual Confusion Matrices
    for name, y_pred in results_dict.items():
        plt.figure(figsize=(6, 5))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=class_names, 
                    yticklabels=class_names)
        
        plt.title(f"Confusion Matrix: {name}")
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        safe_name = name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        filename = f"cm_{safe_name}.png"
        plt.tight_layout()
        plt.savefig(filename, dpi=300)
        print(f"Saved: {filename}")
        plt.close()

    # 2. Generate Standalone F1-Score Comparison
    plt.figure(figsize=(10, 6))
    model_names = []
    f1_scores = []
    
    for name, y_pred in results_dict.items():
        # F1-score specifically for Class 1: Protective Behavior 
        score = f1_score(y_true, y_pred)
        model_names.append(name)
        f1_scores.append(score)
        
    ax = sns.barplot(x=model_names, y=f1_scores, palette='magma')
    plt.title('Comparison of Protective Behavior (Class 1) F1-Scores')
    plt.ylabel('F1-Score')
    plt.ylim(0, 1.0)
    
    for i, v in enumerate(f1_scores):
        ax.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    plt.tight_layout()
    plt.savefig("f1_comparison_report.png", dpi=300)
    print("Saved: f1_comparison_report.png")
    plt.show()

if __name__ == "__main__":
    # 1. Prepare Data
    X_kin, X_emg, y = prepare_dataset("train")
    if len(y) == 0:
        print("Error: No valid data windows could be extracted. Please check the dataset.")
        exit()
        
    print(f"\nData ready! Successfully extracted {len(y)} valid samples.")
    print(f"Kinematic Feature Dimension: {X_kin.shape[1]}")
    print(f"EMG Feature Dimension: {X_emg.shape[1]}")

    # Store predictions for the visualizer
    results_to_plot = {
        "Early Fusion (RF)": [],
        "Middle Fusion (MLP)": [],
        "Late Fusion (RF)": []
    }
    y_true_all = []

    # 2. Set up Cross-Validation
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    y_true_all = []
    y_pred_early_all = []
    y_pred_late_all = []
    y_pred_middle_all = []

    print("\nStarting 5-Fold Cross Validation for 3 Fusion Architectures...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_kin, y), 1):
        print(f"Training Fold {fold}/5...")
        
        # Split and Scale
        scaler = StandardScaler()
        X_kin_train = scaler.fit_transform(X_kin[train_idx])
        X_kin_test = scaler.transform(X_kin[test_idx])
        
        X_emg_train = scaler.fit_transform(X_emg[train_idx])
        X_emg_test = scaler.transform(X_emg[test_idx])
        
        y_train, y_test = y[train_idx], y[test_idx]
        y_true_all.extend(y_test)

        # ARCHITECTURE 1: EARLY FUSION 
        X_early_train = np.hstack((X_kin_train, X_emg_train))
        X_early_test = np.hstack((X_kin_test, X_emg_test))
        
        clf_early = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
        clf_early.fit(X_early_train, y_train)
        results_to_plot["Early Fusion (RF)"].extend(clf_early.predict(X_early_test))

        # ARCHITECTURE 2: MIDDLE FUSION
        clf_mid = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)
        clf_mid.fit(X_early_train, y_train)
        results_to_plot["Middle Fusion (MLP)"].extend(clf_mid.predict(X_early_test))

        # ARCHITECTURE 3: LATE FUSION
        clf_kin = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_emg = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_kin.fit(X_kin_train, y_train)
        clf_emg.fit(X_emg_train, y_train)
        
        prob_late = (clf_kin.predict_proba(X_kin_test)[:, 1] + clf_emg.predict_proba(X_emg_test)[:, 1]) / 2
        results_to_plot["Late Fusion (RF)"].extend((prob_late >= 0.5).astype(int))

    ######################################################
    print("\n" + "="*50)
    print("FINAL QUANTITATIVE EVALUATION")
    print("="*50)

    # 1. Output Precision, Recall, Accuracy
    for name, y_pred in results_to_plot.items():
        evaluate_model(np.array(y_true_all), np.array(y_pred), name)

    # 2. Trigger Visualizations
    plot_model_performance(np.array(y_true_all), results_to_plot)
