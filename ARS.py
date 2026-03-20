import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_emopain_data(data_dir):
    """
    Iterates through folders and loads .mat files with metadata.
    """
    data_list = []
    # Folders are 'train' and 'validation' [cite: 3, 4, 5]
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mat'):
                # Extract metadata from filename 
                # Example: P01_N.mat -> Prefix 'P', ID '01', Suffix 'N'
                participant_type = 'Chronic Pain' if file.startswith('P') else 'Healthy'
                difficulty = 'Normal' if file.split('_')[-1].startswith('N') else 'Difficult'
                
                file_path = os.path.join(root, file)
                mat_contents = loadmat(file_path)
                
                # Each file contains an N x 78 matrix 
                # Note: Replace 'data' with the actual key inside your .mat files
                matrix = mat_contents.get('data') 
                
                data_list.append({
                    'matrix': matrix,
                    'type': participant_type,
                    'difficulty': difficulty,
                    'filename': file
                })
    return data_list


def create_windows(matrix, window_size=180, overlap=0.75):
    step = int(window_size * (1 - overlap))
    windows = []
    labels = []
    
    for i in range(0, len(matrix) - window_size + 1, step):
        window = matrix[i : i + window_size]
        
        # Segmenting by exercise instance ensures no overlap between 
        # different exercises [cite: 24]
        # Check if the exercise type (Column 71) is consistent in the window
        if len(np.unique(window[:, 70])) == 1: 
            windows.append(window)
            # Use Majority Voting or the last frame for the label (Column 73) 
            label = 1 if np.mean(window[:, 72]) > 0.5 else 0 
            labels.append(label)
            
    return np.array(windows), np.array(labels)

def get_joint_coords(row):
    """
    Extracts the 22 joints into an (22, 3) array.
    Columns 1-22: X, 23-44: Y, 45-66: Z.
    """
    # Adjust for 0-indexing
    x = row[0:22]
    y = row[22:44]
    z = row[44:66]
    
    # Stack them to get [[x1, y1, z1], [x2, y2, z2], ...]
    return np.column_stack((x, y, z))

def calculate_angle(joint_coords, idx_a, idx_vertex, idx_c):
    """
    Calculates the 3D angle at a vertex joint between two other joints.
    idx_a, idx_vertex, idx_c are the 0-based indices of the joints (0-21).
    """
    a = joint_coords[idx_a]
    v = joint_coords[idx_vertex]
    c = joint_coords[idx_c]
    
    # Create vectors relative to the vertex
    ba = a - v
    bc = c - v
    
    # Calculate cosine of the angle using dot product
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    
    # Clip to avoid numerical errors outside [-1, 1]
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    
    return np.degrees(angle) # Returns angle in degrees

def extract_kinematic_features(window):
    """
    Processes a window of data (e.g., 180 frames) to extract mean angles.
    """
    angles_in_window = []
    
    for row in window:
        joints = get_joint_coords(row)
        
        # Example: Calculate the angle of the lumbar spine or a knee
        # You will need to map these indices based on Figure 1 in the README [cite: 14]
        knee_angle = calculate_angle(joints, idx_a=15, idx_vertex=16, idx_c=17) 
        angles_in_window.append(knee_angle)
    
    # Feature extraction: return the mean and standard deviation of the angle 
    # to capture movement range and 'stiffness' 
    return {
        'mean_angle': np.mean(angles_in_window),
        'std_angle': np.std(angles_in_window)
    }

def extract_emg_features(window):
    """
    Extracts RMS features for the 4 EMG channels.
    Columns 67-70 (Indices 66-69).
    """
    # Extract the EMG columns
    emg_data = window[:, 66:70] 
    
    # Calculate RMS: sqrt(mean(square(x))) for each channel
    rms_features = np.sqrt(np.mean(np.square(emg_data), axis=0))
    
    return {
        'lumbar_R_rms': rms_features[0],
        'lumbar_L_rms': rms_features[1],
        'trapezius_R_rms': rms_features[2],
        'trapezius_L_rms': rms_features[3]
    }

if __name__ == "__main__":
    import glob

    # 1. Test File Loading
    print("--- Phase 1: Data Loading ---")
    train_files = glob.glob("train/*.mat")
    
    if not train_files:
        print("Error: No .mat files found in 'train/' folder. Check your directory structure.")
    else:
        sample_path = train_files[0]
        print(f"Loading sample file: {sample_path}")
        
        # Load .mat file
        mat_data = loadmat(sample_path)
        # Note: 'data' is the typical key, but verify if your .mat uses a different name
        matrix = mat_data.get('data') 
        
        print(f"Matrix shape: {matrix.shape}") # Should be (N, 78)
        
        if matrix.shape[1] != 78:
            print(f"Warning: Expected 78 columns, found {matrix.shape[1]}.")

        # 2. Test Feature Extraction on first 180 frames (3 seconds)
        print("\n--- Phase 2: Feature Extraction ---")
        if len(matrix) >= 180:
            test_window = matrix[0:180, :]
            
            # Kinematic Test
            k_feats = extract_kinematic_features(test_window)
            print(f"Kinematic Features: {k_feats}")
            
            # EMG Test
            e_feats = extract_emg_features(test_window)
            print(f"EMG Features: {e_feats}")
            
            # 3. Fusion Check
            combined_vector = list(k_feats.values()) + list(e_feats.values())
            print(f"\nFused Feature Vector Length: {len(combined_vector)}")
            print("Successfully processed one window!")
        else:
            print("File too short for a full 180-frame window.")