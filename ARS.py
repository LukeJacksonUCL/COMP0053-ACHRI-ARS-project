import os
import numpy as np
import pandas as pd
from scipy.io import loadmat

def load_emopain_data(data_dir):
    """
    Iterates through folders and loads .mat files with metadata.
    """
    data_list = []
    for root, dirs, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mat'):
                #Extract metadata from filename (e.g. P01_N.mat -> Prefix 'P', ID '01', Suffix 'N')
                participant_type = 'Chronic Pain' if file.startswith('P') else 'Healthy'
                difficulty = 'Normal' if file.split('_')[-1].startswith('N') else 'Difficult'
                contents = loadmat(os.path.join(root, file))
                #Each file contains an N x 78 matrix 
                matrix = contents.get('data')
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
        #segmenting by exercise instance ensures no overlap between different exercises
        #check if the exercise type (Column 71) is consistent in the window
        if len(np.unique(window[:, 70])) == 1: 
            windows.append(window)
            #majority Voting or last frame for the label (col 73)
            if np.mean(window[:, 72]) > 0.5:
                label = 1  
            else:
                label = 0 
            labels.append(label)
            
    return np.array(windows), np.array(labels)

def get_joint_coords(row):
    """
    Extracts the 22 joints into an (22, 3) array.
    Columns 1-22: X, 23-44: Y, 45-66: Z.
    """
    #adjust for 0-indexing
    x = row[0:22]
    y = row[22:44]
    z = row[44:66]
    return np.column_stack((x, y, z)) #stack to get [[x1, y1, z1], [x2, y2, z2], ...]

def calculate_angle(joint_coords, idx_a, idx_vertex, idx_c):
    """
    Calculates the 3D angle at a vertex joint between two other joints.
    idx_a, idx_vertex, idx_c are the 0-based indices of the joints (0-21).
    """
    a = joint_coords[idx_a]
    v = joint_coords[idx_vertex]
    c = joint_coords[idx_c]
    #create vectors relative to the vertex
    ba = a - v
    bc = c - v
    #dot product to calculate cos(angle)
    cos_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc) + 1e-6)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle) #return angle in degrees

def extract_kinematic_features(window):
    #Convert window into (Time, 22, 3) coordinate tensor
    coords = np.array([get_joint_coords(row) for row in window])
    features = {}

    #Define angles based on Figure 1 mapping (format: { 'name': (point_a, vertex, point_b) })
    angle_map = {
        'trunk_flexion': (8, 7, 0),
        'left_knee': (1, 2, 3),
        'right_knee': (4, 5, 6),
        'left_hip': (0, 1, 2),
        'right_hip': (0, 4, 5),
        'neck_flexion': (8, 19, 20)
    }
    for name, (a, v, b) in angle_map.items():
        # Calculate angle for every frame in window
        angles = [calculate_angle(c, a, v, b) for c in coords]
        features[f'{name}_mean'] = np.mean(angles)
        features[f'{name}_std'] = np.std(angles) # Higher std = more movement fluidity

    #Velocity & Acceleration (using head and lumbar)
    for joint_idx, label in [(21, 'head'), (0, 'lumbar')]:
        pos = coords[:, joint_idx, :]
        vel = np.diff(pos, axis=0)
        vel_mag = np.linalg.norm(vel, axis=1)
        features[f'{label}_vel_avg'] = np.mean(vel_mag)
        accel = np.diff(vel, axis=0)
        accel_mag = np.linalg.norm(accel, axis=1)
        features[f'{label}_accel_max'] = np.max(accel_mag)

    #Posture: Trunk Length (Distance between Node 1 and Node 9) (consider normalising using total height if variance is too wide)
    #Detects support/bracing protective behaviour
    trunk_lengths = np.linalg.norm(coords[:, 8, :] - coords[:, 0, :], axis=1)
    features['trunk_length_mean'] = np.mean(trunk_lengths)

    return features

def extract_emg_features(window):
    #Columns 67-70 are indices 66-69 
    emg_data = window[:, 66:70] 
    rms = np.sqrt(np.mean(np.square(emg_data), axis=0)) #Standard RMS
    
    #Waveform Length - sum of absolute differences between consecutive samples
    wl = np.sum(np.abs(np.diff(emg_data, axis=0)), axis=0)
    return {
        'lumbar_R_rms': rms[0], 'lumbar_R_wl': wl[0],
        'lumbar_L_rms': rms[1], 'lumbar_L_wl': wl[1],
        'trapezius_R_rms': rms[2], 'trapezius_R_wl': wl[2],
        'trapezius_L_rms': rms[3], 'trapezius_L_wl': wl[3]
    }

if __name__ == "__main__": #main function just to test if everything is working
    import glob
    #test file loading
    print("Data Loading")
    train_files = glob.glob("train/*.mat")
    if not train_files:
        print("Error: No .mat files found in 'train/' folder.")
    else:
        sample_path = train_files[0]
        print(f"Loading sample file: {sample_path}")
        mat_data = loadmat(sample_path)
        matrix = mat_data.get('data') 
        print(f"Matrix shape: {matrix.shape} (N x 78 expected)") 

        #test feature extraction on first 180 frames
        print("\n---Feature Extraction---")
        if len(matrix) >= 180:
            test_window = matrix[0:180, :]
            
            #kinematic test
            k_feats = extract_kinematic_features(test_window)
            print(f"Kinematic Features ({len(k_feats)} extracted), printing first 5:")
            for key, val in list(k_feats.items())[:5]: #just printing first 5 for more concise test output
                print(f"  - {key}: {val:.4f}")
            
            #EMG test
            e_feats = extract_emg_features(test_window)
            print(f"\nEMG Features ({len(e_feats)} extracted):")
            for key, val in e_feats.items():
                print(f"  - {key}: {val:.4f}")
            
            #fusion check
            combined_vector = list(k_feats.values()) + list(e_feats.values())
            print(f"\nTotal Fused Feature Vector Length: {len(combined_vector)}")
            print("Successfully processed one window")
        else:
            print("File too short for full window.")