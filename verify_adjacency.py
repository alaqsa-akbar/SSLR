import pickle
import numpy as np
import matplotlib.pyplot as plt
import torch

# Define the custom adjacency logic exactly as you requested
def get_edges():
    pairs = []
    
    # --- Right Hand (0-20) ---
    rh_fingers = [[1,2,3,4], [5,6,7,8], [9,10,11,12], [13,14,15,16], [17,18,19,20]]
    for finger in rh_fingers:
        pairs.append((0, finger[0]))
        for k in range(len(finger)-1):
            pairs.append((finger[k], finger[k+1]))

    # --- Left Hand (21-41) ---
    lh_fingers = [[22,23,24,25], [26,27,28,29], [30,31,32,33], [34,35,36,37], [38,39,40,41]]
    for finger in lh_fingers:
        pairs.append((21, finger[0]))
        for k in range(len(finger)-1):
            pairs.append((finger[k], finger[k+1]))

    # --- Face (42-69) ---
    lips = [42, 53, 54, 55, 56, 59, 58, 60, 57, 43, 48, 51, 49, 50, 47, 52, 46, 45, 44]
    for i in range(len(lips)-1): pairs.append((lips[i], lips[i+1]))
    pairs.append((lips[-1], lips[0]))
    
    # Face contours
    pairs.extend([(61,62), (62,63), (63,64), (61,65), (65,66), (66,67)])
    pairs.extend([(61,68), (61,69)])

    # --- Body Custom Logic ---
    # 70-71 connect to 72-73 (Shoulders)
    pairs.extend([(70, 72), (71, 73)])
    
    # Arms: 72-74-76 (Left?), 73-75-77 (Right?)
    pairs.extend([(72,74), (74,76)])
    pairs.extend([(73,75), (75,77)])
    pairs.append((72,73)) # Shoulders connection
    
    # Custom Loops on Arms
    # "Connect 77-79-81-83-77"
    pairs.extend([(77,79), (79,81), (81,83), (83,77)])
    # "Connect 76-78-80-82-76"
    pairs.extend([(76,78), (78,80), (80,82), (82,76)])
    
    # Hips/Shoulder connection
    pairs.append((84, 85))
    pairs.extend([(72, 84), (73, 85)])

    return pairs

def visualize_sample(pkl_file, sample_index=0):
    print(f"Loading {pkl_file}...")
    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get the first available sample
    sample_id = list(data.keys())[sample_index]
    pose = data[sample_id]['keypoints'] # Assuming (Frames, 86, 2)
    print(f"Visualizing Sample ID: {sample_id}, Shape: {pose.shape}")
    
    # Pick a frame where hands are likely visible (middle of sequence)
    frame_idx = pose.shape[0] // 2
    frame = pose[frame_idx] # (86, 2)
    
    edges = get_edges()
    
    plt.figure(figsize=(10, 10))
    plt.title(f"Adjacency Visualization (Sample {sample_id}, Frame {frame_idx})")
    
    # Plot Edges
    for i, j in edges:
        if i < 86 and j < 86:
            # Check if points are not invisible (assuming -1 or 0 indicates invisible)
            # Adjust the threshold as needed for your specific data format
            if not (frame[i,0] == 0 and frame[i,1] == 0) and \
               not (frame[j,0] == 0 and frame[j,1] == 0):
                x_vals = [frame[i, 0], frame[j, 0]]
                # Invert Y because images are usually top-down
                y_vals = [-frame[i, 1], -frame[j, 1]] 
                plt.plot(x_vals, y_vals, 'b-', alpha=0.6, linewidth=1)

    # Plot Nodes
    # Red for Right Hand, Green for Left Hand, Blue for Body
    colors = []
    for k in range(86):
        if k <= 20: colors.append('red')
        elif k <= 41: colors.append('green')
        else: colors.append('blue')
        
    plt.scatter(frame[:, 0], -frame[:, 1], c=colors, s=20)
    
    # Annotate specific points to debug
    special_points = [70, 71, 72, 73, 76, 77, 79, 83]
    for p in special_points:
        plt.text(frame[p,0], -frame[p,1], str(p), fontsize=12, fontweight='bold')

    plt.axis('equal')
    plt.tight_layout()
    plt.savefig("adjacency_viz.png")
    print("Saved visualization to adjacency_viz.png")

if __name__ == "__main__":
    visualize_sample("data/data.pkl", sample_index=0)