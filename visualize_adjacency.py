import pickle
import plotly.graph_objects as go
import numpy as np
import os

def get_adjacency_pairs():
    pairs = []
    
    # --- Right Hand (0-20) ---
    # Wrist: 0
    # Thumb: 0->1->2->3->4
    pairs.extend([(0,1), (1,2), (2,3), (3,4)])
    # Index: 0->5->6->7->8
    pairs.extend([(0,5), (5,6), (6,7), (7,8)])
    # Middle: 0->9->10->11->12
    pairs.extend([(0,9), (9,10), (10,11), (11,12)])
    # Ring: 0->13->14->15->16
    pairs.extend([(0,13), (13,14), (14,15), (15,16)])
    # Pinky: 0->17->18->19->20
    pairs.extend([(0,17), (17,18), (18,19), (19,20)])
    
    # --- Left Hand (21-41) ---
    # Wrist: 21
    # Thumb: 21->22->23->24->25
    pairs.extend([(21,22), (22,23), (23,24), (24,25)])
    # Index: 21->26->27->28->29
    pairs.extend([(21,26), (26,27), (27,28), (28,29)])
    # Middle: 21->30->31->32->33
    pairs.extend([(21,30), (30,31), (31,32), (32,33)])
    # Ring: 21->34->35->36->37
    pairs.extend([(21,34), (34,35), (35,36), (36,37)])
    # Pinky: 21->38->39->40->41
    pairs.extend([(21,38), (38,39), (39,40), (40,41)])
    
    # --- Body (72-85) ---
    # Shoulders: 72-73
    pairs.append((72, 73))
    # Left Arm: 72 (Shoulder) -> 74 (Elbow) -> 76 (Wrist)
    pairs.extend([(72, 74), (74, 76)])
    # Right Arm: 73 (Shoulder) -> 75 (Elbow) -> 77 (Wrist)
    pairs.extend([(73, 75), (75, 77)])
    # Torso: 72->84 (L Shoulder -> L Hip), 73->85 (R Shoulder -> R Hip)
    pairs.extend([(72, 84), (73, 85)])
    # Hips: 84-85
    pairs.append((84, 85))
    
    # --- Connecting Body to Hands ---
    # Left Wrist (76) to Left Hand Root (21)
    pairs.append((76, 21))
    # Right Wrist (77) to Right Hand Root (0)
    pairs.append((77, 0))
    
    # --- Face (42-71) ---
    # Nose (61) to Eyes/Eyebrows?
    # 61 -> 62 (Left Eye Inner?), 61 -> 65 (Right Eye Inner?)
    pairs.extend([(61, 62), (61, 65)])
    # Eyebrows/Eyes: 62-63-64, 65-66-67
    pairs.extend([(62, 63), (63, 64)])
    pairs.extend([(65, 66), (66, 67)])
    
    # Lips (User provided clockwise order)
    lips_indices = [42, 53, 54, 55, 56, 59, 58, 60, 57, 43, 48, 51, 49, 50, 47, 52, 46, 45, 44]
    # Connect them in a loop
    for i in range(len(lips_indices) - 1):
        pairs.append((lips_indices[i], lips_indices[i+1]))
    pairs.append((lips_indices[-1], lips_indices[0])) # Close loop
    
    # Mouth Corners (70, 71) to Lips?
    # Maybe 70 is near 42? 71 near something else?
    # Let's just connect corners to nose for context
    pairs.extend([(61, 70), (61, 71)])
    
    # Cheekbones (68, 69) to Nose
    pairs.extend([(61, 68), (61, 69)])
    
    return pairs

def visualize_adjacency(pkl_file, output_file="visualizations/adjacency_visualization.html"):
    if not os.path.exists(pkl_file):
        print(f"File {pkl_file} not found.")
        return

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Visualize multiple samples to check consistency
    sample_ids = list(data.keys())[:3] # Check first 3 samples
    
    fig = go.Figure()
    pairs = get_adjacency_pairs()
    
    for i, sample_id in enumerate(sample_ids):
        keypoints = data[sample_id]['keypoints']
        frame_idx = keypoints.shape[0] // 2
        frame_points = keypoints[frame_idx]
        
        # Offset samples so they don't overlap
        offset_x = i * 1500 
        
        # Plot Points
        fig.add_trace(go.Scatter(
            x=frame_points[:, 0] + offset_x,
            y=frame_points[:, 1],
            mode='markers+text',
            marker=dict(size=5, color='red'),
            text=[str(j) for j in range(86)],
            textposition="top center",
            hoverinfo='text',
            name=f'Sample {sample_id}'
        ))
        
        # Plot Connections
        edge_x = []
        edge_y = []
        for p1, p2 in pairs:
            x0, y0 = frame_points[p1]
            x1, y1 = frame_points[p2]
            edge_x.extend([x0 + offset_x, x1 + offset_x, None])
            edge_y.extend([y0, y1, None])
            
        fig.add_trace(go.Scatter(
            x=edge_x,
            y=edge_y,
            mode='lines',
            line=dict(width=1, color='blue'),
            hoverinfo='none',
            showlegend=False
        ))
    
    fig.update_layout(
        title=f"Adjacency Matrix Visualization (Samples {sample_ids})",
        xaxis_title="X",
        yaxis_title="Y",
        width=1500,
        height=1000,
        yaxis=dict(autorange="reversed")
    )
    
    fig.write_html(output_file)
    print(f"Saved adjacency visualization to {output_file}")

if __name__ == "__main__":
    visualize_adjacency("data/data.pkl")
