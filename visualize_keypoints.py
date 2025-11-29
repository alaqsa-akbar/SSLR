import pickle
import plotly.graph_objects as go
import numpy as np
import os

def visualize_sample_interactive(pkl_file, output_file="visualizations/keypoint_visualization.html"):
    if not os.path.exists(pkl_file):
        print(f"File {pkl_file} not found.")
        return

    with open(pkl_file, 'rb') as f:
        data = pickle.load(f)
    
    # Get the first sample
    sample_id = list(data.keys())[0]
    keypoints = data[sample_id]['keypoints'] # (Frames, 86, 2)
    
    # Take the middle frame
    frame_idx = keypoints.shape[0] // 2
    frame_points = keypoints[frame_idx]
    
    # Create interactive plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=frame_points[:, 0],
        y=frame_points[:, 1],
        mode='markers+text',
        marker=dict(size=10, color=np.arange(86), colorscale='Viridis'),
        text=[str(i) for i in range(86)],
        textposition="top center",
        hoverinfo='text'
    ))
    
    fig.update_layout(
        title=f"Interactive Keypoint Visualization (Sample {sample_id}, Frame {frame_idx})",
        xaxis_title="X",
        yaxis_title="Y",
        width=1000,
        height=1000,
        yaxis=dict(autorange="reversed") # Image coordinates
    )
    
    fig.write_html(output_file)
    print(f"Saved interactive visualization to {output_file}")

if __name__ == "__main__":
    visualize_sample_interactive("data/data.pkl")
