"""
CoTracker Keypoint Extraction for FiftyOne Datasets

This script processes video samples in a FiftyOne dataset using the CoTracker model
to extract keypoints for each frame. The extracted keypoints are then added to the
dataset as a new field for each frame.

Usage:
1. Ensure you have the required libraries installed: torch, imageio, numpy, fiftyone
2. Make sure your FiftyOne dataset is properly set up with video samples
3. Run the script with your dataset:

   import fiftyone as fo
   dataset = fo.load_dataset("your_dataset_name")
   main(dataset)

4. After running, the dataset will be updated with new 'tracked_keypoints' fields
   for each frame in each video sample.

Note: This script requires a CUDA-capable GPU for optimal performance.
"""

import torch
import imageio
import numpy as np
import fiftyone as fo
from tqdm import tqdm 

def read_video_from_path(path):
    """
    Read a video file and convert it to a tensor.
    
    Args:
        path (str): Path to the video file
    
    Returns:
        torch.Tensor: Video tensor of shape (1, num_frames, 3, height, width)
    """
    reader = imageio.get_reader(path)
    frames = [np.array(im) for im in reader]
    # Stack frames, convert to torch tensor, and rearrange dimensions
    return torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2)[None].float()

def process_batch(batch, model, device, grid_size=30):
    """
    Process a batch of video samples using the CoTracker model.
    
    Args:
        batch (list): List of FiftyOne samples
        model (torch.nn.Module): CoTracker model
        device (str): Device to run inference on ('cuda' or 'cpu')
        grid_size (int): Grid size for CoTracker (default: 30)
    
    Returns:
        list: List of tuples containing predicted tracks and visibility for each sample
    """
    results = []
    for sample in batch:
        video = read_video_from_path(sample.filepath).to(device)
        
        # Run inference without computing gradients
        with torch.no_grad():
            pred_tracks, pred_visibility = model(video, queries=None, grid_size=grid_size)
        
        # Move results to CPU and convert to numpy arrays
        results.append((pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy()))

    torch.cuda.empty_cache()

    return results


def create_keypoints_batch(results, samples):
    """
    Create and add keypoints to a batch of samples.
    
    Args:
        results (list): List of tuples containing predicted tracks and visibility for each sample
        samples (list): List of FiftyOne samples
    """
    for (pred_tracks, pred_visibility), sample in zip(results, samples):
        # Extract frame dimensions from sample metadata
        height, width = sample.metadata.frame_height, sample.metadata.frame_width
        
        # Remove the batch dimension (which is now always 1) and ensure we're working with numpy arrays
        if isinstance(pred_tracks, torch.Tensor):
            pred_tracks = pred_tracks.squeeze(0).cpu().numpy()
        else:
            pred_tracks = np.squeeze(pred_tracks, axis=0)
        
        if isinstance(pred_visibility, torch.Tensor):
            pred_visibility = pred_visibility.squeeze(0).cpu().numpy()
        else:
            pred_visibility = np.squeeze(pred_visibility, axis=0)
        
        # Normalize coordinates to [0, 1] range
        pred_tracks[:, :, 0] /= width
        pred_tracks[:, :, 1] /= height
        
        # Pre-compute frame range
        # Add 1 to total_frame_count because range is exclusive of the last number
        frame_range = range(1, sample.metadata.total_frame_count + 1)
        
        frames_keypoints = {
            frame_number: fo.Keypoints(keypoints=[
                # Create a Keypoint object for each visible point
                fo.Keypoint(points=[(float(x), float(y))], index=point_idx)
                for point_idx, (x, y) in enumerate(pred_tracks[frame_number - 1])
                # Only include keypoints that are visible in this frame
                if pred_visibility[frame_number - 1, point_idx]
            ])
            for frame_number in frame_range
        }
        
        # Add all frames' keypoints to the sample at once
        # This creates a dictionary where keys are frame numbers and values are
        # dictionaries containing the tracked_keypoints
        sample.frames.merge({f: {"tracked_keypoints": kp} for f, kp in frames_keypoints.items()})
        
        # Save the updated sample
        sample.save()

def main(dataset, device='cuda', batch_size=4, grid_size=30):
    """
    Main function to process the entire dataset.
    
    Args:
        dataset (fo.Dataset): FiftyOne dataset to process
        device (str): Device to run inference on ('cuda' or 'cpu')
        batch_size (int): Number of samples to process in each batch
        grid_size (int): Grid size for CoTracker (default: 30)
    """
    # Load the CoTracker model
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()

    samples = list(dataset)
    for i in tqdm(range(0, len(samples), batch_size), desc="Processing batches"):
        batch = samples[i:i+batch_size]
        # Process the batch using the CoTracker model
        # Returns predicted tracks and visibility for all frames in the batch
        results = process_batch(batch, model, device, grid_size)
        # Create and add keypoints to the samples in the current batch
        # This updates the FiftyOne dataset with the new keypoint information
        create_keypoints_batch(results, batch)

    torch.cuda.empty_cache()
    # Reload the dataset to reflect changes
    dataset.reload()
    print("Processing complete. Updated dataset info:")
    print(dataset)