import torch
import imageio
import numpy as np
import fiftyone as fo

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

def process_batch(batch, model, device):
    """
    Process a batch of video samples using the CoTracker model.
    
    Args:
        batch (list): List of FiftyOne samples
        model (torch.nn.Module): CoTracker model
        device (str): Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        tuple: Predicted tracks and visibility for the batch
    """
    # Concatenate video tensors from all samples in the batch
    videos = torch.cat([read_video_from_path(sample.filepath).to(device) for sample in batch])
    
    # Run inference without computing gradients
    with torch.no_grad():
        pred_tracks, pred_visibility = model(videos, grid_size=30)
    
    # Move results to CPU and convert to numpy arrays
    return pred_tracks.cpu().numpy(), pred_visibility.cpu().numpy()


def create_keypoints_batch(pred_tracks, pred_visibility, samples):
    """
    Create and add keypoints to a batch of samples.
    
    Args:
        pred_tracks (np.array): Predicted tracks for the batch
        pred_visibility (np.array): Predicted visibility for the batch
        samples (list): List of FiftyOne samples
    """
    for i, sample in enumerate(samples):
        height, width = sample.metadata.frame_height, sample.metadata.frame_width
        
        # Normalize coordinates to [0, 1] range
        pred_tracks[i, :, :, 0] /= width
        pred_tracks[i, :, :, 1] /= height
        
        # Dictionary to hold keypoints for all frames
        frames_keypoints = {}
        
        # Iterate through each frame in the video
        for frame_number in range(sample.metadata.total_frame_count):
            keypoints = []
            # Iterate through each tracked point
            for point_idx in range(pred_tracks.shape[2]):
                # Only add keypoint if it's visible in this frame
                if pred_visibility[i, frame_number, point_idx]:
                    # Extract x, y coordinates and convert to float
                    x, y = map(float, pred_tracks[i, frame_number, point_idx])
                    # Create a FiftyOne Keypoint object and add to list
                    keypoints.append(fo.Keypoint(points=[(x, y)], index=point_idx))
            
            # Add keypoints to the frame dictionary
            frames_keypoints[frame_number + 1] = fo.Keypoints(keypoints=keypoints)
        
        # Add all frames' keypoints to the sample at once
        sample.frames.merge({f: {"tracked_keypoints": kp} for f, kp in frames_keypoints.items()})
        
        # Save the updated sample
        sample.save()

def main(dataset, device='cuda', batch_size=4):
    """
    Main function to process the entire dataset.
    
    Args:
        dataset (fo.Dataset): FiftyOne dataset to process
        device (str): Device to run inference on ('cuda' or 'cpu')
        batch_size (int): Number of samples to process in each batch
    """
    # Load the CoTracker model
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline").to(device)
    model.eval()

    samples = list(dataset)
    for i in tqdm(range(0, len(samples), batch_size), desc="Processing batches"):
        batch = samples[i:i+batch_size]
        # Process the batch using the CoTracker model
        # Returns predicted tracks and visibility for all frames in the batch
        pred_tracks, pred_visibility = process_batch(batch, model, device)
        # Create and add keypoints to the samples in the current batch
        # This updates the FiftyOne dataset with the new keypoint information
        create_keypoints_batch(pred_tracks, pred_visibility, batch)

    # Reload the dataset to reflect changes
    dataset.reload()
    print("Processing complete. Updated dataset info:")
    print(dataset)
