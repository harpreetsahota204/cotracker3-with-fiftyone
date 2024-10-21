import os
from typing import List, Dict, Optional, Union
import re
import torch
import numpy as np
from torch import Tensor
from concurrent.futures import ThreadPoolExecutor
import imageio

def read_video_from_path(path: str) -> Optional[Tensor]:
    """
    Read a video file and convert it to a 5D tensor.

    :param path: Path to the video file.
    :return: A 5D tensor with shape (1, num_frames, channels, height, width) or None if there's an error.
    """
    try:
        reader = imageio.get_reader(path)
        frames = [np.array(im) for im in reader]
        stacks = np.stack(frames)
        return torch.from_numpy(stacks).permute(0, 3, 1, 2)[None].float()
    except Exception as e:
        print(f"Error reading video file {path}: {e}")
        return None

def list_numbered_mp4_files(directory: str) -> List[str]:
    """
    List all .mp4 files in the given directory .

    :param directory: The directory to search for files.
    :return: A list of paths to .mp4 files .
    """
    return [os.path.join(directory, f) for f in os.listdir(directory)
            if f.endswith('.mp4')]

def process_video(video_path: str) -> Optional[Tensor]:
    """
    Process a single video file and return it as a tensor.

    :param video_path: Path to the video file.
    :return: Tensor representing the video with shape (1, num_frames, channels, height, width) or None if there's an error.
    """
    return read_video_from_path(video_path)

def convert_videos_to_tensors(directory: str) -> Dict[str, Tensor]:
    """
    Convert all numbered .mp4 files in the directory to tensors for each video.

    :param directory: The directory containing the .mp4 files.
    :return: A dictionary with file paths as keys and tensors as values.
    """
    mp4_files = list_numbered_mp4_files(directory)
    
    with ThreadPoolExecutor() as executor:
        results = executor.map(process_video, mp4_files)
    
    return {path: tensor for path, tensor in zip(mp4_files, results) if tensor is not None}

def calculate_tensor_memory_usage(tensor: Tensor) -> float:
    """
    Calculate the memory usage of a single tensor.

    :param tensor: A PyTorch tensor.
    :return: Memory usage in megabytes.
    """
    if tensor.is_cuda:
        return torch.cuda.memory_allocated(tensor.device) / (1024 ** 2)
    else:
        return tensor.element_size() * tensor.numel() / (1024 ** 2)

def calculate_memory_usage(input_data: Union[Dict[str, Tensor], Tensor]) -> Union[Dict[str, float], float]:
    """
    Calculate the memory usage for each video in the dictionary or for a single tensor.

    :param input_data: Either a dictionary with file paths as keys and tensors as values,
                       or a single tensor.
    :return: If input is a dictionary, returns a dictionary with file paths as keys 
             and memory usage in megabytes as values.
             If input is a tensor, returns the memory usage of that tensor in megabytes.
    """
    if isinstance(input_data, Tensor):
        memory_usage = calculate_tensor_memory_usage(input_data)
        print(f"Tensor Shape: {input_data.shape}, Memory usage: {memory_usage:.2f} MB")
        return memory_usage
    
    elif isinstance(input_data, dict):
        memory_usage_dict = {}
        for video_path, tensor in input_data.items():
            memory_usage = calculate_tensor_memory_usage(tensor)
            memory_usage_dict[video_path] = memory_usage
            print(f"Video: {video_path}, Shape: {tensor.shape}, Memory usage: {memory_usage:.2f} MB")
        return memory_usage_dict
    
    else:
        raise ValueError("Input must be either a PyTorch tensor or a dictionary of tensors.")
    
def calculate_model_memory_usage(model):
    # Calculate the total memory used by the model's parameters
    total_memory = 0
    for param in model.parameters():
        # Check if the parameter is on GPU and get its memory usage accordingly
        if param.is_cuda:
            total_memory += param.element_size() * param.numel()
        else:
            total_memory += param.element_size() * param.numel()

    # Convert to megabytes
    total_memory_mb = total_memory / (1024 ** 2)

    print(f"Model memory usage: {total_memory} bytes")
    print(f"Model memory usage: {total_memory_mb:.2f} MB")
