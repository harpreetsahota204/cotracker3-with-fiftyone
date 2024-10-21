#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Function to get video duration in seconds
get_duration() {
    ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$1"
}

# Function to process a single video
process_video() {
    local input_file="$1"
    local output_file="${input_file%.*}_processed.mp4"
    local duration

    duration=$(get_duration "$input_file")

    echo "Processing $input_file (duration: $duration seconds)"

    if (( $(echo "$duration > 10" | bc -l) )); then
        echo "Video longer than 10 seconds. Sampling every third frame and reducing to 7 fps."
        ffmpeg -hide_banner -loglevel error -i "$input_file" \
            -vf "select='not(mod(n,3))',fps=7" \
            -af "aselect='not(mod(n,3))',asetpts=N/SR/TB" \
            -y "$output_file"
    else
        echo "Video shorter than or equal to 10 seconds. Only reducing to 10 fps."
        ffmpeg -hide_banner -loglevel error -i "$input_file" \
            -vf "fps=10" \
            -y "$output_file"
    fi

    if [ $? -eq 0 ]; then
        echo "Processed video saved as $output_file"
        echo "Deleting original file: $input_file"
        rm "$input_file"
    else
        echo "Error processing $input_file. Original file not deleted."
    fi

    echo "--------------------"
}

# Check if a directory is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory>"
    exit 1
fi

directory="$1"

# Check if the provided path is a directory
if [ ! -d "$directory" ]; then
    echo "Error: '$directory' is not a valid directory"
    exit 1
fi

# Enable extended globbing for better pattern matching
shopt -s nullglob

# Find all video files (case-insensitive)
video_files=("$directory"/*.{mp4,MP4,avi,AVI,mov,MOV,mkv,MKV,flv,FLV,webm,WEBM})

shopt -u nullglob

if [ ${#video_files[@]} -eq 0 ]; then
    echo "No video files found in '$directory'"
    exit 1
fi

# Number of parallel jobs (set to number of CPU cores)
num_cores=$(nproc)
echo "Starting processing with $num_cores parallel jobs..."

export -f process_video
export -f get_duration

# Use GNU Parallel for parallel execution
# Install GNU Parallel if not already installed:
# sudo apt-get install parallel
parallel --jobs "$num_cores" process_video ::: "${video_files[@]}"

echo "All videos in '$directory' have been processed."