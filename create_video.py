import cv2
import os
import re

def create_video_from_frames(image_folder, output_video_name, fps):
    """
    Creates a video from a folder of sorted image frames.

    Args:
        image_folder (str): Path to the folder containing the images.
        output_video_name (str): The name of the output video file (e.g., 'output.mp4').
        fps (int): Frames per second for the output video.
    """
    
    print("Searching for frames in:", os.path.abspath(image_folder))
    
    # Get all image files from the folder
    files = [f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
    
    if not files:
        print("Error: No image files found in the specified folder.")
        print("Please ensure you have run the tracking script first and that frames were saved.")
        return

    # --- Sorting the files numerically ---
    # This is crucial to ensure frames are in the correct order (e.g., frame_10 after frame_9)
    def sort_key(f):
        # Extracts numbers from the filename for proper sorting
        return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', f)]
    
    files.sort(key=sort_key)
    
    print(f"Found {len(files)} frames. Starting video creation...")

    # Read the first image to get the frame size (width, height)
    first_image_path = os.path.join(image_folder, files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error: Could not read the first image at {first_image_path}")
        return
        
    height, width, layers = frame.shape
    frame_size = (width, height)

    # --- Initialize the VideoWriter object ---
    # The 'mp4v' is a FourCC code for the MPEG-4 video codec.
    # It's a widely compatible choice for creating .mp4 files.
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    video_writer = cv2.VideoWriter(output_video_name, fourcc, fps, frame_size)

    # Loop through all the sorted image files
    for file in files:
        image_path = os.path.join(image_folder, file)
        # Read the image
        img = cv2.imread(image_path)
        if img is not None:
            # Write the image frame to the video
            video_writer.write(img)
        else:
            print(f"Warning: Could not read frame {file}. Skipping.")
    
    # Release the VideoWriter object to save the file
    video_writer.release()
    print(f"Success! Video has been saved as '{output_video_name}'")


if __name__ == '__main__':
    # --- CONFIGURATION ---
    # You can change these values
    
    IMAGE_DIRECTORY = "modified_images"
    OUTPUT_VIDEO_NAME = "crystal_tracking_output.mp4"
    FRAMES_PER_SECOND = 60  # Adjust for faster or slower playback

    create_video_from_frames(IMAGE_DIRECTORY, OUTPUT_VIDEO_NAME, FRAMES_PER_SECOND)