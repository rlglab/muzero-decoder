import os
import cv2


def images_to_video(image_folder, output_video, fps):
    # Get list of images
    images = [img for img in os.listdir(image_folder) if img.endswith(".png")]
    # images.sort()  # Sort the images by name to maintain the order

    if not images:
        print("No PNG images found in the folder.")
        return

    # Read the first image to get the dimensions
    first_image_path = os.path.join(image_folder, images[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Error reading the first image: {first_image_path}")
        return

    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4
    video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    if not video.isOpened():
        print("Error: VideoWriter not opened.")
        return

    for image in images:
        image_path = os.path.join(image_folder, image)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"Error reading image: {image_path}")
            continue  # Skip this image and continue with the next

        video.write(frame)  # Write the frame to the video

    video.release()  # Release the video writer
    print(f"Video saved as {output_video}")


if __name__ == "__main__":
    image_folder = 'pacman-start_0-end_1209/all'  # Folder containing .png images
    output_video = 'pacman.mp4'  # Output video file
    fps = 15  # Frames per second

    images_to_video(image_folder, output_video, fps)
