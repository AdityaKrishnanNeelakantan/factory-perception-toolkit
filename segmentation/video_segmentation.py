import cv2
from sam2_segmentation import segment_image


def segment_video_to_frames(video_path: str, sam_checkpoint: str, output_dir: str):
    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = f"{output_dir}/frame_{frame_idx:04d}.jpg"
        cv2.imwrite(frame_path, frame)
        segment_image(frame_path, sam_checkpoint, f"{output_dir}/seg_{frame_idx:04d}.jpg")

        frame_idx += 1

    cap.release()
    print("Finished processing video.")


if __name__ == "__main__":
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    video_path = "factory_sample.mp4"
    output_dir = "segmented_frames"

    segment_video_to_frames(video_path, sam_checkpoint, output_dir)
