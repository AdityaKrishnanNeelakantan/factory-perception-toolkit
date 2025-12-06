import cv2
import torch
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
from factory_noise import factory_augment


def load_sam_model(
    sam_checkpoint: str,
    model_type: str = "vit_h"
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator


def segment_image(image_path: str, sam_checkpoint: str, output_path: str):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # apply factory-style augmentations
    aug_image = factory_augment(image)

    mask_generator = load_sam_model(sam_checkpoint)
    masks = mask_generator.generate(aug_image)

    # simple visualization: draw masks as colored regions
    overlay = aug_image.copy()
    for m in masks:
        color = (0, 255, 0)
        overlay[m["segmentation"]] = color

    overlay = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, overlay)
    print(f"Saved segmented image to {output_path}")


if __name__ == "__main__":
    # TODO: update these paths for your machine
    sam_checkpoint = "sam_vit_h_4b8939.pth"
    input_image = "sample_frame.jpg"
    output_image = "sample_frame_segmented.jpg"

    segment_image(input_image, sam_checkpoint, output_image)
