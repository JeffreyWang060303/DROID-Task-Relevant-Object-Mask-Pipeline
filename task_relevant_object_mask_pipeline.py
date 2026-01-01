import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow_datasets as tfds
import tensorflow as tf
import torch
import cv2
from PIL import Image
import spacy
from tqdm import tqdm

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# ===================== CONFIG =====================

DROID_DIR = "/viscam/data/DROID/droid/1.0.1"
DATASET_NAME = "droid_101"

OUT_ROOT = Path("outputs/objmask")
FPS = 15

OBSERVATIONS = [
    "exterior_image_1_left",
    "exterior_image_2_left",
    "wrist_image_left",
]

DEFAULT_COLORS = [
    (160, 32, 240),
    (255, 0, 0),
    (255, 255, 0),
    (0, 0, 255),
]

# Disable TF GPU (critical)
tf.config.set_visible_devices([], "GPU")

# ===================== NLP =====================

nlp = spacy.load("en_core_web_sm")

def extract_objects(instruction: str):
    doc = nlp(instruction.lower())
    objs = []
    for chunk in doc.noun_chunks:
        txt = chunk.text.strip()
        if txt in ["a", "the", "it", "they", "them", "this", "that"]:
            continue
        if txt not in objs:
            objs.append(txt)
    return objs

# ===================== SAM3 =====================

def load_sam3():
    model = build_sam3_image_model()
    model.to("cuda")
    return Sam3Processor(model)

def best_mask_from_prompt(processor, state, prompt):
    with torch.inference_mode():
        out = processor.set_text_prompt(state=state, prompt=prompt)

    masks, scores = out["masks"], out["scores"]
    if len(masks) == 0:
        return None

    idx = torch.argmax(scores).item()
    mask = masks[idx]
    if mask.ndim == 2:
        mask = mask.unsqueeze(0)
    return mask

def overlay_masks(image, masks, colors, alpha=80):
    image = image.convert("RGBA")
    masks = masks.cpu().numpy().astype(np.uint8)

    for mask, color in zip(masks, colors):
        m = Image.fromarray(mask * 255, mode="L")
        overlay = Image.new("RGBA", image.size, color + (0,))
        overlay.putalpha(m.point(lambda v: alpha if v > 0 else 0))
        image = Image.alpha_composite(image, overlay)

    return image

# ===================== EPISODE =====================

def episode_id(idx):
    return f"{DATASET_NAME}_ep_{idx:06d}"

def extract_instruction(ep):
    for step in ep["steps"]:
        return step["language_instruction_3"].numpy().decode("utf-8")
    return ""

def process_episode(ep, global_idx, processor):
    ep_id = episode_id(global_idx)
    out_dir = OUT_ROOT / DATASET_NAME / f"E{ep_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    instruction = extract_instruction(ep)
    objects = extract_objects(instruction)

    # ---- Print instruction + objects (like notebook) ----
    print("\n==============================")
    print(f"Episode {ep_id}")
    print("Instruction:", instruction)
    print("Objects:", objects)
    print("==============================\n")

    writers_mask = {}
    writers_orig = {}

    meta = {
        "episode_id": ep_id,
        "instruction": instruction,
        "objects": objects,
        "cameras": {},
    }

    def get_writers(cam, h, w):
        if cam not in writers_mask:
            cam_dir = out_dir / cam
            cam_dir.mkdir(exist_ok=True)

            orig_path = cam_dir / "original.mp4"
            mask_path = cam_dir / "objmask.mp4"

            writers_orig[cam] = cv2.VideoWriter(
                str(orig_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                FPS,
                (w, h),
            )

            writers_mask[cam] = cv2.VideoWriter(
                str(mask_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                FPS,
                (w, h),
            )

            meta["cameras"][cam] = {
                "original_video": str(orig_path.relative_to(out_dir)),
                "masked_video": str(mask_path.relative_to(out_dir)),
            }

        return writers_orig[cam], writers_mask[cam]

    # ---- Main loop (streaming, no frame buffering) ----
    for step in ep["steps"]:
        for cam in OBSERVATIONS:
            frame = step["observation"][cam].numpy()
            pil = Image.fromarray(frame)

            state = processor.set_image(pil)

            all_masks = []
            colors = []

            for i, obj in enumerate(objects):
                m = best_mask_from_prompt(processor, state, obj)
                if m is not None:
                    all_masks.append(m)
                    colors.append(DEFAULT_COLORS[i % len(DEFAULT_COLORS)])

            if all_masks:
                masks = torch.cat(all_masks, dim=0)
                out_img = overlay_masks(pil, masks, colors)
            else:
                out_img = pil.convert("RGBA")

            out_np = np.array(out_img.convert("RGB"))
            h, w = out_np.shape[:2]

            orig_writer, mask_writer = get_writers(cam, h, w)

            # write original frame
            orig_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

            # write masked frame
            mask_writer.write(cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR))

    # ---- Release writers ----
    for w in writers_orig.values():
        w.release()

    for w in writers_mask.values():
        w.release()

    # ---- Save metadata ----
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)


# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--count", type=int, default=1)
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    builder = tfds.builder_from_directory(DROID_DIR)
    ds = builder.as_dataset(
        split="train",
        read_config=tfds.ReadConfig(try_autocache=False),
    )

    ds = ds.skip(args.offset).take(args.count)

    processor = load_sam3()

    for i, ep in enumerate(ds):
        process_episode(ep, args.offset + i, processor)

    print("All done.")

if __name__ == "__main__":
    main()
