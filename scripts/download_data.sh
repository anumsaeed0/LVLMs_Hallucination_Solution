#!/usr/bin/env bash
# Benchmark data setup. Run from the repo root. ~15GB total.
set -e
mkdir -p data/coco data/pope data/textvqa results

# --- COCO val2014 (POPE + CHAIR images) ---
wget -c http://images.cocodataset.org/zips/val2014.zip -P data/coco
unzip -qn data/coco/val2014.zip -d data/coco

# --- COCO annotations (CHAIR ground-truth objects) ---
wget -c http://images.cocodataset.org/annotations/annotations_trainval2014.zip -P data/coco
unzip -qn data/coco/annotations_trainval2014.zip -d data/coco

# --- POPE question files ---
for split in random popular adversarial; do
  wget -c "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_${split}.json" -P data/pope
done

# --- TextVQA (val questions + train_val images) ---
wget -c https://dl.fbaipublicfiles.com/textvqa/data/TextVQA_0.5.1_val.json -P data/textvqa
wget -c https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip -P data/textvqa
unzip -qn data/textvqa/train_val_images.zip -d data/textvqa   # -> train_images/

# --- A-OKVQA: no download needed (HF hub, images embedded) ---
echo "A-OKVQA loads automatically via: datasets.load_dataset('HuggingFaceM4/A-OKVQA')"
echo "Done."
