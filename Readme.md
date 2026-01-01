# **Task-Relevant Object Mask Generation (SAM3 + spaCy)**

This directory contains scripts for generating **task-relevant object segmentation videos** from the **DROID dataset**.

Unlike robot-only masking, this pipeline:
- Parses the **language instruction**
- Extracts **task-relevant objects** using NLP
- Uses **SAM3** to segment those objects
- Outputs **masked videos + original videos**, organized per experiment

## **Overview**

**Goal:**  
Given one episode from the DROID RLDS dataset, automatically:

1. Read the **language instruction**
2. Extract **object phrases** (e.g. *blue ring*, *wooden tray*)
3. Segment those objects in RGB videos using **SAM3**
4. Save both **original** and **masked** videos for each camera view

## **Models Used**

### **Segmentation**
- **SAM3 (Segment Anything Model v3)**  
  Used for text-prompted image segmentation on RGB frames.

### **Language Processing**
- **spaCy (`en_core_web_sm`)**  
  Used to extract **noun phrases** from task instructions to form object prompts.


## **Files**

- **`task_relevant_object_mask_pipeline.py`**  
  Main processing script that:
  - Loads the DROID dataset (RLDS format)
  - Extracts language instructions
  - Parses task-relevant objects with spaCy
  - Runs SAM3 inference per frame
  - Saves **original + masked videos**
  - Writes instruction/object metadata

- **`run_objmask_5tasks.sbatch`**  
  Example Slurm batch script that:
  - Launches multiple GPU jobs in parallel
  - Each task processes one episode
  - Fully compliant with **viscam** cluster usage rules

---

## **Environment Setup**

Activate the shared **`sam3`** conda environment:

```bash
source /vision/u/yinhang/miniconda3/etc/profile.d/conda.sh
conda activate sam3
````

## **Usage**

1. Single-Episode Test (Debug / Interactive)

Run one episode locally (on a compute node with GPU):

```bash
python task_relevant_object_mask_pipeline.py --offset 0 --count 1
````

This processes one episode, starting from the specified offset.

2. Batch Processing with Slurm (Recommended)

Submit multiple experiments in parallel using Slurm:

```bash
sbatch run_objmask_5tasks.sbatch
````

## **Output Structure**

Outputs are written under the configured output root (e.g. outputs/).

For N experiments, the directory structure is:

```bash
outputs/
└── dataset_name/
    ├── E1/
    │   ├── exterior_image_1_left/
    │   │   ├── original.mp4
    │   │   └── masked.mp4
    │   ├── exterior_image_2_left/
    │   │   ├── original.mp4
    │   │   └── masked.mp4
    │   ├── wrist_image_left/
    │   │   ├── original.mp4
    │   │   └── masked.mp4
    │   └── info.md
    ├── E2/
    ├── E3/
    ├── E4/
    └── E5/
````

## ** meta.json (Per Experiment) **

Each experiment folder contains a metadata file with the following content:

instruction:
"Pick up the blue ring from the table and put it in the wooden tray"

objects:
- blue ring
- table
- wooden tray
