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

This pipeline is designed to scale safely on the Stanford **viscam** cluster using **Slurm**.


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
Required dependencies include:

sam3

torch

tensorflow_datasets

opencv-python

spaCy + en_core_web_sm

If the spaCy model is missing, install it with:

bash
Copy code
python -m spacy download en_core_web_sm
Usage
1. Single-Episode Test (Debug / Interactive)
Run one episode locally (on a compute node with GPU):

bash
Copy code
python task_relevant_object_mask_pipeline.py --offset 0 --count 1
This processes one episode, starting from the specified offset.

2. Batch Processing with Slurm (Recommended)
Submit multiple experiments in parallel using Slurm:

bash
Copy code
sbatch run_objmask_5tasks.sbatch
This launches 5 GPU jobs, each processing a different episode.

Monitor job status with:

bash
Copy code
squeue -u $USER
Output Structure
Outputs are written under the configured output root (e.g. outputs/).

For N experiments, the directory structure is:

text
Copy code
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
info.md (Per Experiment)
Each experiment folder contains a metadata file with:

text
Copy code
instruction:
"Pick up the blue ring from the table and put it in the wooden tray"

objects:
- blue ring
- table
- wooden tray
This makes each experiment self-contained and easy to analyze or reuse later.
