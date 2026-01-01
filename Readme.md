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
'''

Required Dependencies

The pipeline depends on the following packages:

sam3

torch

tensorflow_datasets

opencv-python

spaCy + en_core_web_sm

If the spaCy language model is missing, install it with:

```bash
python -m spacy download en_core_web_sm
