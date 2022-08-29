# System config
export OMP_NUM_THREADS=1
export NUM_WORKERS=4
export SEED=2021

# NOTE: Change the data config based on your detup!
# JSON files
export DATA_DIR=../Dataset/Dataset/
# Resized images with intrinsics and poses
export IMAGE_ROOT=../Dataset/Images/
# Depths and instances rendered over images
export RENDERING_ROOT=../Dataset/Rendering/
# Scan2CAD Full Annotations
export FULL_ANNOT=../Dataset/full_annotations.json

# Model configurations
export RETRIEVAL_MODE=resnet_resnet+image+comp
export E2E=1
export NOC_WEIGHTS=1

# Train and test behavior
export EVAL_ONLY=1
export CHECKPOINT=../Models/model_best1.pth  # "none"
export RESUME=0  # This means from last checkpoint
export OUTPUT_DIR=output
