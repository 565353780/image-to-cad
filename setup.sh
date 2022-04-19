# conda environment
conda create -n roca python=3.8 -y
conda activate roca

# Simple deps
pip install dataclasses opencv-python numpy-quaternion pandas scipy trimesh rtree numba

pip install open3d==0.13.0

# pytorch3d
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
# NOTE: Tested with 0.5 and 0.6
conda install pytorch3d -c pytorch3d-nightly -y

pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# detectron2
python -m pip install detectron2==0.3+cu110 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
