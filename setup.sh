# conda environment
conda create -n roca python=3.8 -y
conda activate roca

# Simple deps
pip install -r requirements.txt

# pytorch3d
conda install pytorch==1.8.0 torchvision==0.9.0 torchaudio==0.8.0 cudatoolkit=11.1 -c pytorch -c conda-forge -y
conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
# NOTE: Tested with 0.5 and 0.6
conda install pytorch3d -c pytorch3d-nightly -y

# detectron2
python -m pip install detectron2==0.4+cu111 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.8/index.html -y

# numba
pip install numba
