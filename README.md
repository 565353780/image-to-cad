# ROCA: Robust CAD Model Alignment and Retrieval from a Single Image

## Source

```bash
https://github.com/cangumeli/ROCA
```

## Download

```bash
https://drive.google.com/drive/folders/1ZOY50DjC85n06fTyYPc8feZiy9-fif3j?usp=sharing
```

unzip

```bash
Data/Dataset.zip
Data/Images.zip
Data/Rendering.zip
```

and create a link

```bash
ln -s <path-to-Models-folder> ./Models
ln -s <path-to-Data-folder> ./Data
```

## Install

### Simple

```bash
source setup.sh
```

### Step by step

```bash
conda create -n roca python=3.8 -y
conda activate roca

cd ..
git clone --recursive https://github.com/cangumeli/Scan2CADRasterizer.git
cd Scan2CADRasterizer
pip install .
cd ../image-to-cad

pip install dataclasses opencv-python numpy-quaternion \
  pandas scipy trimesh rtree numba open3d==0.13.0

conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d-nightly -y

pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f \
  https://download.pytorch.org/whl/torch_stable.html

python -m pip install detectron2==0.3+cu110 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html
```

## Install habitat-sim

```bash
git clone https://github.com/565353780/habitat-sim-manage.git ./network/habitat_sim_manage
sudo apt install libassimp5 libassimp-dev libassimp-doc
conda install habitat-sim -c conda-forge -c aihabitat
pip install numpy, matplotlib
```

## Run

### ROCA-Sim

```bash
conda activate roca
cd network
python sim_demo.py
```

### ROCA-MultiViewMerge

```bash
conda activate roca
cd network
python roca_demo.py
```

## Enjoy it~

