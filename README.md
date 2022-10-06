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
ln -s <path-to-Data-folder> ./Dataset
```

## Install

### Simple

```bash
conda create -n roca python=3.8 -y
conda activate roca
./setup.sh
```

## Install habitat-sim

```bash
git clone https://github.com/565353780/habitat-sim-manage.git ../habitat_sim_manage
sudo apt install libassimp5 libassimp-dev libassimp-doc
conda install habitat-sim -c conda-forge -c aihabitat
pip install numpy, matplotlib
```

## Run

### ROCA-Sim

```bash
python sim_demo.py
```

### ROCA-MultiViewMerge

```bash
python roca_demo.py
```

## Enjoy it~

