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
```

and create a link

```bash
ln -s <path-to-Models-folder> ./Models
ln -s <path-to-Data-folder> ./Data
```

## Install

```
conda create -n roca python=3.8
conda activate roca

git clone --recursive https://github.com/cangumeli/Scan2CADRasterizer.git
cd Scan2CADRasterizer
pip install .
cd ..

cd image-to-cad
source setup.sh
```

## Run

```bash
conda activate roca
cd network
python demo.py --model_path ../Models/model_best.pth --data_dir ../Data/Dataset --config_path ../Models/config.yaml
```

## Enjoy it~

