cd ..
git clone git@github.com:565353780/habitat-sim-manage.git
git clone --recursive https://github.com/cangumeli/Scan2CADRasterizer.git

cd habitat-sim-manage
./dev_setup.sh

cd ../Scan2CADRasterizer
pip install .

pip install dataclasses opencv-python numpy-quaternion \
  pandas scipy trimesh rtree numba open3d==0.13.0

conda install -c fvcore -c iopath -c conda-forge fvcore iopath -y
conda install -c bottler nvidiacub -y
conda install pytorch3d -c pytorch3d-nightly -y

pip install torch==1.7.0+cu110 torchvision==0.8.1+cu110 torchaudio==0.7.0 -f \
  https://download.pytorch.org/whl/torch_stable.html

python -m pip install detectron2==0.3+cu110 -f \
  https://dl.fbaipublicfiles.com/detectron2/wheels/cu110/torch1.7/index.html

