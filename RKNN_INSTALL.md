**IMPORTANT:** Use ubuntu image from [this](https://joshua-riek.github.io/ubuntu-rockchip-download/)


```bash
sudo apt-get update
sudo apt-get install libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 libgl1-mesa-glx libprotobuf-dev gcc

# from rknn-toolkit2 repo
pip install rknn-toolkit-lite2/packages/rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
cd rknpu2/runtime/Linux/librknn_api/aarch64
sudo cp ./librknnrt.so /usr/lib
cd rknpu2/runtime/Linux/librknn_api/include
sudo cp ./rknn_* /usr/include

# from this repo
sh download.sh
pip install -r requirements.txt

python3 onnx2rknn_arcface.py weights/w600k_mbf.onnx weights/w600k_mbf.rknn
python3 onnx2rknn_arcface.py weights/w600k_r50.onnx weights/w600k_r50.rknn

python3 onnx2rknn_scrfd.py weights/det_500m.onnx weights/det_500m.rknn
python3 onnx2rknn_scrfd.py weights/det_2.5g.onnx weights/det_2.5g.rknn
python3 onnx2rknn_scrfd.py weights/det_10g.onnx weights/det_10g.rknn

python3 main.py --source assets/in_video.mp4 --det-weight weights/det_500m.rknn --rec-weight weights/w600k_mbf.rknn
```