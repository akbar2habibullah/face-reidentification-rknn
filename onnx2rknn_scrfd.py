import sys
from rknn.api import RKNN

def export_scrfd_rknn(onnx_path, output_path, platform='rk3566'):
    rknn = RKNN(verbose=True)

    # 1. Config
    # SCRFD uses mean=127.5, std=128.0
    print('--> Config model')
    rknn.config(
        mean_values=[[127.5, 127.5, 127.5]],
        std_values=[[128.0, 128.0, 128.0]],
        target_platform=platform
    )

    # 2. Load ONNX
    # Force input shape to 640x640.
    # SCRFD ONNX usually has input name 'input.1' or 'input'
    print('--> Loading model')
    ret = rknn.load_onnx(
        model=onnx_path,
        inputs=['input.1'],
        input_size_list=[[1, 3, 640, 640]]
    )
    if ret != 0:
        print('Load model failed!')
        return ret

    # 3. Build (FP16 mode)
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        return ret

    # 4. Export
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        return ret

    rknn.release()
    print('done')

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: python onnx2rknn_scrfd.py weights/det_10g.onnx weights/det_10g.rknn")
        sys.exit(1)

    export_scrfd_rknn(sys.argv[1], sys.argv[2])