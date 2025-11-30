import sys
from rknn.api import RKNN

def export_rknn_inference(onnx_path, output_path, platform='rk3566'):
    # Create RKNN execution object
    rknn = RKNN(verbose=True)

    # 1. Config
    # We embed the normalization into the NPU model.
    # ArcFace expects (x - 127.5) / 127.5
    print('--> Config model')
    rknn.config(
        mean_values=[[127.5, 127.5, 127.5]], 
        std_values=[[127.5, 127.5, 127.5]], 
        target_platform=platform
    )
    print('done')

    # 2. Load ONNX
    # ArcFace inputs are typically named 'input' or 'data'. 
    # w600k models usually have input shape [1, 3, 112, 112]
    print('--> Loading model')
    ret = rknn.load_onnx(
        model=onnx_path, 
        input=['input.1'],
        input_size_list=[[1, 3, 112, 112]]
    )
    if ret != 0:
        print('Load model failed!')
        return ret
    print('done')

    # 3. Build Model
    # We use do_quantization=False to use FP16. 
    # This ensures high accuracy without needing a calibration dataset.
    print('--> Building model')
    ret = rknn.build(do_quantization=False)
    if ret != 0:
        print('Build model failed!')
        return ret
    print('done')

    # 4. Export RKNN
    print('--> Export rknn model')
    ret = rknn.export_rknn(output_path)
    if ret != 0:
        print('Export rknn model failed!')
        return ret
    print('done')

    rknn.release()
    return 0

if __name__ == '__main__':
    # Usage: python onnx2rknn.py weights/w600k_mbf.onnx weights/w600k_mbf.rknn
    if len(sys.argv) < 3:
        print("Usage: python onnx2rknn.py <onnx_model> <rknn_output_path>")
        sys.exit(1)
    
    onnx_file = sys.argv[1]
    rknn_file = sys.argv[2]
    
    export_rknn_inference(onnx_file, rknn_file)