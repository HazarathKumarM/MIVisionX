
from amd.rocal.plugin.generic import ROCALClassificationIterator
from amd.rocal.pipeline import Pipeline
import amd.rocal.fn as fn
import amd.rocal.types as types
from parse_config import parse_args
import os
import sys
import cv2
import cupy as cp
import inspect
import builtins
import ast
import importlib
import subprocess
from external_source_operator import external_source
import random

def generate_random_numbers(count):
    """Generate a list of random numbers."""
    return [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]

def draw_patches(img, idx, device):
    # image is expected as a tensor, bboxes as numpy
    if device == "gpu":
        img = cp.asnumpy(img)
    # img = img.transpose([0, 2, 3, 1])
    images_list = []
    for im in img:
        images_list.append(im)
    img = cv2.vconcat(images_list)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite("external_source_contrast" + ".png", img,
                [cv2.IMWRITE_PNG_COMPRESSION, 9])


def main():
    # Create Pipeline instance
    batch_size = 5
    num_threads = 1
    device_id = 0
    local_rank = 0
    world_size = 1
    rocal_cpu = True
    random_seed = 0
    max_height = 720
    max_width = 640
    color_format = types.RGB
    data_path="/media/rpp_audio/MIVisionX-data/rocal_data/coco/coco_10_img/train_10images_2017/"
    decoder_device = 'cpu'
    # Execute the pythonScript containing read_array_from_file definition
    pythonScript = inspect.getsource(generate_random_numbers)
    data_type = types.UINT8
    file_path = os.path.abspath(__file__)
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=random_seed, rocal_cpu=rocal_cpu, tensor_layout=types.NHWC , tensor_dtype=types.UINT8, output_memory_type=types.HOST_MEMORY if rocal_cpu else types.DEVICE_MEMORY)
    with pipe:
        # output = fn.ExternalSource(pythonScript, data_type, sys.getsizeof(pythonScript))
        jpegs, _ = fn.readers.file(file_root=data_path)
        images = fn.decoders.image(jpegs,
                                    file_root=data_path,
                                    device=decoder_device,
                                    max_decoded_width=max_width,
                                    max_decoded_height=max_height,
                                    output_type=color_format,
                                    shard_id=local_rank,
                                    num_shards=world_size,
                                    random_shuffle=False)
        output = external_source(images, filePath = file_path, pythonScript = pythonScript, dtype=data_type, size=5, batch=True)
        contrast_output = fn.contrast(images,
                            contrast=output,
                            contrast_center=output)

        print("output...", output)
        pipe.set_outputs(output)
    pipe.build()
    
    # Dataloader
    data_loader = ROCALClassificationIterator(
        pipe, device="cpu", device_id=local_rank)
    cnt = 0

    # Enumerate over the Dataloader
    for epoch in range(int(2)):
        print("EPOCH:::::", epoch)
        for i, (output_list, labels) in enumerate(data_loader, 0):
            for j in range(len(output_list)):
                print("**************", i, "*******************")
                print("**************starts*******************")
                print("\nImages:\n", output_list[j])
                print("\nLABELS:\n", labels)
                print("**************ends*******************")
                print("**************", i, "*******************")
                draw_patches(output_list[j], cnt, "cpu")
                cnt += len(output_list[j])

        data_loader.reset()

if __name__ == '__main__':
    main()