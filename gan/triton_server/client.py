import numpy as np
import sys
import gevent.ssl

import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

import SimpleITK as sitk
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from datasets.common_ds import GANDataWrapper, get_common_transform

import SimpleITK as sitk

# 1. 尝试连接服务
url = '10.100.37.100:8700'
verbose = False
model_name = 'cta2mbf_generator'

triton_client = httpclient.InferenceServerClient(url=url, verbose=verbose)

img = np.random.rand(1,1,448,448,128)
img = np.array(img, dtype=np.float32)

infile = '/fileser/zhangwd/data/cardiac/cta2mbf/data_114_20210318/5.mbf_myocardium/1069558/cropped_cta.nii.gz'
# img = sitk.ReadImage(infile)
image_shape = [320,320,160]
transforms = get_common_transform(image_shape, 'GAN_INFERENCE')
img = GANDataWrapper.get_processed_image(transforms,infile)
img = np.array(img.src.data, dtype=np.float32)
img = np.expand_dims(img, 0)

inputs = []
inputs.append(httpclient.InferInput('INPUT__0', [1,1,320,320,160], "FP32"))
inputs[0].set_data_from_numpy(img, binary_data=False)
outputs = []
outputs.append(httpclient.InferRequestedOutput('OUTPUT__0', binary_data=True))

results = triton_client.infer(model_name,
                                inputs,
                                outputs=outputs)

output = results.as_numpy('OUTPUT__0')[0][0]
output = np.array(output, dtype=np.int16)
sitk_img = sitk.GetImageFromArray(output)
os.makedirs('./results', exist_ok=True)
sitk.WriteImage(sitk_img, './results/tritonserver_model_test.nii.gz')

print('hello world')