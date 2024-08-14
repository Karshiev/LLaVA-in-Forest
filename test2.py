from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

import torch

import warnings
warnings.filterwarnings('ignore')

import PIL.Image as Image
model_path = "liuhaotian/llava-v1.5-7b"
prompt = "Give me a description of this image. Describe the types of land use found at the center-bottom of the image."
image_file = "/ssd_1/sanjar/rsvlm/datasets/amazon/train/train-jpg/train_1.jpg"

# image_file = "https://llava-vl.github.io/static/images/view.jpg"

args = type('Args', (), {
    "model_path": model_path,
    "model_base": None,
    "model_name": get_model_name_from_path(model_path),
    "query": prompt,
    "conv_mode": None,
    "image_file": image_file,
    "sep": ",",
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
    "max_new_tokens": 512
})()

eval_model(args)