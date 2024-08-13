from llava.model.builder import load_pretrained_model
from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

import torch

import warnings
warnings.filterwarnings('ignore')

import PIL.Image as Image

model_path = "liuhaotian/llava-v1.5-7b"

tokenizer, model, image_processor, context_len = load_pretrained_model(
    model_path=model_path,
    model_base=None,
    model_name=get_model_name_from_path(model_path)
    # cache_dir='./checkpoints/original'
)

# prompt = "Describe this image"
# en_prompt = torch.tensor(tokenizer.encode(prompt), 
#                          device='cuda:0', 
#                          dtype=torch.long).unsqueeze(0)
# print(en_prompt)

# img = Image.open('./datasets/amazon/train/train-jpg/train_0.jpg')

# en_img = image_processor(img)
# print(en_img.keys())

# print(context_len)

# next_token = model(input_ids=en_prompt)

# print(next_token)