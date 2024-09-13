from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

llava_model_name_or_path = 'show_model/model001'
# llava_processor = AutoProcessor.from_pretrained(llava_model_name_or_path, torch_dtype=torch.bfloat16, device_map='cuda:0')
llava_model = LlavaForConditionalGeneration.from_pretrained(llava_model_name_or_path, torch_dtype=torch.bfloat16, device_map='cuda:0')
