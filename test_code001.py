from PIL import Image
import requests
from transformers import AutoProcessor, LlavaForConditionalGeneration
import torch

# 定义模型路径和设备
model_name_or_path = r"E:\playground\ai\models\llava-1.5-7b-hf"
device = "cuda:0"

# 加载预训练模型和处理器
model = LlavaForConditionalGeneration.from_pretrained(
    model_name_or_path, device_map=device, torch_dtype=torch.bfloat16
)
processor = AutoProcessor.from_pretrained(model_name_or_path)

# 定义提示和图像URL
prompt = "USER: <image>\nWhat's the content of the image? ASSISTANT:"
url = "https://www.ilankelman.org/stopsigns/australia.jpg"

# 下载并打开图像
image = Image.open(requests.get(url, stream=True). raw)

# 处理输入
inputs = processor(text=prompt, images=image, return_tensors="pt")

# 将输入张量移动到指定设备
# inputs = {key: value.to(device) for key, value in inputs.items()}
for temp_key in inputs.keys():
    inputs[temp_key] = inputs[temp_key].to(device)


# 生成输出
generate_ids = model.generate(**inputs, max_new_tokens=15)

output = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# 打印输出
print(output)
