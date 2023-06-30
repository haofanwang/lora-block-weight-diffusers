# lora-block-weight-diffusers
When applying Lora, strength can be set block by block. Support for diffusers framework.

## Why it works?
The core idea behind is that there are differences in the semantics captured by different layers of the neural network, and the lightweight LoRA is easy to overfit, thus appropriate
adjustment of weights can improve generated result to some extents.

## Usage
```bash
"""
For LoRA in stable diffusion model, there are 17 blocks.

BLOCK_IDS = ["BASE",
             "IN01","IN02","IN03","IN04","IN05","IN06",
             "MID",
             "OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09"]

"BASE": TextEncoder
"IN01","IN02","IN03","IN04","IN05","IN06": UNet_Downblocks
"MID": UNet_Midblock
"OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09": UNet_Upblocks

We can manually set stregths for each blocks. The weight_ratios can be a string separated by comma or a list of float values.
"""

from load_lora import *

# convert .bin
state_dict = block_weight_bin(file_dir="your.bin", weight_ratios="1,1,1,1,1,1,1,0.5,1,1,1,1,1,1,1,1,1", save_dir=None)

# convert .safetensors
state_dict = block_weight_safetensors(file_dir="your.safetensors", weight_ratios="1,1,1,1,1,1,1,0.5,1,1,1,1,1,1,1,1,1", save_dir="./new.safetensors")
```

## How to set weights?
To achieve best result, it is better to adjust case by case, but there do exists some general suggestions.

<center><img src="https://github.com/haofanwang/lora-block-weight-diffusers/raw/main/setting.png" width="70%" height="70%"></center> 

## Acknowledgement
This project is inspired by [sd-webui-lora-block-weight](https://github.com/hako-mikan/sd-webui-lora-block-weight), and mainly tagetting for the convenience of advanced developers.

## Contact
If you have any issue, feel free to contact me via haofanwang.ai@gmail.com.
