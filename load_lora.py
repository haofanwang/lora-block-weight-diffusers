"""
This script is developed by Haofan Wang, with aim to support block weighting lora in diffusers framework. We support both .bin (trained via diffusers) and .safetensors (common used in civitai) formats. We modify the weight in given LoRA and save it to a new file that has same format as before. The detailed explanation of block weighting can be found in https://github.com/hako-mikan/sd-webui-lora-block-weight.

There are 17 blocks in Stable Diffusion. The downblocks (inblock) have 6 blocks, the midblock has 1 block, and the upblock (outblock) have 9 blocks. The extra 1 block refers to text encoder.

The order of weight ratios is BASE,IN01,IN02,IN03,IN04,IN05,IN06,MID,OUT01,OUT02,OUT03,OUT04,OUT05,OUT06,OUT07,OUT8,OUT9

Our project is inspired by https://github.com/hako-mikan/sd-webui-lora-block-weight.
"""


import torch
from safetensors.torch import load_file, save_file


BLOCK_IDS = ["BASE",
             "IN01","IN02","IN03","IN04","IN05","IN06",
             "MID",
             "OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09"]

DOWNBLOCK_IDS_2_LAYER_IDS = {"00":"IN01", "01":"IN02",
                             "10":"IN03", "11":"IN04",
                             "20":"IN05", "21":"IN06",}

MIDBLOCK_IDS_2_LAYER_IDS = {"00":"MID"}

UPBLOCK_IDS_2_LAYER_IDS = {"10":"OUT01", "11":"OUT02", "12":"OUT03", 
                           "20":"OUT04", "21":"OUT05", "22":"OUT05",
                           "30":"OUT06", "31":"OUT07", "32":"OUT08",}

def get_layer_id_bin(layer_name):
    layer_items = layer_name.split('.')
    layer_id = ""
    for item in layer_items:
        try:
            block_id = int(item)
            layer_id += item
        except Exception:
            continue
    return layer_id[:2]

def block_weight_bin(file_dir, weight_ratios, save_dir=None):
    
    """
    Only for UNet now.
    
    block_weight_bin("pytorch_lora_weights.bin", "1,1,1,1,1,1,1,0.5,1,1,1,1,1,1,1,1,1")
    """

    file = torch.load(file_dir)
    
    if isinstance(weight_ratios,str):
        weight_ratios = weight_ratios.split(',')
        weight_ratios = [float(weight_ratio) for weight_ratio in weight_ratios]
    
    assert len(weight_ratios) == 17
    
    for item in file.keys():
        block_id = None
        if "down_block" in item:
            block_id = BLOCK_IDS.index(DOWNBLOCK_IDS_2_LAYER_IDS[get_layer_id_bin(item)])
        if "mid_block" in item:
            block_id = BLOCK_IDS.index(MIDBLOCK_IDS_2_LAYER_IDS[get_layer_id_bin(item)])
        if "up_block" in item:
            block_id = BLOCK_IDS.index(UPBLOCK_IDS_2_LAYER_IDS[get_layer_id_bin(item)])
        if block_id is not None:
            file[item] *= weight_ratios[block_id]
    
    if save_dir is not None:
        torch.save(file, save_dir)
        
    return file

def get_layer_id_safetensors(layer_name):
    layer_items = layer_name.split('_')
    layer_id = ""
    for item in layer_items:
        try:
            block_id = int(item)
            layer_id += item
        except Exception:
            continue
    return layer_id[:2]

def block_weight_safetensors(file_dir, weight_ratios, save_dir=None):
    
    """
    block_weight_safetensors("your.safetensors", "1,1,1,1,1,1,1,0.5,1,1,1,1,1,1,1,1,1")
    """
    
    # lora_te_text_model_encoder_layers_0_mlp_fc1.alpha
    # lora_unet_down_blocks_0_attentions_1_transformer_blocks_0_attn2_to_k.lora_up.weight
    # lora_unet_up_blocks_3_attentions_2_transformer_blocks_0_attn2_to_q.lora_down.weight
    file = load_file(file_dir)
    
    if isinstance(weight_ratios,str):
        weight_ratios = weight_ratios.split(',')
        weight_ratios = [float(weight_ratio) for weight_ratio in weight_ratios]
    
    assert len(weight_ratios) == 17
    
    for item in file.keys():
        if "text" in item:
            file[item] *= weight_ratios[0]
        block_id = None
        if "down_block" in item:
            block_id = BLOCK_IDS.index(DOWNBLOCK_IDS_2_LAYER_IDS[get_layer_id_safetensors(item)])
        if "mid_block" in item:
            block_id = BLOCK_IDS.index("MID")
        if "up_block" in item:
            block_id = BLOCK_IDS.index(UPBLOCK_IDS_2_LAYER_IDS[get_layer_id_safetensors(item)])
        if block_id is not None:
            file[item] *= weight_ratios[block_id]
    
    if save_dir is not None:
        save_file(file, save_dir)
    
    return file

if __name__ == "__main__":
    
    pass
    