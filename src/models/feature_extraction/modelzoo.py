from timm.layers import SwiGLUPacked
from huggingface_hub import login
from dotenv import load_dotenv
from typing import Union
import logging
import torch
import timm
import os

load_dotenv()

logger = logging.getLogger()
logging.basicConfig(level=logging.DEBUG)

def vith14UNI2(checkpoint: Union[bool, str]):
    timm_kwargs = {
                'model_name': 'vit_giant_patch14_224',
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
    model = timm.create_model(
        pretrained=False, **timm_kwargs
    )

    msg = model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
    logger.info(msg)

    return model

def optimus(checkpoint: Union[bool, str]):
    hf_token = os.environ.get("HF_TOKEN")
    login(hf_token)
    
    model = timm.create_model("hf-hub:bioptimus/H-optimus-0", pretrained=True, init_values=1e-5, dynamic_img_size=False)
    
    return model

def virchow2(checkpoint: Union[bool, str]):
    hf_token = os.environ.get("HF_TOKEN")
    login(hf_token)

    model = timm.create_model("hf-hub:paige-ai/Virchow2", pretrained=True, mlp_layer=SwiGLUPacked, act_layer=torch.nn.SiLU)

    return model

def vitGigaPath(checkpoint: Union[bool, str], token=None):
    # -- construct model
    model = timm.create_model(
        model_name="vit_giant_patch14_dinov2",
        img_size=224, 
        patch_size=16,
        init_values=1e-5, 
        num_classes=0, 
        dynamic_img_size=True, 
        depth=40, 
        num_heads=24, 
        mlp_ratio=5.33334)
    model.input_size = 'any square image whose dimensions are divisible by 16 and with 3 channels'
    model.embedding_dim = 1536

    # -- load checkpoint
    if checkpoint:
        if isinstance(checkpoint, str):
            msg = model.load_state_dict(torch.load(checkpoint, weights_only=False), strict=False)
        else:
            if token is not None:
                os.environ['HF_TOKEN'] = token
                try:
                    model = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
                    msg = 'model and checkpoint loaded succesfully from Hugging Face'
                except:
                    msg = 'Unsuccesful attempt to load the model and parameters from Hugging Face. Check token validity'
            else:
                msg = 'An alternative way of loading the model without checkpoints stored locally is described at https://huggingface.co/prov-gigapath/prov-gigapath'
    else:
        msg = 'Returning model without pre-trained parameters'

    logger.info(msg)

    return model 
