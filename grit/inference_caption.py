import os
import hydra
import random
import numpy as np
from omegaconf import DictConfig

from datasets.caption.field import TextField
from datasets.caption.coco import build_coco_dataloaders
from models.caption import Transformer, GridFeatureNetwork, CaptionGenerator

from models.caption.detector import build_detector
from models.common.attention import MemoryAttention

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from engine.caption_engine import *


import torch

# model
from models.common.attention import MemoryAttention
from models.caption.detector import build_detector
from models.caption import Transformer, GridFeatureNetwork, CaptionGenerator

# dataset
from PIL import Image
from datasets.caption.field import TextField
from datasets.caption.transforms import get_transform
from engine.utils import nested_tensor_from_tensor_list


@hydra.main(config_path="configs/caption", config_name="coco_config")
def run_main(config: DictConfig) -> None:
    import ipdb; ipdb.set_trace()
    device = torch.device(f"cuda:0")
    detector = build_detector(config).to(device)

    grit_net = GridFeatureNetwork(
        pad_idx=config.model.pad_idx,
        d_in=config.model.grid_feat_dim,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        attention_module=MemoryAttention,
        **config.model.grit_net,
    )
    cap_generator = CaptionGenerator(
        vocab_size=config.model.vocab_size,
        max_len=config.model.max_len,
        pad_idx=config.model.pad_idx,
        cfg=config.model.cap_generator,
        dropout=config.model.dropout,
        attn_dropout=config.model.attn_dropout,
        **config.model.cap_generator,
    )

    model = Transformer(
        grit_net,
        cap_generator,
        detector=detector,
        use_gri_feat=config.model.use_gri_feat,
        use_reg_feat=config.model.use_reg_feat,
        config=config,
    )
    model = model.to(device)

    # load checkpoint
    model.eval()
    if os.path.exists(config.exp.checkpoint):
        checkpoint = torch.load(config.exp.checkpoint, map_location='cpu')
        missing, unexpected = model.load_state_dict(checkpoint['state_dict'], strict=False)
        print("model missing:", len(missing))
        print("model unexpected:", len(unexpected))
        
    model.cached_features = False

    # prepare utils
    transform = get_transform(config.dataset.transform_cfg)['valid']
    text_field = TextField(vocab_path=config.vocab_path if 'vocab_path' in config else config.dataset.vocab_path)

    # load image
    list_path = "/mnt/d/data/COSMOS/amc/public_test_mmsys"
    # import os
    from tqdm import tqdm
    list_img = os.listdir(list_path)
    out_put_dict = {}
    for img_name in tqdm(list_img):
        img_path = os.path.join(list_path, img_name)
        rgb_image = Image.open(img_path).convert('RGB')
        image = transform(rgb_image)
        images = nested_tensor_from_tensor_list([image]).to(device)
        
        # inference and decode
        with torch.no_grad():
            out, _ = model(images,                   
                        seq=None,
                        use_beam_search=True,
                        max_len=config.model.beam_len,
                        eos_idx=config.model.eos_idx,
                        beam_size=config.model.beam_size,
                        out_size=1,
                        return_probs=False,
                        )
            caption = text_field.decode(out, join_words=True)[0]
            out_put_dict[img_name] = caption
            print(caption)
    with open("/mnt/d/data/COSMOS/COSMOS/grit/caption_grit_acm_new.json", "w") as fl:
        json.dump(out_put_dict, fl)

if __name__ == "__main__":
    run_main()