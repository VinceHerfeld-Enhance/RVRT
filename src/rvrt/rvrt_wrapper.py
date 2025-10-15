import torch
import torch.nn as nn
from .network_rvrt import RVRT

CHECKPOINT = "model_zoo/003_RVRT_videosr_bd_Vimeo_14frames.pth"

class RVRTWrapper(nn.Module):
    def __init__(self, model):
        super(RVRTWrapper, self).__init__()
        self.model = RVRT(upscale=4, clip_size=2, img_size=[2, 64, 64], window_size=[2, 8, 8], num_blocks=[1, 2, 1],
                    depths=[2, 2, 2], embed_dims=[144, 144, 144], num_heads=[6, 6, 6],
                    inputconv_groups=[1, 1, 1, 1, 1, 1], deformable_groups=12, attention_heads=12,
                    attention_window=[3, 3], cpu_cache_length=100).eval()
        self.model.load_state_dict(torch.load(CHECKPOINT))
        

    def forward(self, x):
        return self.model(x)