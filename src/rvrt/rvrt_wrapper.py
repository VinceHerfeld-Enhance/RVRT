import torch
import torch.nn as nn
from .network_rvrt import RVRT
from pathlib import Path

CHECKPOINT = Path(__file__).parent / "model_zoo/003_RVRT_videosr_bd_Vimeo_14frames.pth"


def load_rvrt_model():
    model = RVRT(
        upscale=4,
        clip_size=2,
        img_size=[2, 64, 64],
        window_size=[2, 8, 8],
        num_blocks=[1, 2, 1],
        depths=[2, 2, 2],
        embed_dims=[144, 144, 144],
        num_heads=[6, 6, 6],
        inputconv_groups=[1, 1, 1, 1, 1, 1],
        deformable_groups=12,
        attention_heads=12,
        attention_window=[3, 3],
        cpu_cache_length=100,
    ).eval()
    pretrained_model = torch.load(CHECKPOINT)
    model.load_state_dict(
        pretrained_model["params"] if "params" in pretrained_model.keys() else pretrained_model, strict=True
    )
    model.sf = 4  # Scale factor

    return model


def rvrt_inference_chunked(model, video, clip_size=2, overlap=1):
    """
    Run RVRT on a long video using temporal chunking.

    Args:
        model: RVRT instance (pretrained)
        video: torch.Tensor, shape (B, T, C, H, W)
        clip_size: number of frames per clip
        overlap: number of overlapping frames between chunks (default 1)

    Returns:
        out_video: torch.Tensor, shape (B, T, C, H*scale, W*scale)
    """
    B, T, C, H, W = video.shape
    pad_frames = (clip_size - (T % clip_size)) % clip_size
    if pad_frames > 0:
        last_frame = video[:, -1:].repeat(1, pad_frames, 1, 1, 1)
        video = torch.cat([video, last_frame], dim=1)

    outputs = []
    idx = 0

    while idx < T:
        end_idx = min(idx + clip_size - 1, T)
        clip = video[:, idx : end_idx + 1]  # shape (B, clip_size, C, H, W)
        print(clip.shape)
        with torch.no_grad():
            out_clip = model(clip)  # shape (B, clip_size, C, H*scale, W*scale)

        # Exclude overlapping frames that have already been appended
        if idx == 0:
            outputs.append(out_clip)
        else:
            outputs.append(out_clip[:, overlap:])  # skip first 'overlap' frames

        idx += clip_size - overlap

    # Concatenate all output clips along temporal dimension
    out_video = torch.cat(outputs, dim=1)  # (B, T, C, H*scale, W*scale)
    out_video = out_video[:, :T]  # Remove any extra padded frames
    return out_video
