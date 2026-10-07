"""Qwen3-Omni thinker with Clef's joint schema head.

The thinker reads video frames and the clip's audio on one interleaved timeline
(`use_audio_in_video=True`). The head turns its final hidden states into one logit per
allowed option, exactly as in Clef. The head starts from scratch, because Clef's
trained head expects a different backbone.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import torch

from vjev import clef_schema as cs

OMNI_ID = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
FPS = 2.0
SR = 16_000
# Clef-flash's head sizes (joint_head_config.json), with the hidden size taken from the thinker.
HEAD_CONFIG = {"width": 1024, "routing_layers": 2, "layers": 4, "heads": 16, "feedforward": 4096}
# LoRA on the text model's attention only. Fullmatch, so the audio tower's layers are excluded.
LORA_TARGETS = r"model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj"
MEDIA_KEYS = ("input_features", "feature_attention_mask", "pixel_values_videos", "video_grid_thw",
              "video_second_per_grid")
PREFIX = f"<|im_start|>system\n{cs.SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\nSTATE:\n"


@dataclass
class Encoded:
    record: cs.EncodedRecord
    media: dict[str, torch.Tensor]
    audio: bool


def encode(processor, record: dict, frames: np.ndarray, audio: np.ndarray | None) -> Encoded:
    """Encode text with Clef's format, then splice the media tokens in after the prefix."""
    tok = processor.tokenizer
    base = cs.encode_record(tok, record)
    if len(frames) % 2:  # the video processor groups frames in pairs
        frames = frames[:-1]
    kwargs = {"fps": FPS, "do_sample_frames": False, "return_tensors": "pt"}
    if audio is not None:
        n = int(len(frames) / FPS * SR)
        audio = np.pad(audio[:n], (0, max(0, n - len(audio))))  # silent clips get silence
        out = processor(text=["<|vision_start|><|video_pad|><|vision_end|>\n"], videos=[frames],
                        audio=[audio], use_audio_in_video=True, sampling_rate=SR, **kwargs)
    else:
        out = processor(text=["<|vision_start|><|video_pad|><|vision_end|>\n"], videos=[frames],
                        **kwargs)
    media_ids = out["input_ids"][0].tolist()
    p, m = len(cs._tokens(tok, PREFIX)), len(media_ids)
    ids = base.input_ids
    questions = tuple(
        replace(q,
                question_span=(q.question_span[0] + m, q.question_span[1] + m),
                option_spans=tuple((a + m, b + m) for a, b in q.option_spans))
        for q in base.questions
    )
    record_out = replace(base, input_ids=(*ids[:p], *media_ids, *ids[p:]), questions=questions)
    media = {}
    for k in MEDIA_KEYS:
        if k in out:
            v = out[k]
            media[k] = v if isinstance(v, torch.Tensor) else torch.tensor(v)
    return Encoded(record_out, media, audio is not None)


class TypedOmni(torch.nn.Module):
    def __init__(self, thinker, head: cs.JointSchemaHead, out_weight: torch.Tensor):
        super().__init__()
        self.thinker = thinker
        self.head = head
        # lm_head is replaced with Identity, so this is the only copy of the output embeddings.
        self.register_buffer("out_weight", out_weight, persistent=False)

    def forward(self, enc: Encoded) -> list[torch.Tensor]:
        dev = self.out_weight.device
        ids = torch.tensor([enc.record.input_ids], device=dev)
        mask = torch.ones_like(ids)
        media = {k: v.to(dev) for k, v in enc.media.items()}
        for k in ("input_features", "pixel_values_videos"):
            if k in media:
                media[k] = media[k].to(torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = self.thinker(input_ids=ids, attention_mask=mask, use_cache=False,
                                  use_audio_in_video=enc.audio, **media).logits
            return self.head(hidden, ids, mask, [enc.record], self.out_weight)[0]


def load(path: str, lora_r: int, lora_alpha: int, dropout: float = 0.05):
    """Return (model, processor). Only LoRA and the head are trainable."""
    from peft import LoraConfig, get_peft_model
    from transformers import Qwen3OmniMoeForConditionalGeneration, Qwen3OmniMoeProcessor

    full = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        path, dtype=torch.bfloat16, device_map={"": "cuda"}, attn_implementation="sdpa")
    full.disable_talker()
    thinker = full.thinker
    thinker.requires_grad_(False)
    thinker.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    out_weight = thinker.lm_head.weight.detach()
    thinker.lm_head = torch.nn.Identity()  # we only need hidden states, not 152k-way logits
    thinker = get_peft_model(thinker, LoraConfig(
        r=lora_r, lora_alpha=lora_alpha, lora_dropout=dropout, target_modules=LORA_TARGETS))
    hidden = full.config.thinker_config.text_config.hidden_size
    head = cs.JointSchemaHead(hidden_size=hidden, dropout=0.1, **HEAD_CONFIG).to("cuda")
    processor = Qwen3OmniMoeProcessor.from_pretrained(path)
    return TypedOmni(thinker, head, out_weight), processor
