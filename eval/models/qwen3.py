"""
Qwen3-VL model wrapper (Qwen3VL-*-Instruct checkpoints via transformers).

The checkpoint is set by `hf_id` in the config `model:` section
(e.g. Qwen/Qwen3-VL-2B-Instruct). Runs on CUDA if available, else CPU.
"""

from __future__ import annotations

from models.model import Model, Part


class Qwen3VL(Model):
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        hf_id = cfg.get("hf_id")
        if not hf_id:
            raise SystemExit("model.hf_id missing in config (e.g. Qwen/Qwen3-VL-2B-Instruct)")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        print(f"Loading {hf_id} on {self.device} ({dtype}) …")
        self.processor = AutoProcessor.from_pretrained(hf_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            hf_id, dtype=dtype, device_map=self.device
        )
        self.model.eval()

    def generate(self, parts: list[Part]) -> str:
        import torch
        from PIL import Image

        content = []
        images = []
        for kind, value in parts:
            if kind == "image":
                images.append(Image.open(value).convert("RGB"))
                content.append({"type": "image"})
            else:
                content.append({"type": "text", "text": value})
        messages = [{"role": "user", "content": content}]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(
            text=[text], images=images or None, return_tensors="pt"
        ).to(self.device)

        temperature = self.cfg.get("temperature", 0.0)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.cfg.get("max_tokens", 256),
                do_sample=temperature > 0,
                temperature=temperature or None,
                top_p=self.cfg.get("top_p", None),
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        return self.processor.decode(new_tokens, skip_special_tokens=True).strip()
