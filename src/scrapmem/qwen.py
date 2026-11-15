"""Qwen3-VL optical perception and local text/image generation."""

from __future__ import annotations

from pathlib import Path
import base64
import io
import json

from .backends import VLMBackend


class QwenClient:
    def __init__(
        self,
        model="Qwen/Qwen3-VL-2B-Instruct",
        *,
        device_map="auto",
        load_in_4bit=True,
        revision=None,
        max_image_pixels=1048576,
    ):
        try:
            import torch
            from transformers import (
                AutoProcessor,
                Qwen3VLForConditionalGeneration,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Install scrapmem[models,vision] for the Qwen backend"
            ) from exc
        options = dict(device_map=device_map, revision=revision, torch_dtype="auto")
        if load_in_4bit:
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "4-bit Qwen serving requires a CUDA device; set load_in_4bit=false for CPU"
                )
            options["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            model, **options
        ).eval()
        self.processor = AutoProcessor.from_pretrained(
            model, revision=revision, max_pixels=max_image_pixels
        )
        self.torch = torch
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self.name = (
            model + (f"@{revision}" if revision else "") + f":pixels={max_image_pixels}"
        )

    def _messages(self, messages):
        from PIL import Image

        converted = []
        for message in messages:
            content = message["content"]
            if isinstance(content, str):
                converted.append(message)
                continue
            parts = []
            for part in content:
                if part["type"] == "text":
                    parts.append(part)
                elif part["type"] == "image_url":
                    url = part["image_url"]["url"]
                    if not url.startswith("data:image/"):
                        raise ValueError("Local Qwen images must use data URLs")
                    data = base64.b64decode(url.split(",", 1)[1], validate=True)
                    with Image.open(io.BytesIO(data)) as image:
                        parts.append({"type": "image", "image": image.convert("RGB")})
                else:
                    raise ValueError("Unsupported message content type")
            converted.append(dict(message, content=parts))
        return converted

    def chat(self, messages, *, max_tokens=1024, json_object=False, timeout=None):
        # Greedy decoding (temperature 0).
        converted = self._messages(messages)
        inputs = self.processor.apply_chat_template(
            converted,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        generation = {"max_new_tokens": max_tokens, "do_sample": False}
        if timeout is not None:
            generation["max_time"] = timeout
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation)
        new = output[:, inputs["input_ids"].shape[1] :]
        self.usage["input_tokens"] += inputs["input_ids"].shape[1]
        self.usage["output_tokens"] += new.shape[1]
        return self.processor.batch_decode(
            new, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    def json(self, messages, **kwargs):
        raw = self.chat(messages, json_object=True, **kwargs).strip()
        if raw.startswith("```") and raw.endswith("```"):
            raw = "\n".join(raw.splitlines()[1:-1])
        result = json.loads(raw)
        if not isinstance(result, dict):
            raise ValueError("Expected a JSON object from Qwen")
        return result

    def image_token_count(self, image):
        from PIL import Image

        with Image.open(image) as raw:
            batch = self.processor.image_processor(
                images=raw.convert("RGB"), return_tensors="pt"
            )
        grid = batch["image_grid_thw"]
        merge = self.model.config.vision_config.spatial_merge_size
        return int((grid.prod(dim=-1) // merge**2).sum().item())


class NativeQwenBackend(VLMBackend):
    """Share the perception model across OCR and text-side node extraction."""

    def __init__(self, perception: dict, answer: dict | None = None):
        self.perceiver = QwenClient(**perception)
        self.answerer = (
            QwenClient(**answer) if answer and answer != perception else self.perceiver
        )
