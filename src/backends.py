from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol


class LLMBackend(Protocol):
    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        ...


@dataclass
class OllamaBackend:
    model: str
    host: str = "http://localhost:11434"
    timeout_seconds: int = 180

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        url = f"{self.host.rstrip('/')}/api/generate"
        options: dict[str, float | int] = {}
        if max_tokens is not None:
            options["num_predict"] = max_tokens
        if temperature is not None:
            options["temperature"] = temperature

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
        }
        if options:
            payload["options"] = options

        request = urllib.request.Request(
            url=url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"Ollama request failed with HTTP {exc.code}: {detail}"
            ) from exc
        except urllib.error.URLError as exc:
            raise RuntimeError(
                f"Ollama request failed for {url}: {exc.reason}"
            ) from exc

        payload = json.loads(body)
        text = payload.get("response")
        if not isinstance(text, str):
            raise RuntimeError(f"Ollama response did not contain text: {payload}")
        return text.strip()


def strip_thinking_blocks(text: str) -> str:
    return re.sub(r"<think>\s*.*?\s*</think>\s*", "", text, flags=re.DOTALL).strip()


def configure_hf_environment() -> None:
    os.environ.setdefault("HF_HOME", "hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", "hf_cache/hub")
    os.environ.setdefault("TRANSFORMERS_CACHE", "hf_cache/transformers")
    os.environ.setdefault("HF_DATASETS_CACHE", "hf_cache/datasets")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


@dataclass
class HuggingFaceBackend:
    model: str
    _tokenizer: object | None = field(default=None, init=False, repr=False)
    _processor: object | None = field(default=None, init=False, repr=False)
    _model: object | None = field(default=None, init=False, repr=False)
    _torch: object | None = field(default=None, init=False, repr=False)
    _device: str = field(default="cpu", init=False, repr=False)
    _uses_processor: bool = field(default=False, init=False, repr=False)

    def _lazy_load(self) -> None:
        has_text_frontend = self._tokenizer is not None or self._processor is not None
        if self._model is not None and has_text_frontend and self._torch is not None:
            return

        configure_hf_environment()
        try:
            import torch
            from transformers import (
                AutoConfig,
                AutoModelForCausalLM,
                AutoProcessor,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise RuntimeError(
                "HuggingFace backend requires torch and transformers to be installed."
            ) from exc

        if torch.cuda.is_available():
            self._device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self._device = "mps"
        else:
            self._device = "cpu"

        model_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self._device == "cuda":
            model_kwargs["device_map"] = "auto"
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                model_kwargs["dtype"] = torch.bfloat16
            else:
                model_kwargs["dtype"] = torch.float16
        elif self._device == "mps":
            model_kwargs["dtype"] = torch.float16
        else:
            model_kwargs["dtype"] = torch.float32

        config = AutoConfig.from_pretrained(self.model, trust_remote_code=True)
        architectures = set(getattr(config, "architectures", []) or [])
        model_type = getattr(config, "model_type", None)
        qwen35_model_class = None
        if (
            "Qwen3_5MoeForConditionalGeneration" in architectures
            or model_type == "qwen3_5_moe"
        ):
            try:
                from transformers import Qwen3_5MoeForConditionalGeneration
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen3.6 MoE models require transformers with "
                    "Qwen3_5MoeForConditionalGeneration support. Install "
                    "requirements-hf.txt, which pins transformers>=4.57.1."
                ) from exc
            qwen35_model_class = Qwen3_5MoeForConditionalGeneration
        elif (
            "Qwen3_5ForConditionalGeneration" in architectures
            or model_type == "qwen3_5"
        ):
            try:
                from transformers import Qwen3_5ForConditionalGeneration
            except ImportError as exc:
                raise RuntimeError(
                    "Qwen3.6 requires transformers with Qwen3_5ForConditionalGeneration "
                    "support. Install requirements-hf.txt, which pins transformers>=4.57.1."
                ) from exc
            qwen35_model_class = Qwen3_5ForConditionalGeneration

        if qwen35_model_class is not None:
            processor = AutoProcessor.from_pretrained(self.model, trust_remote_code=True)
            model = qwen35_model_class.from_pretrained(
                self.model,
                **model_kwargs,
            )
            self._processor = processor
            self._uses_processor = True
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
            if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
            self._tokenizer = tokenizer
            self._uses_processor = False

        if self._device != "cpu" and self._device != "cuda":
            model.to(self._device)
        model.eval()

        self._model = model
        self._torch = torch

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self._lazy_load()
        assert self._model is not None
        assert self._torch is not None

        model = self._model
        torch = self._torch
        generation_temperature = 0.0 if temperature is None else temperature
        do_sample = generation_temperature > 0

        if self._uses_processor:
            assert self._processor is not None
            processor = self._processor
            messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
            try:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=True,
                    return_tensors="pt",
                )
            except TypeError:
                inputs = processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt",
                )
            inputs = inputs.to(model.device)
            generation_kwargs = {
                "max_new_tokens": max_tokens or 768,
                "do_sample": do_sample,
            }
            if do_sample:
                generation_kwargs["temperature"] = generation_temperature
            with torch.inference_mode():
                output = model.generate(**inputs, **generation_kwargs)

            prompt_tokens = inputs["input_ids"].shape[-1]
            generated_tokens = output[:, prompt_tokens:]
            decoded = processor.batch_decode(
                generated_tokens,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            )[0]
            return strip_thinking_blocks(decoded)

        assert self._tokenizer is not None
        tokenizer = self._tokenizer
        messages = [{"role": "user", "content": prompt}]
        if hasattr(tokenizer, "apply_chat_template"):
            try:
                rendered_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                rendered_prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
        else:
            rendered_prompt = prompt

        inputs = tokenizer(rendered_prompt, return_tensors="pt")
        inputs = {name: tensor.to(self._device) for name, tensor in inputs.items()}

        generation_kwargs = {
            "max_new_tokens": max_tokens or 768,
            "do_sample": do_sample,
            "pad_token_id": (
                tokenizer.pad_token_id
                if tokenizer.pad_token_id is not None
                else tokenizer.eos_token_id
            ),
            "eos_token_id": tokenizer.eos_token_id,
        }
        if do_sample:
            generation_kwargs["temperature"] = generation_temperature
        with torch.inference_mode():
            output = model.generate(**inputs, **generation_kwargs)

        prompt_tokens = inputs["input_ids"].shape[-1]
        generated_tokens = output[0][prompt_tokens:]
        decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        return strip_thinking_blocks(decoded)
