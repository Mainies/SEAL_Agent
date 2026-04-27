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
    _model: object | None = field(default=None, init=False, repr=False)
    _torch: object | None = field(default=None, init=False, repr=False)
    _device: str = field(default="cpu", init=False, repr=False)

    def _lazy_load(self) -> None:
        if self._model is not None and self._tokenizer is not None and self._torch is not None:
            return

        configure_hf_environment()
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
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

        tokenizer = AutoTokenizer.from_pretrained(self.model, trust_remote_code=True)
        if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

        model_kwargs: dict[str, object] = {
            "trust_remote_code": True,
            "low_cpu_mem_usage": True,
        }
        if self._device == "cuda":
            model_kwargs["device_map"] = "auto"
            if hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported():
                model_kwargs["torch_dtype"] = torch.bfloat16
            else:
                model_kwargs["torch_dtype"] = torch.float16
        elif self._device == "mps":
            model_kwargs["torch_dtype"] = torch.float16
        else:
            model_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(self.model, **model_kwargs)
        if self._device != "cpu" and self._device != "cuda":
            model.to(self._device)
        model.eval()

        self._tokenizer = tokenizer
        self._model = model
        self._torch = torch

    def generate(
        self,
        prompt: str,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        self._lazy_load()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._torch is not None

        tokenizer = self._tokenizer
        model = self._model
        torch = self._torch

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

        generation_temperature = 0.0 if temperature is None else temperature
        do_sample = generation_temperature > 0
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
        decoded = re.sub(r"<think>\s*.*?\s*</think>\s*", "", decoded, flags=re.DOTALL)
        return decoded.strip()
