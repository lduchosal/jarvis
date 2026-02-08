"""Kyutai STT — real-time speech-to-text on Apple Silicon."""

import json
import queue
import sys
import time

import numpy as np
import mlx.core as mx
import mlx.nn as nn
import sentencepiece
import sounddevice as sd
import rustymimi
from huggingface_hub import hf_hub_download
from moshi_mlx import models, utils


SAMPLE_RATE = 24000
BLOCK_SIZE = 1920  # 80ms chunks


def load_model(hf_repo="kyutai/stt-1b-en_fr-mlx", max_steps=4000):
    """Load STT model, text tokenizer, audio tokenizer, and LmGen."""
    config_path = hf_hub_download(hf_repo, "config.json")
    with open(config_path) as f:
        config_dict = json.load(f)

    stt_config = config_dict.get("stt_config", {})
    lm_config = models.LmConfig.from_config_dict(config_dict)

    # Build model
    model = models.Lm(lm_config)
    model.set_dtype(mx.bfloat16)

    # Load weights (detect quantization from filename)
    moshi_name = config_dict.get("moshi_name", "model.safetensors")
    weights_path = hf_hub_download(hf_repo, moshi_name)
    if weights_path.endswith(".q4.safetensors"):
        nn.quantize(model, bits=4, group_size=32)
    elif weights_path.endswith(".q8.safetensors"):
        nn.quantize(model, bits=8, group_size=64)
    model.load_weights(weights_path, strict=True)

    # Text tokenizer
    tokenizer_name = config_dict["tokenizer_name"]
    tokenizer_path = hf_hub_download(hf_repo, tokenizer_name)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    # Audio tokenizer (streaming version for real-time)
    mimi_name = config_dict["mimi_name"]
    mimi_path = hf_hub_download(hf_repo, mimi_name)
    other_codebooks = lm_config.other_codebooks  # 32 for STT
    audio_tokenizer = rustymimi.StreamTokenizer(mimi_path, num_codebooks=other_codebooks)

    # Condition tensor (if model has conditioners)
    ct = None
    if model.condition_provider is not None:
        ct = model.condition_provider.condition_tensor("description", "very_good")

    # Warmup
    model.warmup(ct)

    # Build generator — greedy text (temp=0), standard audio sampling
    gen = models.LmGen(
        model=model,
        max_steps=max_steps,
        text_sampler=utils.Sampler(temp=0.0, top_k=50),
        audio_sampler=utils.Sampler(temp=0.8, top_k=250),
        check=False,
    )

    return {
        "gen": gen,
        "model": model,
        "text_tokenizer": text_tokenizer,
        "audio_tokenizer": audio_tokenizer,
        "ct": ct,
        "other_codebooks": other_codebooks,
        "stt_config": stt_config,
        "max_steps": max_steps,
        "mimi_path": mimi_path,
    }


def reset(bundle):
    """Reset generator and audio tokenizer for a new listening session.

    Must be called between conversation turns to avoid buffer overflow
    (LmGen has a finite max_steps, StreamTokenizer buffer is bounded).
    """
    model = bundle["model"]
    ct = bundle["ct"]
    max_steps = bundle["max_steps"]
    mimi_path = bundle["mimi_path"]
    other_codebooks = bundle["other_codebooks"]

    bundle["gen"] = models.LmGen(
        model=model,
        max_steps=max_steps,
        text_sampler=utils.Sampler(temp=0.0, top_k=50),
        audio_sampler=utils.Sampler(temp=0.8, top_k=250),
        check=False,
    )
    bundle["audio_tokenizer"] = rustymimi.StreamTokenizer(
        mimi_path, num_codebooks=other_codebooks
    )


def listen(bundle, duration=None):
    """Capture mic and transcribe in real-time. Prints text to stdout."""
    gen = bundle["gen"]
    text_tokenizer = bundle["text_tokenizer"]
    audio_tokenizer = bundle["audio_tokenizer"]
    ct = bundle["ct"]
    other_codebooks = bundle["other_codebooks"]

    input_queue = queue.Queue()
    running = True

    def on_input(in_data, frames, time_info, status):
        if running:
            input_queue.put_nowait(in_data[:, 0].astype(np.float32).copy())

    # Warmup audio tokenizer (4 silent frames, like moshi_mlx/local.py)
    for _ in range(4):
        silence = np.zeros(BLOCK_SIZE, dtype=np.float32)
        audio_tokenizer.encode(silence)
        while True:
            time.sleep(0.001)
            data = audio_tokenizer.get_encoded()
            if data is not None:
                break

    print("Listening... (Ctrl+C to stop)", file=sys.stderr, flush=True)
    start_time = time.time()
    step = 0

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=BLOCK_SIZE, callback=on_input):
        try:
            while running:
                if duration and (time.time() - start_time) >= duration:
                    break

                try:
                    pcm = input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                # Encode PCM → audio tokens (non-blocking StreamTokenizer)
                audio_tokenizer.encode(pcm)
                encoded = None
                for _ in range(100):  # wait up to 100ms
                    encoded = audio_tokenizer.get_encoded()
                    if encoded is not None:
                        break
                    time.sleep(0.001)

                if encoded is None:
                    continue

                # Shape: (codebooks, 1) → transpose to (1, codebooks) → slice
                audio_tokens = mx.array(encoded).transpose(1, 0)[:, :other_codebooks]

                # Run model step → (text_token, transformer_out)
                text_token = gen.step(audio_tokens[0], ct)
                text_token = text_token[0].item()
                step += 1

                # Decode and print text token (skip padding tokens 0 and 3)
                if text_token not in (0, 3):
                    text = text_tokenizer.id_to_piece(text_token)
                    text = text.replace("\u2581", " ")
                    print(text, end="", flush=True)

        except KeyboardInterrupt:
            pass
        finally:
            running = False

    print(file=sys.stderr)
    elapsed = time.time() - start_time
    tokens_per_sec = step / elapsed if elapsed > 0 else 0
    print(f"[{step} steps, {tokens_per_sec:.1f} tok/s, {elapsed:.1f}s]",
          file=sys.stderr, flush=True)


def listen_until_silence(bundle, silence_threshold=15, max_duration=30):
    """Listen until silence detected. Returns transcribed text.

    Args:
        silence_threshold: consecutive silent steps before stopping (15 ≈ 1.2s)
        max_duration: max listen time in seconds
    """
    gen = bundle["gen"]
    text_tokenizer = bundle["text_tokenizer"]
    audio_tokenizer = bundle["audio_tokenizer"]
    ct = bundle["ct"]
    other_codebooks = bundle["other_codebooks"]

    input_queue = queue.Queue()
    running = True
    accumulated = []
    silence_count = 0

    def on_input(in_data, frames, time_info, status):
        if running:
            input_queue.put_nowait(in_data[:, 0].astype(np.float32).copy())

    # Warmup audio tokenizer
    for _ in range(4):
        silence = np.zeros(BLOCK_SIZE, dtype=np.float32)
        audio_tokenizer.encode(silence)
        while True:
            time.sleep(0.001)
            data = audio_tokenizer.get_encoded()
            if data is not None:
                break

    start_time = time.time()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1,
                        blocksize=BLOCK_SIZE, callback=on_input):
        try:
            while running:
                if (time.time() - start_time) >= max_duration:
                    break

                try:
                    pcm = input_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                audio_tokenizer.encode(pcm)
                encoded = None
                for _ in range(100):
                    encoded = audio_tokenizer.get_encoded()
                    if encoded is not None:
                        break
                    time.sleep(0.001)

                if encoded is None:
                    continue

                audio_tokens = mx.array(encoded).transpose(1, 0)[:, :other_codebooks]
                text_token = gen.step(audio_tokens[0], ct)
                text_token = text_token[0].item()

                if text_token not in (0, 3):
                    text = text_tokenizer.id_to_piece(text_token)
                    text = text.replace("\u2581", " ")
                    accumulated.append(text)
                    silence_count = 0
                    print(text, end="", flush=True)
                else:
                    if accumulated:
                        silence_count += 1

                if silence_count >= silence_threshold and accumulated:
                    break

        except KeyboardInterrupt:
            pass
        finally:
            running = False

    print(file=sys.stderr)
    return "".join(accumulated).strip()
