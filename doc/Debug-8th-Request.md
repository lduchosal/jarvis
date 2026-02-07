# Debug: Daemon timeouts on various texts

## Symptom
The daemon consistently times out on certain texts (~30% failure rate). User reported "fails at the 8th command every time" — actually non-deterministic, but frequent enough to seem consistent.

## Root cause
`model.generate_voice_design()` default parameters cause the autoregressive decoder to enter infinite generation loops:
- `temperature=0.9` — too high, causes erratic token sampling
- `repetition_penalty=1.05` — too low, model gets stuck in repetitive token loops
- `max_tokens=4096` — way too high for short/medium text, lets loops run forever

## Fix (in handlers.py)
Tuned generation parameters:
```python
max_tokens = max(256, min(4096, len(text) * 20))
gen_kwargs = {
    "temperature": 0.7,        # was 0.9
    "repetition_penalty": 1.2,  # was 1.05
    "max_tokens": max_tokens,   # was 4096
}
```

## Additional fixes (in daemon.py)
1. **Retry on timeout**: `MAX_RETRIES = 2` with `GenerationTimeout(BaseException)` so it's not caught by handler's `except Exception`
2. **Dynamic timeout**: scales with text length: `min(60, max(10, 10 + len(text) // 10))`
3. **Chunk limit**: `max_chunks = max(50, len(text) * 5)` in handlers.py

## Results

| Metric | Before | After |
|--------|--------|-------|
| Success rate | 68.5% (37/54) | **96.3% (52/54)** |
| Short texts | 9/10 | **10/10** |
| Medium texts | 2/10 | **10/10** |
| Long texts | 1/3 | **3/3** |
| Very long | 0/1 | **1/1** |
| Edge cases | 6/10 | **8/10** |
| Repeat | 20/20 | 20/20 |
| Only failures | — | empty + whitespace (expected) |

## Key insight
The bug was NOT in the daemon code — it was in the **generation parameters**. The default `temperature=0.9` from the library is calibrated for Chinese text examples. French/English text with lower temperature and higher repetition penalty is much more stable.
