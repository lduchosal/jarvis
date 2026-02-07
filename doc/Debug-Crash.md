# Debug Report: Daemon Crash après ~8 requêtes

## Date
2026-02-07

## Symptôme

Le daemon se bloque ou crash après ~8 requêtes successives. Le stress test reste stuck, le daemon ne répond plus.

## Investigation

### Logs daemon

```
[1] ok 1.8s "Bonjour." -> speakers [mem:5284MB]
[2] ok 1.2s "Oui." -> speakers [mem:5284MB]
...
[8] ok 1.4s "À demain." -> speakers [mem:5284MB]
(silence — plus aucune réponse)
```

Mémoire stable à 5284MB — pas de fuite mémoire.

### Crash report macOS

```
Process: Python [47605]
Exception Type: EXC_BAD_ACCESS (SIGSEGV)
Triggered by Thread: 37 Thread-11 (target)

Thread 37 Crashed:
0  libmlx.dylib  mlx::core::metal::Device::end_encoding(int)
1  libmlx.dylib  mlx::core::gpu::eval(mlx::core::array&)
2  libmlx.dylib  mlx::core::eval_impl(...)
```

SEGFAULT dans `libmlx.dylib` sur le thread 37 (`Thread-11 (target)` = notre `threading.Thread`).

## Cause racine

**MLX (Metal GPU) n'est pas thread-safe.**

Le daemon utilisait `threading.Thread` pour exécuter `handlers.handle()` avec un timeout. Appeler `model.generate_voice_design()` depuis un thread secondaire corrompt l'état interne du device Metal GPU, provoquant un `EXC_BAD_ACCESS` (pointer authentication failure).

Le crash arrivait après ~8 requêtes car l'accumulation d'appels GPU concurrents (threads daemon qui n'ont pas fini + nouvelles requêtes) finissait par corrompre la mémoire.

## Fix

Supprimé `threading.Thread`. Tout s'exécute sur le **main thread**.

Timeout via `signal.alarm(SIGALRM)` à la place — fonctionne uniquement sur le main thread, interrompt proprement la génération sans créer de thread.

```python
# Avant (crash)
thread = threading.Thread(target=lambda: handlers.handle(model, request))
thread.start()
thread.join(timeout=60)

# Après (stable)
signal.alarm(60)
try:
    result = handlers.handle(model, request)
except GenerationTimeout:
    result = {"status": "error", "message": "timeout"}
finally:
    signal.alarm(0)
```

## Leçon

Ne jamais appeler MLX depuis un thread Python secondaire. Toute interaction avec le modèle MLX doit rester sur le main thread. Si on a besoin de concurrence, utiliser des processus séparés (`multiprocessing`) plutôt que des threads.
