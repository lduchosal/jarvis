# Feature: Speech-to-Text (STT)

## Objectif

Le daemon écoute le micro et transcrit la voix en texte en temps réel. Streaming, compatible Apple Silicon (ANE/MLX), bon en français.

## Contraintes

- Apple Neural Engine (ANE) / MLX compatible
- Léger et rapide (low latency)
- Streamable (traitement chunk par chunk en temps réel)
- Français natif ou excellent
- Disponible sur Hugging Face
- Installable via pip

## Modèles candidats

### 1. Kyutai STT 1B en_fr (recommandé)

Le meilleur match pour notre cas : vrai streaming + français natif + MLX.

| Critère | Détail |
|---------|--------|
| HuggingFace | `kyutai/stt-1b-en_fr-mlx` |
| Paramètres | ~1B |
| Streaming | Vrai streaming (pas du chunked Whisper) |
| Latence | 0.5s de délai, ~4x temps réel sur Apple Silicon |
| Français | Natif bilingue EN/FR (600h de données FR) |
| MLX | Implémentation native MLX |
| VAD | Détection de voix intégrée |
| Licence | CC-BY 4.0 |

```bash
pip install moshi-mlx>=0.2.6
```

```python
# Transcription depuis le micro
python scripts/stt_from_mic_mlx.py

# Transcription d'un fichier
python -m moshi_mlx.run_inference --hf-repo kyutai/stt-1b-en_fr-mlx audio.mp3
```

**Points forts** : vrai streaming, VAD intégré, sortie ponctuée et capitalisée, testé sur audio jusqu'à 2h.

### 2. Whisper large-v3-turbo via mlx-audio (fallback simple)

Déjà dans notre stack — zéro nouvelle dépendance.

| Critère | Détail |
|---------|--------|
| HuggingFace | `mlx-community/whisper-large-v3-turbo-asr-fp16` |
| Paramètres | 809M |
| Streaming | Pseudo-streaming (chunks de 30s avec token streaming) |
| Français | Très bon (3-8% WER) |
| MLX | Natif via mlx-audio |
| Licence | MIT |

```python
from mlx_audio.stt.generate import generate_transcription

result = generate_transcription(
    model="mlx-community/whisper-large-v3-turbo-asr-fp16",
    audio="audio.wav",
)

# Streaming token par token
for text in model.stream_transcribe(audio="speech.wav"):
    print(text, end="", flush=True)
```

**Limite** : pas de vrai streaming. Whisper traite des blocs de 30s.

### 3. Whisper French Distil (meilleure précision FR)

Le plus précis en français, fine-tuné sur 2500h+ de données FR.

| Critère | Détail |
|---------|--------|
| HuggingFace | `bofenghuang/whisper-large-v3-french-distil-dec4` |
| Paramètres | ~400M (dec4) à ~1B (dec16) |
| Streaming | Non (batch) |
| Français | Le meilleur : 3.57% WER (MLS), 7.18% (CommonVoice) |
| MLX | Disponible en format MLX |
| Licence | MIT |

**Variantes** : dec16 (précis), dec8 (équilibré), dec4 (rapide, 4.3x speedup), dec2 (ultra-rapide)

### Comparatif

| Modèle | Streaming | Français | Latence | Déjà dans le stack |
|--------|-----------|----------|---------|---------------------|
| **Kyutai STT 1B** | Vrai | Natif | 0.5s | Non |
| Whisper turbo (mlx-audio) | Pseudo | Très bon | ~1-2s chunks | Oui |
| Whisper FR Distil | Non | Meilleur | Batch | Non |

### Modèles écartés

| Modèle | Raison |
|--------|--------|
| Moonshine | Pas de français |
| WhisperKit | Swift only, pas de Python |
| Voxtral Mini 4B | Trop lourd (4B params) |
| whisper.cpp | Pas HuggingFace natif, build C++ |

## Recommandation

**Kyutai STT 1B en_fr** pour le streaming temps réel (boucle conversationnelle).
**Whisper FR Distil dec4** si on a besoin de précision maximale en batch.

## Intégration dans le daemon

```
micro → sounddevice.InputStream
     → chunks audio
     → Kyutai STT (streaming)
     → texte transcrit
     → envoi au daemon (action: "generate" ou futur pipeline Claude)
```

Le STT pourrait tourner dans le même daemon ou dans un processus séparé communiquant via le socket.
