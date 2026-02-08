# Ideas & Future Improvements

## Robustesse

- [ ] Tester les textes longs — trouver la limite du modèle et gérer le découpage automatique
- [ ] Tester plusieurs requêtes rapides à la suite — file d'attente ou rejet ?
- [ ] Gestion du cas où deux daemons tournent en même temps (lock file / PID file)
- [ ] Timeout sur la génération pour éviter les blocages
- [ ] Reconnexion automatique du client si le daemon redémarre

## Latence & Performance

- [ ] Mesurer le temps réel du daemon (time-to-first-audio vs mode inline)
- [ ] Pré-warm du modèle avec une génération silencieuse au démarrage
- [ ] Garder le `sd.OutputStream` ouvert entre les requêtes (éviter open/close à chaque fois)

## Voix & Qualité

- [ ] Profils de voix nommés (ex: `--voice jarvis` au lieu de `-i "deep masculine..."`)
- [ ] Tester différentes langues et documenter ce qui marche bien
- [ ] Mixer les langues dans un même texte (code-switching)

## UX / CLI

- [ ] Mode daemon en background (`q3tts serve --daemon` avec fork + detach)
- [ ] Auto-start du daemon si pas lancé (le client le lance tout seul)
- [ ] Indicateur de progression côté client pendant la génération
- [ ] Mode interactif / REPL : taper du texte en boucle sans relancer le client

## Speech-to-Text (STT)

- [ ] Le daemon écoute le micro en continu via `sounddevice.InputStream`
- [ ] Un modèle STT (Whisper MLX ?) transcrit la voix en texte en temps réel
- [ ] Détection de fin de parole (silence / VAD) pour déclencher la transcription
- [ ] Utiliser `kAudioDevicePropertyVoiceActivityDetectionState` (CoreAudio HAL) pour la VAD hardware — le Neural Engine M4 détecte l'activité vocale nativement, plus fiable et rapide que notre compteur de tokens silence. Nécessite un binding Python vers CoreAudio (PyObjC ou ctypes).
- [ ] Séparation de sources audio : isoler les voix, filtrer le bruit ambiant
- [ ] Exploiter Voice Isolation macOS (Neural Engine + beamforming 3 micros) — transparent via `sounddevice`, l'utilisateur active dans Control Center. Avec AirPods Pro 3, double beamforming (H2 + M4).
- [ ] Reconnaissance multi-locuteurs (speaker diarization) : identifier qui parle dans la discussion
- [ ] Pipeline complet : micro → STT → LLM → TTS → speaker (boucle conversationnelle)

## LLM (Claude)

- [ ] Instance Claude persistante (conversation avec historique)
- [ ] Le texte transcrit par le STT est envoyé à Claude via l'API Anthropic
- [ ] La réponse de Claude est envoyée au daemon TTS pour être prononcée
- [ ] Boucle complète : micro → STT → Claude → TTS → speaker = discussion orale
- [ ] System prompt personnalisé pour Jarvis (ton, personnalité, contexte)
- [ ] Streaming de la réponse Claude → TTS phrase par phrase (réduire la latence perçue)

## Intégrations

- [ ] Pipe depuis d'autres outils (ex: `claude | q3tts` pour lire les réponses)
- [ ] Webhook / API HTTP en plus du socket Unix

## Visualisation

- [ ] Générer un spectrogramme de l'audio généré (affichage temps réel ou post-génération)

## Dev Experience

- [ ] Tests automatisés (envoyer du texte, vérifier que le daemon répond ok)
- [ ] CI : lint + type check sur `handlers.py` et `q3tts_daemon.py`
- [ ] File watcher optionnel pour pré-reload de `handlers.py` avant la requête
