# Feature: Stability & Stress Testing

## Problème

Le daemon crash ou se bloque après plusieurs requêtes successives. On ne connaît pas les limites du modèle TTS (longueur max, caractères spéciaux, etc.). Avant d'ajouter STT et Claude, le daemon doit être solide.

## Objectifs

1. Trouver les limites du modèle (longueur de texte, caractères, langues)
2. Garantir que le daemon survit à des milliers de requêtes
3. Découper automatiquement les textes longs
4. Documenter les cas limites

## 1. Script de stress test

### `tests/stress_test.py`

Un script qui envoie des milliers de requêtes au daemon et collecte les résultats.

```
uv run tests/stress_test.py --count 1000 --report report.json
```

### Catégories de tests

| Catégorie | Exemples | Objectif |
|-----------|----------|----------|
| **Phrases courtes** (1-5 mots) | "Bonjour", "Oui", "Non merci" | Vérifier le minimum viable |
| **Phrases moyennes** (10-30 mots) | Citations, descriptions | Cas d'usage normal |
| **Phrases longues** (50-200 mots) | Paragraphes, articles | Trouver la limite haute |
| **Textes très longs** (500+ mots) | Pages entières | Doit échouer proprement ou être découpé |
| **Caractères spéciaux** | Émojis, ponctuation, chiffres, URLs | Edge cases |
| **Texte vide / whitespace** | "", " ", "\n" | Ne doit pas crasher |
| **Langues mixtes** | "Hello, bonjour, 你好" | Code-switching |
| **Répétitions rapides** | Même texte 100x d'affilée | Memory leaks, stabilité audio |

### Métriques collectées

Par requête :
- Texte envoyé (tronqué)
- Longueur du texte (chars / mots)
- Statut de la réponse (ok / error / timeout / crash)
- Temps de réponse (ms)
- Message d'erreur si applicable

Agrégées :
- Taux de succès (%)
- Temps moyen / médian / p95 / max
- Nombre de crashs daemon
- Longueur max qui fonctionne
- Longueur à partir de laquelle ça échoue

### Corpus de test

Générer un corpus varié :
- Phrases aléatoires en français (lorem ipsum FR)
- Extraits de livres (domaine public)
- Phrases avec ponctuation variée (!, ?, ..., ;, :)
- Nombres et dates ("le 14 juillet 2025", "3.14159")
- Cas edge : une seule lettre, un seul mot, un paragraphe entier

## 2. Découpage automatique des textes longs

### Problème

Le modèle a une limite de tokens en entrée. Les textes longs font crasher ou bloquer la génération.

### Solution : splitter dans `handlers.py`

Découper le texte en chunks avant génération :

1. Splitter sur les fins de phrases (`. `, `! `, `? `)
2. Si une phrase dépasse la limite, splitter sur les virgules (`, `)
3. Si toujours trop long, splitter sur les espaces
4. Générer et streamer chaque chunk séquentiellement

```python
def split_text(text: str, max_chars: int = 200) -> list[str]:
    """Split text into chunks that the model can handle."""
    ...
```

### Trouver la limite

Le stress test doit déterminer expérimentalement :
- Longueur max en caractères qui fonctionne à 100%
- Longueur à partir de laquelle le taux d'échec augmente
- Le `max_chars` optimal pour le splitter

## 3. Protection du daemon

### PID file / Lock file

```python
PID_PATH = Path.home() / ".q3tts.pid"
```

- Au démarrage : vérifier si un daemon tourne déjà (lire le PID, vérifier si le process existe)
- Écrire le PID dans le fichier
- Supprimer au shutdown

### Timeout sur la génération

Si `model.generate_voice_design()` bloque plus de N secondes, interrompre.

```python
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("generation timed out")

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(30)  # 30 secondes max
try:
    result = handlers.handle(model, request)
finally:
    signal.alarm(0)  # annuler le timer
```

### Watchdog mémoire

Loguer l'usage mémoire après chaque requête. Si ça monte sans redescendre → fuite mémoire.

```python
import resource
mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
log(f"[mem] {mem // 1024}MB")
```

## 4. Rapport de stabilité

Le stress test génère un rapport :

```json
{
  "total_requests": 1000,
  "success": 985,
  "errors": 10,
  "timeouts": 3,
  "crashes": 2,
  "success_rate": "98.5%",
  "avg_response_ms": 1234,
  "p95_response_ms": 3456,
  "max_text_length_ok": 180,
  "min_text_length_fail": 250,
  "memory_start_mb": 4500,
  "memory_end_mb": 4520
}
```

## Checklist d'implémentation

- [ ] Écrire `tests/stress_test.py` avec corpus varié
- [ ] Trouver la limite de longueur du modèle
- [ ] Implémenter le découpage automatique dans `handlers.py`
- [ ] Ajouter PID file dans `q3tts_daemon.py`
- [ ] Ajouter timeout sur la génération
- [ ] Ajouter log mémoire après chaque requête
- [ ] Générer et analyser le premier rapport de stabilité
