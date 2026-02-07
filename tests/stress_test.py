"""Stress test for jarvis daemon — sends varied requests and collects metrics."""

import json
import sys
import time
from pathlib import Path

import click

from jarvis.cli import SOCKET_PATH, send_request


# --- Corpus ---

SHORT = [
    "Bonjour.",
    "Oui.",
    "Non merci.",
    "Salut !",
    "D'accord.",
    "Parfait.",
    "Bonne nuit.",
    "À demain.",
    "Pourquoi ?",
    "Bien sûr.",
]

MEDIUM = [
    "Le daemon fonctionne correctement et répond à toutes les requêtes.",
    "Il fait beau aujourd'hui, le soleil brille et les oiseaux chantent.",
    "La programmation est un art qui demande patience et persévérance.",
    "Je voudrais commander un café noir avec un croissant s'il vous plaît.",
    "Les résultats du test montrent une amélioration significative.",
    "Le train de Paris arrive à quatorze heures trente sur le quai numéro trois.",
    "Cette nouvelle fonctionnalité permet de générer de la parole en temps réel.",
    "N'oubliez pas de sauvegarder votre travail avant de quitter.",
    "Le match de football commence dans une heure au stade municipal.",
    "La réunion est reportée à vendredi prochain à dix heures.",
]

LONG = [
    "Aujourd'hui nous avons implémenté un daemon qui garde le modèle en mémoire et accepte des requêtes via un socket Unix. Cela élimine cinq secondes de latence à chaque appel. Le système utilise un mécanisme de hot-reload qui permet de modifier le code sans redémarrer le processus.",
    "La France est un pays situé en Europe de l'Ouest. Sa capitale est Paris, une ville connue pour sa tour Eiffel, ses musées et sa gastronomie. Le pays compte environ soixante-sept millions d'habitants et possède une histoire riche qui remonte à des milliers d'années.",
    "Pour préparer une bonne ratatouille, il faut des courgettes, des aubergines, des poivrons, des tomates, de l'ail et des oignons. On commence par couper tous les légumes en rondelles, puis on les fait revenir dans de l'huile d'olive. On ajoute les herbes de Provence et on laisse mijoter pendant une heure.",
]

VERY_LONG = [
    "L'intelligence artificielle est un domaine de l'informatique qui vise à créer des systèmes capables de réaliser des tâches qui nécessitent normalement l'intelligence humaine. " * 5,
]

EDGE_CASES = [
    "",
    " ",
    "A",
    "123456789",
    "Hello, bonjour, 你好 !",
    "Il a dit : « C'est incroyable ! » — et il avait raison...",
    "3.14159 euros, soit 50% de réduction !",
    "test@email.com",
    "Aaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    "😀 😂 🎉",
]

REPEAT = ["Test de répétition."] * 20


def build_corpus() -> list[dict]:
    """Build the full test corpus with metadata."""
    corpus = []
    for text in SHORT:
        corpus.append({"text": text, "category": "short"})
    for text in MEDIUM:
        corpus.append({"text": text, "category": "medium"})
    for text in LONG:
        corpus.append({"text": text, "category": "long"})
    for text in VERY_LONG:
        corpus.append({"text": text, "category": "very_long"})
    for text in EDGE_CASES:
        corpus.append({"text": text, "category": "edge_case"})
    for text in REPEAT:
        corpus.append({"text": text, "category": "repeat"})
    return corpus


# --- Test runner ---

def run_test(entry: dict, index: int, total: int, silent: bool) -> dict:
    """Run a single test and return the result."""
    import socket as _socket

    text = entry["text"]
    category = entry["category"]
    char_count = len(text)
    word_count = len(text.split()) if text.strip() else 0

    result = {
        "index": index,
        "category": category,
        "text_preview": text[:60] + ("..." if len(text) > 60 else ""),
        "char_count": char_count,
        "word_count": word_count,
        "status": None,
        "response_ms": None,
        "error": None,
    }

    t0 = time.time()
    try:
        request = {
            "action": "generate",
            "text": text,
            "language": "French",
            "instruct": "warm masculine voice",
            "output": "/dev/null" if silent else None,
        }
        resp = send_request(request, timeout=120)
        elapsed = (time.time() - t0) * 1000
        result["response_ms"] = round(elapsed)
        result["status"] = resp.get("status", "unknown")
        if resp.get("status") != "ok":
            result["error"] = resp.get("message", "unknown error")
    except _socket.timeout:
        elapsed = (time.time() - t0) * 1000
        result["response_ms"] = round(elapsed)
        result["status"] = "timeout"
        result["error"] = "socket timeout"
    except ConnectionError as e:
        elapsed = (time.time() - t0) * 1000
        result["response_ms"] = round(elapsed)
        result["status"] = "crash"
        result["error"] = str(e)
    except OSError as e:
        elapsed = (time.time() - t0) * 1000
        result["response_ms"] = round(elapsed)
        result["status"] = "connection_error"
        result["error"] = str(e)

    # Print progress
    status_icon = "✓" if result["status"] == "ok" else "✗"
    click.echo(
        f"  [{index+1}/{total}] {status_icon} {category:12s} "
        f"{result['response_ms']:>6}ms  "
        f"{char_count:>4} chars  "
        f"{result['text_preview']}"
    )

    return result


def generate_report(results: list[dict]) -> dict:
    """Generate aggregate statistics from test results."""
    total = len(results)
    successes = [r for r in results if r["status"] == "ok"]
    errors = [r for r in results if r["status"] == "error"]
    timeouts = [r for r in results if r["status"] == "timeout"]
    crashes = [r for r in results if r["status"] == "crash"]
    conn_errors = [r for r in results if r["status"] == "connection_error"]

    success_times = [r["response_ms"] for r in successes if r["response_ms"] is not None]

    # Find length limits
    ok_chars = [r["char_count"] for r in successes]
    fail_chars = [r["char_count"] for r in results if r["status"] != "ok" and r["char_count"] > 0]

    # Per-category stats
    categories = {}
    for r in results:
        cat = r["category"]
        if cat not in categories:
            categories[cat] = {"total": 0, "ok": 0, "errors": 0, "times": []}
        categories[cat]["total"] += 1
        if r["status"] == "ok":
            categories[cat]["ok"] += 1
        else:
            categories[cat]["errors"] += 1
        if r["response_ms"] is not None:
            categories[cat]["times"].append(r["response_ms"])

    for cat in categories:
        times = categories[cat]["times"]
        if times:
            times.sort()
            categories[cat]["avg_ms"] = round(sum(times) / len(times))
            categories[cat]["median_ms"] = round(times[len(times) // 2])
            categories[cat]["p95_ms"] = round(times[int(len(times) * 0.95)])
            categories[cat]["max_ms"] = max(times)
        del categories[cat]["times"]

    report = {
        "total_requests": total,
        "success": len(successes),
        "errors": len(errors),
        "timeouts": len(timeouts),
        "crashes": len(crashes),
        "connection_errors": len(conn_errors),
        "success_rate": f"{len(successes) / total * 100:.1f}%" if total > 0 else "N/A",
        "response_times": {},
        "text_length": {},
        "per_category": categories,
        "failed_details": [
            {"index": r["index"], "category": r["category"], "chars": r["char_count"],
             "status": r["status"], "error": r["error"], "text_preview": r["text_preview"]}
            for r in results if r["status"] != "ok"
        ],
    }

    if success_times:
        success_times.sort()
        report["response_times"] = {
            "avg_ms": round(sum(success_times) / len(success_times)),
            "median_ms": round(success_times[len(success_times) // 2]),
            "p95_ms": round(success_times[int(len(success_times) * 0.95)]),
            "max_ms": max(success_times),
            "min_ms": min(success_times),
        }

    if ok_chars:
        report["text_length"]["max_chars_ok"] = max(ok_chars)
    if fail_chars:
        report["text_length"]["min_chars_fail"] = min(fail_chars)

    return report


def run_stress(silent: bool, delay: float, report_path: str, category: str | None):
    """Run stress tests against the jarvis daemon."""

    if not SOCKET_PATH.exists():
        click.echo("Error: daemon not running. Start with: jah serve", err=True)
        sys.exit(1)

    corpus = build_corpus()
    if category:
        corpus = [e for e in corpus if e["category"] == category]

    total = len(corpus)
    click.echo(f"Running {total} tests (delay={delay}s, silent={silent})...")
    click.echo()

    results = []
    for i, entry in enumerate(corpus):
        result = run_test(entry, i, total, silent)
        results.append(result)
        if delay > 0 and i < total - 1:
            time.sleep(delay)

    # Generate report
    report = generate_report(results)

    click.echo()
    click.echo("=" * 60)
    click.echo("STABILITY REPORT")
    click.echo("=" * 60)
    click.echo(f"  Total:      {report['total_requests']}")
    click.echo(f"  Success:    {report['success']}")
    click.echo(f"  Errors:     {report['errors']}")
    click.echo(f"  Timeouts:   {report['timeouts']}")
    click.echo(f"  Crashes:    {report['crashes']}")
    click.echo(f"  Conn err:   {report['connection_errors']}")
    click.echo(f"  Rate:       {report['success_rate']}")
    click.echo()

    if report["response_times"]:
        rt = report["response_times"]
        click.echo(f"  Avg:        {rt['avg_ms']}ms")
        click.echo(f"  Median:     {rt['median_ms']}ms")
        click.echo(f"  P95:        {rt['p95_ms']}ms")
        click.echo(f"  Max:        {rt['max_ms']}ms")
        click.echo()

    if report["text_length"]:
        tl = report["text_length"]
        if "max_chars_ok" in tl:
            click.echo(f"  Max chars OK:   {tl['max_chars_ok']}")
        if "min_chars_fail" in tl:
            click.echo(f"  Min chars fail: {tl['min_chars_fail']}")
        click.echo()

    click.echo("Per category:")
    for cat, stats in report["per_category"].items():
        rate = f"{stats['ok']}/{stats['total']}"
        avg = f"{stats.get('avg_ms', '?')}ms" if 'avg_ms' in stats else "?"
        click.echo(f"  {cat:12s}  {rate:>6}  avg {avg}")
    click.echo()

    if report["failed_details"]:
        click.echo("Failed requests:")
        for f in report["failed_details"]:
            click.echo(f"  [{f['index']}] {f['category']} ({f['chars']} chars) {f['status']}: {f['error']}")
            click.echo(f"       {f['text_preview']}")
        click.echo()

    # Save report
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    click.echo(f"Report saved to {report_path}")
