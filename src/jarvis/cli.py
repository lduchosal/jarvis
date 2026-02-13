"""jah — CLI client for the jarvis TTS daemon."""

import json
import socket
import struct
import sys
from pathlib import Path

import click

SOCKET_PATH = Path.home() / ".q3tts.sock"

SUBCOMMANDS = {"serve", "stop", "status", "stress", "listen", "echo", "talk", "panel"}


def send_message(sock: socket.socket, msg: dict):
    payload = json.dumps(msg).encode("utf-8")
    sock.sendall(struct.pack("!I", len(payload)) + payload)


def read_message(sock: socket.socket) -> dict:
    raw_len = b""
    while len(raw_len) < 4:
        chunk = sock.recv(4 - len(raw_len))
        if not chunk:
            raise ConnectionError("daemon disconnected")
        raw_len += chunk
    msg_len = struct.unpack("!I", raw_len)[0]

    data = b""
    while len(data) < msg_len:
        chunk = sock.recv(msg_len - len(data))
        if not chunk:
            raise ConnectionError("daemon disconnected")
        data += chunk
    return json.loads(data.decode("utf-8"))


def daemon_is_running() -> bool:
    """Check if daemon is running by testing if socket file exists."""
    return SOCKET_PATH.exists()


def send_request(request: dict, timeout: float = 120) -> dict:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(str(SOCKET_PATH))
        send_message(s, request)
        return read_message(s)


class JahGroup(click.Group):
    """Custom group that allows `jah "text"` without a subcommand."""

    def parse_args(self, ctx, args):
        # If first arg looks like text (not a subcommand), insert "speak" command
        if args and args[0] not in SUBCOMMANDS and not args[0].startswith("-"):
            args = ["speak"] + args
        return super().parse_args(ctx, args)


@click.group(cls=JahGroup)
def cli():
    """jah — Jarvis voice assistant."""
    pass


@cli.command()
@click.argument("text", required=False)
@click.option("-o", "--output", default=None, help="Save audio to file")
@click.option("-l", "--language", default="English", help="Language (default: English)")
@click.option("-i", "--instruct", default=None, help="Voice instruction")
def speak(text, output, language, instruct):
    """Generate speech from text."""
    # Piped input always wins over positional argument
    if not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    elif text is None:
        click.echo("Usage: jah \"text to speak\"")
        sys.exit(1)

    if not text:
        click.echo("Error: text cannot be empty.", err=True)
        sys.exit(1)

    if not daemon_is_running():
        click.echo("Error: daemon is not running. Start it with: jah serve", err=True)
        sys.exit(1)

    request = {
        "action": "generate",
        "text": text,
        "language": language,
        "instruct": instruct,
        "output": output,
    }

    resp = send_request(request)
    if resp.get("status") == "ok":
        click.echo("ok")
    else:
        click.echo(f"Error: {resp.get('message', 'unknown error')}", err=True)
        sys.exit(1)


@cli.command()
@click.option("-m", "--model", default=None, help="TTS model: 1.7b (default) or 0.6b")
@click.option("-w", "--workers", default=3, type=int, help="Number of parallel TTS workers (default: 3)")
def serve(model, workers):
    """Start the TTS daemon."""
    from jarvis.daemon import main as daemon_main
    daemon_main(model_name=model, n_workers=workers)


@cli.command()
def stop():
    """Stop the running daemon."""
    if not daemon_is_running():
        click.echo("Daemon is not running.")
        return

    resp = send_request({"action": "shutdown"})
    if resp.get("status") == "ok":
        click.echo("Daemon stopped.")
    else:
        click.echo(f"Error: {resp.get('message', 'unknown')}", err=True)


@cli.command()
def status():
    """Check if the daemon is running."""
    if daemon_is_running():
        click.echo("Daemon is running.")
    else:
        click.echo("Daemon is not running.")


@cli.command()
@click.option("--silent", is_flag=True, help="Skip audio playback (output to /dev/null)")
@click.option("--delay", default=0.5, help="Delay between requests in seconds")
@click.option("--report", default="tests/stability_report.json", help="Path for JSON report")
@click.option("--category", default=None, help="Only run tests from this category")
def stress(silent, delay, report, category):
    """Run stress tests against the daemon."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "stress_test",
        Path(__file__).parent.parent.parent / "tests" / "stress_test.py",
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.run_stress(silent=silent, delay=delay, report_path=report, category=category)


@cli.command()
@click.option("--duration", type=float, default=None, help="Max listen duration in seconds")
def listen(duration):
    """Real-time speech-to-text from microphone."""
    from jarvis.stt import load_model, listen as stt_listen

    click.echo("Loading STT model...", err=True)
    bundle = load_model()
    click.echo("Model loaded.", err=True)
    stt_listen(bundle, duration=duration)


@cli.command(name="echo")
@click.option("-l", "--language", default="French", help="Language for TTS (default: French)")
@click.option("-i", "--instruct", default=None, help="Voice instruction for TTS")
def echo_cmd(language, instruct):
    """Listen to mic, then repeat back via TTS."""
    from jarvis.stt import load_model, listen_until_silence, reset

    if not daemon_is_running():
        click.echo("Error: TTS daemon not running. Start with: jah serve", err=True)
        sys.exit(1)

    click.echo("Loading STT model...", err=True)
    bundle = load_model()
    click.echo("Ready. Speak now.", err=True)

    while True:
        try:
            text = listen_until_silence(bundle)
            if not text:
                continue
            click.echo(f"> {text}", err=True)
            resp = send_request({
                "action": "generate",
                "text": text,
                "language": language,
                "instruct": instruct,
            })
            if resp.get("status") != "ok":
                click.echo(f"TTS error: {resp.get('message')}", err=True)
            reset(bundle)
        except KeyboardInterrupt:
            break

    click.echo("\nDone.", err=True)


@cli.command()
@click.option("-l", "--language", default="French", help="Language for TTS (default: French)")
@click.option("-m", "--model", default=None, help="Claude model (e.g. haiku, sonnet, opus)")
def talk(language, model):
    """Voice conversation with Claude Code (STT -> Claude -> TTS)."""
    if not daemon_is_running():
        click.echo("Error: TTS daemon not running. Start with: jah serve", err=True)
        sys.exit(1)

    import asyncio
    from jarvis.talk import run_talk
    asyncio.run(run_talk(language=language, model=model))


@cli.command()
@click.option("-l", "--language", default="French", help="Language for TTS (default: French)")
@click.option("--tts/--no-tts", default=False, help="Read responses aloud via TTS daemon")
@click.option("-r", "--resume", default=None, help="Resume session: 'latest' or session name")
def panel(language, tts, resume):
    """Multi-model debate: Claude 4.6 + Codex 5.3 + Gemini 2.5."""
    if tts and not daemon_is_running():
        click.echo("Error: --tts requires TTS daemon. Start with: jah serve", err=True)
        sys.exit(1)

    import asyncio
    from jarvis.panel import run_panel
    asyncio.run(run_panel(language=language, tts=tts, resume=resume))


if __name__ == "__main__":
    cli()
