# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "click",
# ]
# ///
"""q3tts client — sends requests to the daemon over Unix socket."""

import json
import socket
import struct
import sys
from pathlib import Path

import click

SOCKET_PATH = Path.home() / ".q3tts.sock"


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


@click.group(invoke_without_command=True)
@click.argument("text", required=False)
@click.option("-o", "--output", default=None, help="Save audio to file")
@click.option("-l", "--language", default="English", help="Language (default: English)")
@click.option("-i", "--instruct", default=None, help="Voice instruction")
@click.pass_context
def cli(ctx, text, output, language, instruct):
    """q3tts — TTS client for the q3tts daemon."""
    if ctx.invoked_subcommand is not None:
        return

    # No subcommand — treat as text generation
    if text is None:
        if not sys.stdin.isatty():
            text = sys.stdin.read().strip()
        else:
            click.echo("Usage: q3tts_client.py [OPTIONS] TEXT")
            click.echo("       q3tts_client.py serve")
            click.echo("       q3tts_client.py stop")
            click.echo("       q3tts_client.py status")
            sys.exit(1)

    if not text:
        click.echo("Error: text cannot be empty.", err=True)
        sys.exit(1)

    if not daemon_is_running():
        click.echo("Error: daemon is not running. Start it with: uv run src/q3tts_daemon.py", err=True)
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
def serve():
    """Start the daemon (runs q3tts_daemon.py)."""
    import subprocess
    daemon_script = Path(__file__).parent / "q3tts_daemon.py"
    click.echo(f"Starting daemon: {daemon_script}")
    subprocess.run([sys.executable, str(daemon_script)])


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


if __name__ == "__main__":
    cli()
