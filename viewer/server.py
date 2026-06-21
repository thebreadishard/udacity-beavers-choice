"""
viewer/server.py
================

A tiny, dependency-free HTTP server that powers the Beaver's Choice **pixel office**
viewer. It serves the static front-end (``index.html`` / ``style.css`` / ``app.js``) and
exposes a single JSON endpoint that tails the agent transcript:

    GET /events?since=N  ->  {"events": [...], "total": M}

``N`` is the number of events the client has already seen, so the client polls with the
running count and only receives new lines. The transcript itself is produced by
``agent_transcript.py`` (one JSON object per agent state change).

Run it::

    python viewer/server.py                 # serves http://127.0.0.1:8000
    python viewer/server.py --port 9000
    python viewer/server.py --transcript path/to/transcript.jsonl

Then open the printed URL (or VS Code's Simple Browser) and run the multi-agent system in
another terminal. The viewer supports a **live** mode (tail while agents run) and a
**replay** mode (play a finished transcript back), both selectable in the UI.

The server is read-only: it never writes to or modifies the transcript.
"""

from __future__ import annotations

import argparse
import json
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# Directory containing the static front-end (this file's folder).
VIEWER_DIR = Path(__file__).resolve().parent

# Static assets we are willing to serve, mapped to their content types. Restricting to an
# explicit allow-list avoids any path-traversal / arbitrary-file-read concerns.
STATIC_FILES = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "application/javascript; charset=utf-8"),
}


def _read_events(transcript_path: Path, since: int) -> tuple[list, int]:
    """
    Read transcript events, returning ``(new_events, total)``.

    Each non-empty line of the transcript is a JSON object. ``since`` is the number of
    events the caller already has; only events at index ``>= since`` are returned. Malformed
    lines are skipped so a half-written final line never breaks the viewer.

    Args:
        transcript_path: Path to the ``.jsonl`` transcript file.
        since: Count of events the client has already received.

    Returns:
        A tuple of (list of new event dicts, total event count seen so far).
    """
    if not transcript_path.exists():
        return [], 0

    events: list = []
    try:
        with transcript_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    # A line may be mid-write while an agent run is in progress; skip it.
                    continue
    except OSError:
        return [], 0

    total = len(events)
    if since < 0:
        since = 0
    return events[since:], total


class ViewerHandler(BaseHTTPRequestHandler):
    """Serves the static viewer assets and the ``/events`` transcript feed."""

    # Set on the server instance in ``main`` and read here.
    transcript_path: Path = Path("transcript.jsonl")

    def log_message(self, fmt, *args):  # noqa: D401 - quiet the default noisy logging
        """Suppress per-request console spam; keep the terminal readable during demos."""
        return

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_static(self, filename: str, content_type: str) -> None:
        file_path = VIEWER_DIR / filename
        try:
            body = file_path.read_bytes()
        except OSError:
            self.send_error(404, "Not found")
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - required name from BaseHTTPRequestHandler
        parsed = urlparse(self.path)
        route = parsed.path

        if route == "/events":
            params = parse_qs(parsed.query)
            try:
                since = int(params.get("since", ["0"])[0])
            except (ValueError, IndexError):
                since = 0
            new_events, total = _read_events(self.transcript_path, since)
            self._send_json({"events": new_events, "total": total})
            return

        static = STATIC_FILES.get(route)
        if static is not None:
            self._send_static(*static)
            return

        self.send_error(404, "Not found")


def main() -> None:
    parser = argparse.ArgumentParser(description="Beaver's Choice pixel-office viewer.")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default 127.0.0.1).")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind (default 8000).")
    parser.add_argument(
        "--transcript",
        default=str(VIEWER_DIR.parent / "transcript.jsonl"),
        help="Path to the transcript.jsonl file (default: workspace root).",
    )
    parser.add_argument(
        "--no-open", action="store_true", help="Do not auto-open the browser."
    )
    args = parser.parse_args()

    ViewerHandler.transcript_path = Path(args.transcript).resolve()

    server = ThreadingHTTPServer((args.host, args.port), ViewerHandler)
    url = f"http://{args.host}:{args.port}/"
    print(f"Pixel office viewer running at {url}")
    print(f"Tailing transcript: {ViewerHandler.transcript_path}")
    print("Press Ctrl+C to stop.")

    if not args.no_open:
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down viewer.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
