"""Dev tunnel: gives the server on this laptop a public https address so Twilio's servers can reach it.
Twilio calls US (a webhook when the parent answers, then a WebSocket carrying the audio), so it needs an address on the internet.
ngrok keeps an outbound connection open from this machine to ngrok's servers and forwards whatever arrives at the public URL down it."""
import re
from pathlib import Path


def set_env_value(path: Path, key: str, value: str) -> None:
    """Set KEY=value in a .env file, replacing the existing line (or appending one). Other lines are untouched."""
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    line = f"{key}={value}"
    if re.search(rf"^{re.escape(key)}=.*$", text, flags=re.M):
        text = re.sub(rf"^{re.escape(key)}=.*$", lambda _: line, text, count=1, flags=re.M)
    else:
        text += ("" if text.endswith("\n") or not text else "\n") + line + "\n"
    path.write_text(text, encoding="utf-8")


def open_tunnel(port: int, authtoken: str) -> str:
    """Start an ngrok tunnel to localhost:port and return its public https URL."""
    from pyngrok import conf, ngrok

    if not authtoken:
        raise SystemExit("NGROK_AUTHTOKEN is empty. Sign up free at dashboard.ngrok.com, copy your authtoken into .env, and run again.")
    conf.get_default().auth_token = authtoken
    return ngrok.connect(port, "http").public_url.replace("http://", "https://")
