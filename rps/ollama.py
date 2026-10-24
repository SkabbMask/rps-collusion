"""Minimal Ollama chat client (stdlib only)."""

import json
import urllib.error
import urllib.request
from urllib.parse import urlparse


def normalize_url(url):
    """Accept 'ip:port', 'http://ip:port' or a full 'http://ip:port/path' and return a chat endpoint URL."""
    if "://" not in url:
        url = "http://" + url
    if urlparse(url).path in ("", "/"):
        url = url.rstrip("/") + "/api/chat"
    return url


class OllamaClient:
    def __init__(self, url, model, options=None, think=None, timeout=120):
        self.url = normalize_url(url)
        self.model = model
        self.options = options or {}
        self.think = think
        self.timeout = timeout

    def chat(self, messages, schema=None):
        body = {"model": self.model, "messages": messages, "stream": False, "options": self.options}
        if schema is not None:
            body["format"] = schema
        if self.think is not None:
            body["think"] = self.think
        req = urllib.request.Request(
            self.url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read())["message"]["content"]
        except urllib.error.HTTPError as e:
            raise OllamaError(f"{self.url} ({self.model}): HTTP {e.code}: {e.read().decode(errors='replace')}") from None

    def check(self):
        """Fail early if the server is unreachable or doesn't have the model (only for standard /api/chat URLs)."""
        if not self.url.endswith("/api/chat"):
            return
        tags_url = self.url[: -len("chat")] + "tags"
        try:
            with urllib.request.urlopen(tags_url, timeout=10) as resp:
                available = [m["name"] for m in json.loads(resp.read())["models"]]
        except (urllib.error.URLError, OSError) as e:
            raise OllamaError(f"cannot reach Ollama at {tags_url}: {e}") from None
        names = set(available) | {n.removesuffix(":latest") for n in available}
        if self.model not in names:
            raise OllamaError(f"model '{self.model}' not found at {tags_url}; available: {', '.join(available)}")


class OllamaError(RuntimeError):
    pass
