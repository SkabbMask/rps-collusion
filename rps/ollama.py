"""Minimal Ollama chat client (stdlib only)."""

import json
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
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())["message"]["content"]
