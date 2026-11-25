"""Lightweight non-blocking Telegram Bot API notifier (pure stdlib + requests)."""
from __future__ import annotations

import os
import threading
from pathlib import Path

import requests


def load_local_env(paths: list[Path]) -> None:
    """Load KEY=VALUE pairs from local .env files if not already set in os.environ."""
    for p in paths:
        if not p.exists():
            continue
        try:
            for line in p.read_text(encoding="utf-8").splitlines():
                s = line.strip()
                if not s or s.startswith("#") or "=" not in s:
                    continue
                k, v = s.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")
                if k and k not in os.environ and v:
                    os.environ[k] = v
        except OSError:
            pass


class TelegramNotifier:
    """Thread-safe async Telegram notifier reading TELEGRAM_BOT_TOKEN & TELEGRAM_CHAT_ID."""

    def __init__(self) -> None:
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    @property
    def enabled(self) -> bool:
        return bool(self.token and self.chat_id)

    def reload_from_env(self, env_paths: list[Path]) -> None:
        # Re-read local .env files so user can edit quant_lab/.env without even restarting!
        for p in env_paths:
            if not p.exists():
                continue
            try:
                for line in p.read_text(encoding="utf-8").splitlines():
                    s = line.strip()
                    if not s or s.startswith("#") or "=" not in s:
                        continue
                    k, v = s.split("=", 1)
                    k, v = k.strip(), v.strip().strip('"').strip("'")
                    if k in ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID") and v:
                        os.environ[k] = v
            except OSError:
                pass
        self.token = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
        self.chat_id = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

    def send_sync(self, text: str) -> tuple[bool, str]:
        if not self.enabled:
            return False, "TELEGRAM_BOT_TOKEN veya TELEGRAM_CHAT_ID tanımlı değil."
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            r = requests.post(url, json=payload, timeout=10)
            if r.status_code == 200 and r.json().get("ok"):
                return True, "Telegram mesajı gönderildi."
            return False, f"Telegram API HTTP {r.status_code}: {r.text[:160]}"
        except Exception as exc:
            return False, f"Bağlantı hatası: {type(exc).__name__}: {exc}"

    def send_async(self, text: str) -> None:
        if not self.enabled:
            return
        threading.Thread(target=self.send_sync, args=(text,), daemon=True, name="tg-send").start()
