"""Local Sovereign Identity Vault (PBKDF2-SHA256) & Multi-Thread Conversation Store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import secrets
import time
from typing import Any
import uuid


class LocalIdentityVault:
    """
    100% Local Sovereign Profile & Session Manager for the Never-AI Desktop App.
    No cloud telemetry, no external servers — passwords hashed locally with PBKDF2-HMAC-SHA256.
    """

    def __init__(self, vault_path: str | Path = "memory_store/identities.json") -> None:
        self.vault_path = Path(vault_path)
        self.vault_path.parent.mkdir(parents=True, exist_ok=True)
        self.users: dict[str, dict[str, Any]] = {}
        self.sessions: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        if self.vault_path.exists():
            data = json.loads(self.vault_path.read_text(encoding="utf-8"))
            self.users = data.get("users", {})
            self.sessions = data.get("sessions", {})

    def _save(self) -> None:
        tmp = self.vault_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"users": self.users, "sessions": self.sessions}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.vault_path)

    @staticmethod
    def _normalize_handle(handle: str) -> str:
        clean = handle.strip().lstrip("@").lower()
        clean = "".join(ch for ch in clean if ch.isalnum() or ch in ("_", "-", "."))
        return clean or "sovereign"

    @staticmethod
    def _hash_passphrase(passphrase: str, salt_hex: str) -> str:
        dk = hashlib.pbkdf2_hmac(
            "sha256",
            passphrase.encode("utf-8"),
            bytes.fromhex(salt_hex),
            120_000,
        )
        return dk.hex()

    @staticmethod
    def _normalize_account_type(account_type: str | None) -> str:
        val = (account_type or "user").strip().lower()
        return "lab" if val in ("lab", "rd", "expert", "developer", "dev") else "user"

    def register(
        self,
        handle: str,
        display_name: str,
        passphrase: str = "",
        account_type: str = "user",
    ) -> dict[str, Any]:
        norm = self._normalize_handle(handle)
        if len(norm) < 2:
            raise ValueError("L'identifiant doit contenir au moins 2 caractères.")
        if norm in self.users:
            raise ValueError(f"Le profil @{norm} existe déjà sur cette machine.")

        acc_type = self._normalize_account_type(account_type)
        salt = secrets.token_hex(16)
        pw_hash = self._hash_passphrase(passphrase, salt) if passphrase else ""
        profile = {
            "handle": norm,
            "display_name": display_name.strip() or f"@{norm}",
            "account_type": acc_type,
            "salt": salt,
            "pw_hash": pw_hash,
            "created_at": round(time.time(), 2),
        }
        self.users[norm] = profile
        token = secrets.token_urlsafe(24)
        self.sessions[token] = norm
        self._save()
        return self._public_profile(profile, token)

    def login(self, handle: str, passphrase: str = "") -> dict[str, Any]:
        norm = self._normalize_handle(handle)
        profile = self.users.get(norm)
        if not profile:
            raise ValueError(f"Aucun profil local @{norm} trouvé. Créez-le d'abord.")
        if profile.get("pw_hash"):
            candidate = self._hash_passphrase(passphrase, profile["salt"])
            if not secrets.compare_digest(candidate, profile["pw_hash"]):
                raise ValueError("Clé locale incorrecte.")
        if "account_type" not in profile:
            profile["account_type"] = "user"
        token = secrets.token_urlsafe(24)
        self.sessions[token] = norm
        self._save()
        return self._public_profile(profile, token)

    def guest_session(self, account_type: str = "user") -> dict[str, Any]:
        acc_type = self._normalize_account_type(account_type)
        norm = "local"
        if norm not in self.users:
            self.users[norm] = {
                "handle": norm,
                "display_name": "Compte Utilisateur" if acc_type == "user" else "Ingénieur Labo",
                "account_type": acc_type,
                "salt": secrets.token_hex(16),
                "pw_hash": "",
                "created_at": round(time.time(), 2),
            }
        else:
            self.users[norm]["account_type"] = acc_type
            self.users[norm]["display_name"] = (
                "Explorateur Never-AI" if acc_type == "user" else "Architecte Labo R&D"
            )
        token = secrets.token_urlsafe(24)
        self.sessions[token] = norm
        self._save()
        return self._public_profile(self.users[norm], token)

    def set_account_type(self, token: str, account_type: str) -> dict[str, Any]:
        norm = self.sessions.get(token)
        if not norm or norm not in self.users:
            raise ValueError("Session invalide ou expirée.")
        acc_type = self._normalize_account_type(account_type)
        self.users[norm]["account_type"] = acc_type
        if norm == "local":
            self.users[norm]["display_name"] = (
                "Explorateur Never-AI" if acc_type == "user" else "Architecte Labo R&D"
            )
        self._save()
        return self._public_profile(self.users[norm], token)

    def resolve_token(self, token: str | None) -> dict[str, Any] | None:
        if not token:
            return None
        norm = self.sessions.get(token)
        if not norm or norm not in self.users:
            return None
        return self._public_profile(self.users[norm], token)

    @staticmethod
    def _public_profile(profile: dict[str, Any], token: str) -> dict[str, Any]:
        acc_type = profile.get("account_type", "user")
        return {
            "handle": profile["handle"],
            "display_name": profile["display_name"],
            "account_type": acc_type,
            "account_label": "Édition Commerciale" if acc_type == "user" else "Laboratoire R&D",
            "created_at": profile["created_at"],
            "token": token,
        }


class ConversationStore:
    """Persistent multi-thread conversation history for the Never-AI Desktop client."""

    def __init__(self, store_path: str | Path = "memory_store/conversations.json") -> None:
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.threads: dict[str, dict[str, Any]] = {}
        self._load()

    def _load(self) -> None:
        if self.store_path.exists():
            self.threads = json.loads(self.store_path.read_text(encoding="utf-8")).get("threads", {})

    def _save(self) -> None:
        tmp = self.store_path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"threads": self.threads}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(self.store_path)

    def create_thread(self, owner: str = "local", title: str = "Nouvelle discussion") -> dict[str, Any]:
        tid = uuid.uuid4().hex[:12]
        now = round(time.time(), 2)
        thread = {
            "id": tid,
            "owner": owner,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "messages": [],
        }
        self.threads[tid] = thread
        self._save()
        return thread

    def list_threads(self, owner: str | None = None) -> list[dict[str, Any]]:
        items = list(self.threads.values())
        if owner:
            items = [t for t in items if t.get("owner") in (owner, "local")]
        items.sort(key=lambda t: t.get("updated_at", 0), reverse=True)
        return [
            {
                "id": t["id"],
                "title": t["title"],
                "owner": t["owner"],
                "updated_at": t["updated_at"],
                "message_count": len(t.get("messages", [])),
            }
            for t in items
        ]

    def get_thread(self, thread_id: str) -> dict[str, Any] | None:
        return self.threads.get(thread_id)

    def delete_thread(self, thread_id: str) -> bool:
        if thread_id in self.threads:
            del self.threads[thread_id]
            self._save()
            return True
        return False

    def append_exchange(
        self,
        thread_id: str,
        user_message: str,
        assistant_payload: dict[str, Any],
        owner: str = "local",
    ) -> dict[str, Any]:
        if thread_id not in self.threads:
            t = self.create_thread(owner=owner)
            thread_id = t["id"]
        thread = self.threads[thread_id]
        if not thread["messages"] and thread["title"] == "Nouvelle discussion":
            snippet = user_message.strip().replace("\n", " ")
            thread["title"] = snippet[:42] + ("…" if len(snippet) > 42 else "")

        now = round(time.time(), 2)
        thread["messages"].append(
            {
                "role": "user",
                "content": user_message,
                "timestamp": now,
            }
        )
        thread["messages"].append(
            {
                "role": "never",
                "content": assistant_payload.get("visible_answer", ""),
                "thoughts": assistant_payload.get("thoughts", []),
                "tool_traces": assistant_payload.get("tool_traces", []),
                "routed_tools": assistant_payload.get("routed_tools", []),
                "artifacts": assistant_payload.get("artifacts", []),
                "citations": assistant_payload.get("citations", []),
                "persona": assistant_payload.get("persona", "polymathe"),
                "deep_think": assistant_payload.get("deep_think", False),
                "titans_mean_surprise": assistant_payload.get("titans_mean_surprise", 0.0),
                "timestamp": now,
            }
        )
        thread["updated_at"] = now
        self._save()
        return thread

    def update_thread_messages(
        self,
        thread_id: str,
        messages: list[dict[str, Any]],
        title: str | None = None,
        owner: str = "local",
    ) -> dict[str, Any]:
        if thread_id not in self.threads:
            now = round(time.time(), 2)
            self.threads[thread_id] = {
                "id": thread_id,
                "owner": owner,
                "title": title or "Discussion",
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
        thread = self.threads[thread_id]
        if title:
            thread["title"] = title[:80]
        thread["messages"] = messages
        thread["updated_at"] = round(time.time(), 2)
        self._save()
        return thread

    def fork_thread(
        self,
        thread_id: str,
        up_to_index: int | None = None,
        owner: str = "local",
    ) -> dict[str, Any] | None:
        src = self.threads.get(thread_id)
        if not src:
            return None
        msgs = list(src.get("messages", []))
        if up_to_index is not None:
            msgs = msgs[: max(0, int(up_to_index) + 1)]
        new_t = self.create_thread(
            owner=owner or src.get("owner", "local"),
            title=f"⎇ {src.get('title', 'Discussion')[:36]}",
        )
        new_t["messages"] = json.loads(json.dumps(msgs, ensure_ascii=False))
        new_t["forked_from"] = thread_id
        new_t["updated_at"] = round(time.time(), 2)
        self._save()
        return new_t

