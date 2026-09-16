from __future__ import annotations

import json
import secrets
from pathlib import Path

from pydantic import BaseModel

from rag.config import settings


class ApiKeyRecord(BaseModel):
    name: str
    companies: list[str]
    """List of company_ids this key may act on. '*' authorizes every company."""

    def authorizes(self, company_id: str) -> bool:
        return "*" in self.companies or company_id in self.companies


def load_api_keys() -> dict[str, ApiKeyRecord]:
    path = settings.api_keys_file
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {key: ApiKeyRecord.model_validate(record) for key, record in raw.items()}


def save_api_keys(keys: dict[str, ApiKeyRecord]) -> None:
    path = settings.api_keys_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({k: v.model_dump() for k, v in keys.items()}, indent=2), encoding="utf-8"
    )


def create_api_key(name: str, companies: list[str]) -> str:
    keys = load_api_keys()
    new_key = f"sk-{secrets.token_urlsafe(32)}"
    keys[new_key] = ApiKeyRecord(name=name, companies=companies)
    save_api_keys(keys)
    return new_key


def revoke_api_key(api_key: str) -> bool:
    keys = load_api_keys()
    if api_key not in keys:
        return False
    del keys[api_key]
    save_api_keys(keys)
    return True
