from rag.auth import ApiKeyRecord, create_api_key, load_api_keys, revoke_api_key
from rag.config import settings


def test_authorizes_exact_company_match():
    record = ApiKeyRecord(name="acme bot", companies=["acme"])
    assert record.authorizes("acme")
    assert not record.authorizes("other-co")


def test_authorizes_wildcard():
    record = ApiKeyRecord(name="admin", companies=["*"])
    assert record.authorizes("acme")
    assert record.authorizes("anything")


def test_create_load_revoke_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "api_keys_file", tmp_path / "api_keys.json")

    key = create_api_key(name="acme bot", companies=["acme"])
    assert key.startswith("sk-")

    loaded = load_api_keys()
    assert key in loaded
    assert loaded[key].name == "acme bot"
    assert loaded[key].companies == ["acme"]

    assert revoke_api_key(key) is True
    assert key not in load_api_keys()
    assert revoke_api_key(key) is False


def test_load_api_keys_missing_file_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "api_keys_file", tmp_path / "does_not_exist.json")
    assert load_api_keys() == {}
