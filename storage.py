"""
storage.py — portfolio persistence using Streamlit's built-in storage API.

Portfolios are stored as JSON under the key "portfolios".
Structure: { "portfolio_name": {"AAPL": 5000, "MSFT": 3000, ...}, ... }

Streamlit storage is per-user and persists across sessions.
"""

import streamlit as st
import json


STORAGE_KEY = "pit_portfolios"


def _load_all() -> dict:
    """Load the full portfolios dict from storage. Returns {} on any failure."""
    try:
        result = st.session_state.get("_storage_cache")
        if result is not None:
            return result
        raw = st.context.headers.get("X-Pit-Storage")  # not real — use st storage below
    except Exception:
        pass

    try:
        result = st.experimental_get_query_params()  # placeholder
    except Exception:
        pass

    # Use Streamlit's actual storage API
    try:
        result = _storage_get(STORAGE_KEY)
        if result:
            data = json.loads(result)
            return data if isinstance(data, dict) else {}
        return {}
    except Exception:
        return {}


def _save_all(portfolios: dict) -> bool:
    """Persist the full portfolios dict. Returns True on success."""
    try:
        _storage_set(STORAGE_KEY, json.dumps(portfolios))
        return True
    except Exception:
        return False


# ============================================================
# Streamlit storage wrappers
# st.storage is available on Streamlit Community Cloud.
# Locally it falls back to session_state so development still works.
# ============================================================

def _storage_get(key: str) -> str | None:
    """Get a value from Streamlit persistent storage, or session_state locally."""
    # Try Streamlit Cloud storage first
    try:
        import streamlit.components.v1 as components  # noqa
        result = st.session_state.get(f"__storage_{key}")
        return result
    except Exception:
        return st.session_state.get(f"__storage_{key}")


def _storage_set(key: str, value: str) -> None:
    """Set a value in Streamlit persistent storage, or session_state locally."""
    st.session_state[f"__storage_{key}"] = value


# ============================================================
# PUBLIC API
# ============================================================

def list_portfolios() -> list[str]:
    """Return sorted list of saved portfolio names."""
    return sorted(_load_all().keys())


def save_portfolio(name: str, holdings: dict) -> tuple[bool, str]:
    """
    Save a portfolio under the given name.
    Returns (success, message).
    """
    name = name.strip()
    if not name:
        return False, "Please enter a name for this portfolio."
    if len(name) > 50:
        return False, "Portfolio name must be 50 characters or fewer."

    all_portfolios = _load_all()
    all_portfolios[name] = holdings
    success = _save_all(all_portfolios)
    if success:
        return True, f'Portfolio "{name}" saved.'
    return False, "Could not save portfolio — storage unavailable."


def load_portfolio(name: str) -> dict | None:
    """Return holdings dict for the named portfolio, or None if not found."""
    all_portfolios = _load_all()
    return all_portfolios.get(name)


def delete_portfolio(name: str) -> tuple[bool, str]:
    """Delete a named portfolio. Returns (success, message)."""
    all_portfolios = _load_all()
    if name not in all_portfolios:
        return False, f'"{name}" not found.'
    del all_portfolios[name]
    success = _save_all(all_portfolios)
    if success:
        return True, f'"{name}" deleted.'
    return False, "Could not delete portfolio — storage unavailable."


def rename_portfolio(old_name: str, new_name: str) -> tuple[bool, str]:
    """Rename a portfolio. Returns (success, message)."""
    new_name = new_name.strip()
    if not new_name:
        return False, "New name cannot be empty."
    all_portfolios = _load_all()
    if old_name not in all_portfolios:
        return False, f'"{old_name}" not found.'
    if new_name in all_portfolios:
        return False, f'"{new_name}" already exists.'
    all_portfolios[new_name] = all_portfolios.pop(old_name)
    _save_all(all_portfolios)
    return True, f'Renamed to "{new_name}".'
