"""Digit-only .env values reach every text setting as the text the operator wrote.

lib_layered_config's dotenv and environment layers turn an unquoted value into an int or float
whenever that number reads back as the same text (``2026`` stays ``2026``). A pydantic ``str``
field is strict about type, so every model that validates a configuration section must turn such
a number back into its digits; otherwise an ordinary ``.env`` line such as ``INDEX__COLLECTION=2026``
or ``EMBEDDING__API_KEY=123456`` stops the program.

These tests load a real ``.env`` through the real loader, so a change in the library's coercion
shows up here rather than on a host.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import pytest
from pydantic import SecretStr, ValidationError

from semdex.adapters.config.chunker import get_chunker_config
from semdex.adapters.config.embedding import get_embedding_config
from semdex.adapters.config.extractor import get_extractor_config
from semdex.adapters.config.health import get_health_config
from semdex.adapters.config.index import get_index_config
from semdex.adapters.config.loader import get_config
from semdex.adapters.config.mcp import get_mcp_config
from semdex.adapters.config.sanitize import get_sanitize_config
from semdex.adapters.config.summary import get_summary_config
from semdex.adapters.config.vectorstore import get_vector_store_config
from semdex.adapters.email.config import load_email_config_from_dict
from semdex.adapters.logging.setup import (
    _build_runtime_config,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

    from lib_layered_config import Config


@dataclass(frozen=True)
class _DigitCase:
    """One text setting fed a digit-only ``.env`` value."""

    section: str
    key: str
    raw: str
    reader: Callable[[Config], Any]


_DIGIT_CASES = [
    _DigitCase("index", "collection", "2026", get_index_config),
    _DigitCase("index", "default_label", "7", get_index_config),
    _DigitCase("embedding", "api_key", "123456", get_embedding_config),
    _DigitCase("embedding", "model", "3", get_embedding_config),
    _DigitCase("summary", "api_key", "123456", get_summary_config),
    _DigitCase("extractor", "ocr_language", "1", get_extractor_config),
    _DigitCase("chunker", "semantic_model", "5", get_chunker_config),
    _DigitCase("health", "keep_alive", "300", get_health_config),
    _DigitCase("mcp", "host", "127", get_mcp_config),
    _DigitCase("vector_store", "dsn", "42", get_vector_store_config),
]


def _load_dotenv(directory: Path, line: str) -> Config:
    dotenv = directory / ".env"
    dotenv.write_text(f"{line}\n", encoding="utf-8")
    return get_config(dotenv_path=str(dotenv))


@pytest.mark.os_agnostic
@pytest.mark.parametrize("case", _DIGIT_CASES, ids=lambda case: f"{case.section}.{case.key}")
def test_a_digit_only_value_reaches_a_text_setting_as_its_text(
    tmp_path: Path, clear_config_cache: None, case: _DigitCase
) -> None:
    config = _load_dotenv(tmp_path, f"{case.section.upper()}__{case.key.upper()}={case.raw}")
    # Premise: the library hands the value over as a number. Without it the test proves nothing.
    assert isinstance(config.get(case.section, {})[case.key], int)

    parsed = case.reader(config)

    assert getattr(parsed, case.key) == case.raw


@pytest.mark.os_agnostic
@pytest.mark.parametrize(
    ("env_key", "attr"),
    [("LIB_LOG_RICH__SERVICE", "service"), ("LIB_LOG_RICH__ENVIRONMENT", "environment")],
)
def test_a_digit_only_logging_name_is_read_as_its_digits(
    tmp_path: Path, clear_config_cache: None, env_key: str, attr: str
) -> None:
    config = _load_dotenv(tmp_path, f"{env_key}=2026")
    assert isinstance(config.get("lib_log_rich", {})[attr], int)

    runtime = _build_runtime_config(config)  # pyright: ignore[reportPrivateUsage]

    assert getattr(runtime, attr) == "2026"


@pytest.mark.os_agnostic
@pytest.mark.parametrize(
    ("env_key", "attr", "expected"),
    [
        ("EMAIL__SMTP_PASSWORD", "smtp_password", SecretStr("123456")),
        ("EMAIL__SMTP_USERNAME", "smtp_username", "4711"),
        ("EMAIL__LOCAL_HOSTNAME", "smtp_local_hostname", "10"),
    ],
)
def test_a_digit_only_email_setting_is_read_as_its_digits(
    tmp_path: Path, clear_config_cache: None, env_key: str, attr: str, expected: object
) -> None:
    raw = expected.get_secret_value() if isinstance(expected, SecretStr) else expected
    config = _load_dotenv(tmp_path, f"{env_key}={raw}")
    assert isinstance(config.get("email", {})[env_key.split("__", 1)[1].lower()], int)

    loaded = load_email_config_from_dict(config.as_dict())

    assert getattr(loaded, attr) == expected


@pytest.mark.os_agnostic
def test_a_numeric_setting_still_reads_as_a_number(tmp_path: Path, clear_config_cache: None) -> None:
    config = _load_dotenv(tmp_path, "SANITIZE__WINDOW_CHARS=500")

    assert get_sanitize_config(config).window_chars == 500


@pytest.mark.os_agnostic
def test_an_out_of_range_digit_only_value_is_still_refused(tmp_path: Path, clear_config_cache: None) -> None:
    config = _load_dotenv(tmp_path, "SANITIZE__MIN_DETECTORS=9")

    with pytest.raises(ValidationError, match="min_detectors"):
        get_sanitize_config(config)
