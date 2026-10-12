"""The base every configuration section model derives from.

lib_layered_config reads an unquoted ``.env`` or environment value that spells a number
losslessly as an int or float (``INDEX__COLLECTION=2026`` arrives as ``2026``). A pydantic ``str``
field refuses a number, so a section whose text setting happens to be made of digits would stop
the program. The conversion is lossless, so turning the number back into text restores exactly
what the operator wrote; numeric settings keep reading as numbers.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ConfigSection(BaseModel):
    """A configuration section model: a text setting keeps the digits it was given."""

    model_config = ConfigDict(coerce_numbers_to_str=True)


__all__ = ["ConfigSection"]
