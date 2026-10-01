"""Pydantic output models for the MCP tool responses.

The MCP tools are an output boundary: each is annotated to return one of these
models (or a list of them), and returns a model instance directly. FastMCP derives
the tool's ``outputSchema`` from the annotation and serializes the instance itself,
so the wire shape is generated from the model rather than hand-built as a dict.
Defining the shapes here keeps each tool's output typed and single-sourced.

Every model and enum field states its own one-line wire description. Pydantic would
otherwise copy the class docstring into the schema, and those are written for
developers (multi-paragraph, RST markup, deployment advice), not for an MCP client.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, WithJsonSchema

from ...domain.enums import Partition, StoreBackend


def _wire_enum(enum: type[StrEnum], description: str) -> WithJsonSchema:
    """Inline schema for an enum field: its values plus a one-line description.

    Replaces the ``$defs`` entry pydantic builds from the enum's docstring; validation is
    unaffected, only the published schema changes.
    """
    return WithJsonSchema({"type": "string", "enum": [member.value for member in enum], "description": description})


class DatasetInfo(BaseModel):
    """One row of ``list_datasets``: safe metadata only (never the dsn / secrets)."""

    model_config = ConfigDict(
        json_schema_extra={"description": "One dataset: safe metadata only, never the DSN or other secrets."}
    )

    name: str
    backend: Annotated[StoreBackend, _wire_enum(StoreBackend, "Vector store backend that holds the dataset.")]
    collection: str
    partition: Annotated[
        Partition, _wire_enum(Partition, "How the dataset's collections are separated within its store backend.")
    ]
    read_only: bool
    writable: bool


class SearchHit(BaseModel):
    """One hit from the single-dataset ``search`` tool.

    ``summary`` is the per-document summary from the opt-in summary tier (``None``
    when the tier is off), so a calling LLM can triage relevance without opening
    the source.
    """

    model_config = ConfigDict(json_schema_extra={"description": "One hit from the single-dataset search tool."})

    uri: str
    ordinal: int
    score: float
    text: str
    label: str
    collection: str
    summary: str | None = Field(
        default=None,
        description="Per-document summary from the opt-in summary tier; None when that tier is off.",
    )


class FusedSearchHit(BaseModel):
    """One hit from the fan-out ``search_datasets`` tool, tagged with its dataset.

    Order results by ``rrf_score`` (the fused rank), never the per-dataset ``score``
    (raw similarity is not comparable across datasets / models).
    """

    model_config = ConfigDict(
        json_schema_extra={
            "description": "One hit from the fan-out search_datasets tool, tagged with its dataset; order by rrf_score."
        }
    )

    uri: str
    ordinal: int
    score: float = Field(
        description="Raw per-dataset similarity score; NOT comparable across datasets/models - order by rrf_score."
    )
    rrf_score: float = Field(
        description="Reciprocal Rank Fusion score across all searched datasets; order results by this, not score."
    )
    dataset: str
    text: str
    label: str
    collection: str
    summary: str | None = Field(
        default=None,
        description="Per-document summary from the opt-in summary tier; None when that tier is off.",
    )


class ReindexResult(BaseModel):
    """Outcome of the ``reindex`` tool: sources (re)indexed, pruned, unchanged."""

    model_config = ConfigDict(
        json_schema_extra={"description": "Outcome of reindex: sources indexed, pruned and left unchanged."}
    )

    dataset: str
    indexed: int
    pruned: int
    unchanged: int


class RememberResult(BaseModel):
    """Outcome of the ``remember`` tool: where the knowledge was stored."""

    model_config = ConfigDict(json_schema_extra={"description": "Outcome of remember: where the knowledge was stored."})

    uri: str
    id: str = Field(description="The entry_id 'forget' takes to delete this entry, or 'remember' to update it.")
    chunks: int


class ForgetResult(BaseModel):
    """Outcome of the ``forget`` tool (idempotent - true even for an unknown id)."""

    model_config = ConfigDict(
        json_schema_extra={"description": "Outcome of forget; idempotent, true even for an unknown entry_id."}
    )

    uri: str
    forgotten: bool = Field(description="Always true, including for an unknown entry_id - forget is idempotent.")


__all__ = [
    "DatasetInfo",
    "ForgetResult",
    "FusedSearchHit",
    "ReindexResult",
    "RememberResult",
    "SearchHit",
]
