"""
Tests for scripts/generate_types_from_spec.py — the two-layer defence against
a hostile OpenAPI spec injecting module-level code through a ``$ref`` (or an
``anyOf`` / ``items`` ``$ref``) segment.
"""

import importlib.util
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "generate_types_from_spec.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_gen_types_under_test", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


gen = _load_module()


# ── Layer (a): $ref / anyOf / items component validation ──────────────────


class TestMapTypeRefValidation:
    def test_clean_ref_passes_through(self):
        assert gen.map_type({"$ref": "#/components/schemas/User"}) == "User"

    def test_ref_with_newline_falls_back_to_any(self):
        # A hostile $ref whose final segment carries a newline + injected code.
        hostile = "#/components/schemas/X\nimport os\nos.system('touch /tmp/pwned')"
        assert gen.map_type({"$ref": hostile}) == "Any"

    def test_ref_with_non_identifier_chars_falls_back(self):
        assert gen.map_type({"$ref": "#/components/schemas/Not-Valid"}) == "Any"
        assert gen.map_type({"$ref": "#/components/schemas/has space"}) == "Any"

    def test_hostile_ref_inside_anyof_falls_back(self):
        prop = {
            "anyOf": [
                {"type": "string"},
                {"$ref": "#/components/schemas/Y\nBAD = __import__('os')"},
            ]
        }
        assert gen.map_type(prop) == "str | Any"

    def test_hostile_ref_inside_array_items_falls_back(self):
        prop = {"type": "array", "items": {"$ref": "#/components/schemas/Z\nraise SystemExit"}}
        assert gen.map_type(prop) == "list[Any]"


# ── Layer (b): AST guard rejects non-type statements ──────────────────────


class TestAssertTypesOnly:
    def test_clean_module_passes(self):
        source = (
            '"""doc"""\n'
            "from __future__ import annotations\n"
            "from typing import Any, Literal, TypedDict\n"
            "class Foo(TypedDict):\n"
            "    a: str\n"
            "Bar = dict[str, Any]\n"
        )
        gen._assert_types_only(source)  # should not raise

    def test_injected_call_expression_rejected(self):
        source = "import os\nos.system('touch /tmp/pwned')\n"
        with pytest.raises(ValueError, match="unexpected"):
            gen._assert_types_only(source)

    def test_injected_function_def_rejected(self):
        source = "def evil():\n    return 1\n"
        with pytest.raises(ValueError, match="unexpected"):
            gen._assert_types_only(source)

    def test_syntax_error_propagates(self):
        with pytest.raises(SyntaxError):
            gen._assert_types_only("class Foo(:\n")


# ── End-to-end: a malicious spec produces safe, valid output ──────────────


class TestRenderModuleWithMaliciousSpec:
    def test_hostile_ref_is_neutralised_in_output(self):
        spec = {
            "info": {"version": "1.0.0"},
            "components": {
                "schemas": {
                    "Widget": {
                        "type": "object",
                        "properties": {
                            "kind": {
                                "$ref": (
                                    "#/components/schemas/Kind\n"
                                    "import os\n"
                                    "os.system('touch /tmp/pwned')"
                                )
                            }
                        },
                    }
                }
            },
        }
        output = gen.render_module(spec)
        # The breakout never lands in the source; the field degrades to Any.
        assert "os.system" not in output
        assert "import os" not in output
        assert "kind: Any" in output

    def test_hostile_version_is_dropped(self):
        spec = {
            "info": {"version": "1.0.0\nBAD = 1"},
            "components": {"schemas": {}},
        }
        output = gen.render_module(spec)
        assert "BAD = 1" not in output
        assert "(vunknown)" in output

    def test_malicious_schema_name_skipped(self):
        spec = {
            "info": {"version": "1.0.0"},
            "components": {
                "schemas": {
                    "Good": {"type": "object", "properties": {"x": {"type": "string"}}},
                    "Bad\nEVIL = 1": {"type": "object"},
                }
            },
        }
        output = gen.render_module(spec)
        assert "class Good(" in output
        assert "EVIL = 1" not in output
