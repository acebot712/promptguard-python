"""
Tests for scripts/generate_types_from_spec.py — the two-layer defence against
a hostile OpenAPI spec injecting module-level code through a ``$ref`` (or an
``anyOf`` / ``items`` ``$ref``) segment, and the shape of the module it emits.

The shape matters because nobody reads this file before it lands: the weekly
sync workflow regenerates it unattended and opens a PR. If the generator emits
something ``ruff format`` rejects, the PR arrives red on a file no human wrote.
"""

import ast
import importlib.util
import subprocess
import sys
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


# ── Module shape: the output must survive `ruff format --check` ───────────


def _spec_with(schemas: dict) -> dict:
    return {"info": {"version": "1.0.0"}, "components": {"schemas": schemas}}


# One of each block the generator can emit, described and undescribed, so the
# formatting assertions below cover every path through generate_typed_dict().
_REPRESENTATIVE_SPEC = _spec_with(
    {
        "Described": {
            "type": "object",
            "description": "A described schema.\n\nWith a second paragraph.",
            "properties": {"a": {"type": "string", "description": "A field"}},
            "required": ["a"],
        },
        "Undescribed": {
            "type": "object",
            "properties": {"b": {"type": "integer"}},
        },
        "DescribedEnum": {
            "description": "A described enum.",
            "enum": ["x", "y"],
        },
        "PlainEnum": {"enum": ["z"]},
        "DescribedAlias": {"type": "object", "description": "A described alias."},
        "PlainAlias": {"type": "object"},
    }
)


def _ruff() -> str:
    """Path to the ruff pinned in the dev extra, or skip."""
    candidate = Path(sys.executable).parent / "ruff"
    if candidate.exists():
        return str(candidate)
    pytest.skip("ruff is not installed in this environment")


class TestGeneratedModuleIsFormatted:
    def test_output_is_ruff_format_clean(self):
        source = gen.render_module(_REPRESENTATIVE_SPEC)
        result = subprocess.run(  # noqa: S603 - fixed argv, no shell
            [_ruff(), "format", "--diff", "--stdin-filename", "api_types.py", "-"],
            input=source,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, f"ruff format would rewrite the output:\n{result.stdout}"

    def test_two_blank_lines_between_top_level_blocks(self):
        source = gen.render_module(_REPRESENTATIVE_SPEC)
        # Every top-level `class` past the first is preceded by two blank lines.
        lines = source.splitlines()
        class_lines = [i for i, line in enumerate(lines) if line.startswith("class ")]
        assert class_lines, "expected at least one class"
        for i in class_lines:
            assert lines[i - 1] == "" and lines[i - 2] == "", (
                f"line {i + 1} ({lines[i]!r}) is not preceded by two blank lines"
            )

    def test_no_trailing_blank_line(self):
        source = gen.render_module(_REPRESENTATIVE_SPEC)
        assert source.endswith("\n")
        assert not source.endswith("\n\n")


class TestDescriptionsArePlacedCorrectly:
    def test_schema_description_becomes_the_class_docstring(self):
        tree = ast.parse(gen.render_module(_REPRESENTATIVE_SPEC))
        described = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Described"
        )
        assert ast.get_docstring(described) == "A described schema. With a second paragraph."

    def test_no_floating_strings_at_module_level(self):
        """Only the module docstring may be a bare string statement.

        The generator used to emit each schema's description as a module-level
        string above its class: a no-op statement documenting nothing.
        """
        tree = ast.parse(gen.render_module(_REPRESENTATIVE_SPEC))
        stray = [n for n in tree.body[1:] if gen._is_docstring_expr(n)]
        assert not stray, f"{len(stray)} bare string statement(s) left at module level"

    def test_alias_and_enum_descriptions_become_comments(self):
        source = gen.render_module(_REPRESENTATIVE_SPEC)
        # An alias has no body to hold a docstring, so it gets a comment.
        assert "# A described enum.\nDescribedEnum = Literal[" in source
        assert "# A described alias.\nDescribedAlias = dict[str, Any]" in source
        assert "PlainAlias = dict[str, Any]" in source


class TestDocstringSanitisation:
    def test_description_ending_in_a_quote_stays_parseable(self):
        # Without escaping, the trailing quote butts against the closing
        # delimiter and the module fails to parse.
        spec = _spec_with(
            {
                "Quoted": {
                    "type": "object",
                    "description": 'he said "hi"',
                    "properties": {"a": {"type": "string"}},
                }
            }
        )
        tree = ast.parse(gen.render_module(spec))
        quoted = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        assert ast.get_docstring(quoted) == 'he said "hi"'

    def test_triple_quote_cannot_close_the_docstring_early(self):
        spec = _spec_with(
            {
                "Hostile": {
                    "type": "object",
                    "description": '"""\nEVIL = 1\n"""',
                    "properties": {"a": {"type": "string"}},
                }
            }
        )
        output = gen.render_module(spec)
        assert "EVIL = 1" not in [line.strip() for line in output.splitlines()]
        ast.parse(output)
