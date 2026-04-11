#!/usr/bin/env python3
"""
Generate Python type definitions from the OpenAPI developer spec.

Output goes to promptguard/generated/api_types.py — ONLY types, no runtime code.
Hand-written client code in promptguard/ is never touched.

Usage:
    python scripts/generate_types_from_spec.py openapi-developer.json
"""

from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path


def resolve_ref(ref: str) -> str:
    return ref.rsplit("/", 1)[-1]


def map_type(prop: dict) -> str:
    if "$ref" in prop:
        return resolve_ref(prop["$ref"])
    if "anyOf" in prop:
        types = [map_type(p) for p in prop["anyOf"]]
        return " | ".join(types)
    if "enum" in prop:
        return " | ".join(f'"{v}"' for v in prop["enum"])

    t = prop.get("type", "")

    if t == "string":
        return "str"
    if t in ("integer", "number"):
        return "float" if t == "number" else "int"
    if t == "boolean":
        return "bool"
    if t == "array":
        items = prop.get("items", {})
        inner = map_type(items)
        return f"list[{inner}]"
    if t == "object":
        return "dict[str, Any]"
    return "Any"


def generate_typed_dict(name: str, schema: dict) -> str:
    lines: list[str] = []

    if schema.get("description"):
        lines.append(f'"""{schema["description"]}"""')
        lines.append("")

    if "enum" in schema:
        values = ", ".join(f'"{v}"' for v in schema["enum"])
        lines.append(f"{name} = Literal[{values}]")
        lines.append("")
        return "\n".join(lines)

    props = schema.get("properties")
    if not props:
        lines.append(f"{name} = dict[str, Any]")
        lines.append("")
        return "\n".join(lines)

    required = set(schema.get("required", []))

    if required == set(props.keys()):
        lines.append(f"class {name}(TypedDict):")
    elif not required:
        lines.append(f"class {name}(TypedDict, total=False):")
    else:
        lines.append(f"class {name}(TypedDict, total=False):")

    for prop_name, prop_def in props.items():
        py_type = map_type(prop_def)
        if prop_def.get("description"):
            lines.append(f"    # {prop_def['description']}")
        lines.append(f"    {prop_name}: {py_type}")

    lines.append("")
    return "\n".join(lines)


def main():
    if len(sys.argv) < 2:
        sys.stderr.write(
            "Usage: python scripts/generate_types_from_spec.py <openapi-developer.json>\n"
        )
        sys.exit(2)

    spec_path = sys.argv[1]
    if not Path(spec_path).exists():
        sys.stderr.write(f"Spec file not found: {spec_path}\n")
        sys.exit(2)

    spec = json.loads(Path(spec_path).read_text())
    schemas = spec.get("components", {}).get("schemas", {})
    version = spec.get("info", {}).get("version", "unknown")

    header = textwrap.dedent(f"""\
        \"\"\"
        Auto-generated from OpenAPI spec (v{version}).
        DO NOT EDIT — regenerate with: python scripts/generate_types_from_spec.py

        These are type-only definitions. Custom client logic lives in
        promptguard/guard.py, promptguard/client.py, promptguard/patches/,
        and promptguard/integrations/ — those files are never modified
        by this generator.
        \"\"\"

        from __future__ import annotations

        from typing import Any, Literal, TypedDict

    """)

    body = "\n\n".join(
        generate_typed_dict(name, schema) for name, schema in sorted(schemas.items())
    )

    output = header + body + "\n"

    out_dir = Path(__file__).parent.parent / "promptguard" / "generated"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "api_types.py"
    out_path.write_text(output)

    init_path = out_dir / "__init__.py"
    if not init_path.exists():
        init_path.write_text("")

    count = len(schemas)
    sys.stderr.write(f"Generated {count} types → {out_path}\n")


if __name__ == "__main__":
    main()
