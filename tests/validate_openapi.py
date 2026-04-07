#!/usr/bin/env python3
"""
Validate that the Python SDK's types and field names match the
OpenAPI developer spec.

Usage:
    python tests/validate_openapi.py openapi-developer.json

Exit code 0 = SDK is in sync; non-zero = drift detected.
"""

import json
import sys
from pathlib import Path


def load_spec(path: str) -> dict:
    return json.loads(Path(path).read_text())


def get_schema_properties(spec: dict, schema_name: str) -> set[str]:
    schema = spec.get("components", {}).get("schemas", {}).get(schema_name, {})
    return set(schema.get("properties", {}).keys())


def get_required_fields(spec: dict, schema_name: str) -> set[str]:
    schema = spec.get("components", {}).get("schemas", {}).get(schema_name, {})
    return set(schema.get("required", []))


def validate_security_scheme(spec: dict) -> list[str]:
    errors: list[str] = []
    schemes = spec.get("components", {}).get("securitySchemes", {})
    api_key_auth = schemes.get("ApiKeyAuth", {})

    if api_key_auth.get("type") != "apiKey":
        errors.append(f"ApiKeyAuth type should be 'apiKey', got '{api_key_auth.get('type')}'")
    if api_key_auth.get("in") != "header":
        errors.append(f"ApiKeyAuth 'in' should be 'header', got '{api_key_auth.get('in')}'")
    if api_key_auth.get("name") != "X-API-Key":
        errors.append(f"ApiKeyAuth name should be 'X-API-Key', got '{api_key_auth.get('name')}'")
    return errors


def validate_error_schemas(spec: dict) -> list[str]:
    errors: list[str] = []
    schemas = spec.get("components", {}).get("schemas", {})

    if "QuotaErrorDetail" in schemas:
        props = set(schemas["QuotaErrorDetail"].get("properties", {}).keys())
        expected = {
            "message",
            "type",
            "code",
            "current_plan",
            "requests_used",
            "requests_limit",
            "upgrade_url",
        }
        missing = expected - props
        if missing:
            errors.append(f"QuotaErrorDetail missing fields: {missing}")
    else:
        errors.append("QuotaErrorDetail schema not found in spec")

    if "ErrorDetail" in schemas:
        props = set(schemas["ErrorDetail"].get("properties", {}).keys())
        expected = {"message", "type", "code"}
        missing = expected - props
        if missing:
            errors.append(f"ErrorDetail missing fields: {missing}")
    else:
        errors.append("ErrorDetail schema not found in spec")

    return errors


def validate_scan_schema(spec: dict) -> list[str]:
    errors: list[str] = []

    req_props = get_schema_properties(spec, "ScanRequest")
    if "content" not in req_props:
        errors.append("ScanRequest missing 'content' field")
    if "text" in req_props:
        errors.append("ScanRequest has deprecated 'text' field")

    resp_props = get_schema_properties(spec, "ScanResponse")
    expected = ("blocked", "decision", "reason", "confidence", "eventId", "processingTimeMs")
    errors.extend(f"ScanResponse missing '{f}' field" for f in expected if f not in resp_props)
    return errors


def validate_redact_schema(spec: dict) -> list[str]:
    errors: list[str] = []

    req_props = get_schema_properties(spec, "RedactRequest")
    if "content" not in req_props:
        errors.append("RedactRequest missing 'content' field")
    if "text" in req_props:
        errors.append("RedactRequest has deprecated 'text' field")

    resp_props = get_schema_properties(spec, "RedactResponse")
    errors.extend(
        f"RedactResponse missing '{f}' field"
        for f in ("original", "redacted", "piiFound")
        if f not in resp_props
    )
    return errors


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python validate_openapi.py <openapi-developer.json>\n")
        sys.exit(2)

    spec_path = sys.argv[1]
    if not Path(spec_path).exists():
        sys.stderr.write(f"Spec file not found: {spec_path}\n")
        sys.exit(2)

    spec = load_spec(spec_path)
    all_errors: list[str] = []

    all_errors.extend(validate_security_scheme(spec))
    all_errors.extend(validate_error_schemas(spec))
    all_errors.extend(validate_scan_schema(spec))
    all_errors.extend(validate_redact_schema(spec))

    if all_errors:
        sys.stderr.write(f"SDK/OpenAPI drift detected ({len(all_errors)} issue(s)):\n")
        for err in all_errors:
            sys.stderr.write(f"  - {err}\n")
        sys.exit(1)
    else:
        sys.stderr.write("SDK types match OpenAPI spec.\n")
        sys.exit(0)
