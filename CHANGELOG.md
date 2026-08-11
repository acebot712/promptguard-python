# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`Unreleased` holds work that is merged but not yet published. Move entries into
a dated version section when a release goes out — an `Unreleased` block that
survives three releases is a changelog nobody is maintaining.


## [Unreleased]

### Fixed

- **Gemini calls made through Google's current SDK were not being scanned.**
  Auto-instrumentation patched `google-generativeai`, which Google deprecated
  when Gemini 2.0 shipped. The current package is `google-genai`, a different
  import path, so `promptguard.init()` returned no error and protected nothing.
  Both SDKs are now patched; the deprecated one is kept because customers are
  still on it. The new patch additionally scans `config.system_instruction` and
  can redact (the legacy call signature could not).

### Added

- **An installed provider SDK we did not hook is now named at startup.**
  `init()` previously warned only when it patched *nothing*, so a customer with
  `openai` and `google-genai` installed saw a healthy startup and no mention of
  their unscanned Gemini traffic. Each patch module declares what it detects,
  and anything installed-but-unhooked is warned about by name with the fix.
- `instrumentation_report()` — the same facts as data, so coverage can be
  asserted in your own CI rather than read from startup logs:
  `assert promptguard.instrumentation_report()["detected_unpatched"] == []`.

