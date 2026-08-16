# Changelog

All notable changes to this project are documented here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

`Unreleased` holds work that is merged but not yet published. Move entries into
a dated version section when a release goes out — an `Unreleased` block that
survives three releases is a changelog nobody is maintaining.


## [Unreleased]

### Added

- **`promptguard.verify()` — a positive check that protection is actually
  live.** `init()` returning cleanly has never meant anything is being scanned:
  the SDK fails open, so a rejected API key, an unreachable Guard API or a
  provider SDK we never hooked all leave an application that runs perfectly and
  blocks nothing. Each of those already logged a warning, but a warning in a log
  nobody tails is indistinguishable from silence. `verify()` makes the real
  calls — reachability, authentication, a live injection probe, a PII probe —
  and returns what came back, so a deployment can assert it instead of assuming
  it: `assert promptguard.verify()["ok"]`. The checks and their names mirror
  `promptguard verify` in the CLI so the two agree on what "working" means. It
  never raises for a failed check, so a CI step sees every problem at once
  rather than only the first.

## [1.12.0] — 2026-08-11

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


<!-- Entries below are reconstructed from git tags: this file did not exist
     until 2026-08-11, so the tag subject is all the detail anyone recorded.
     Accurate about WHAT shipped, thin on WHY. Entries from here on are
     written when the release is cut -- scripts/check_changelog.py fails the
     release if the version being published has no section. -->

## [1.11.1] — 2026-08-03

### Changed

- v1.11.1 — dependency updates

## [1.11.0] — 2026-07-13

### Changed

- chore(release): bump version to 1.11.0 (LangChain + LlamaIndex adapters) (#19)

## [1.10.0] — 2026-07-12

### Changed

- chore(release): 1.10.0

## [1.9.0] — 2026-06-01

### Changed

- release: v1.9.0 — proxy-suffix auto-fix, retry controls, SecurityScanResult (#14)

## [1.8.1] — 2026-04-19

### Changed

- chore(release): 1.8.1 (#7)

## [1.8.0] — 2026-04-11

### Changed

- chore: bump version to 1.8.0

## [1.7.1] — 2026-04-10

### Changed

- chore: update model references from gpt-4o to gpt-5-nano and bump to v1.7.1

## [1.7.0] — 2026-04-07

### Changed

- feat: add API contract testing, OpenAPI validation, and quota error handling

## [1.6.0] — 2026-04-06

### Changed

- feat: bump version to 1.6.0

## [1.5.3] — 2026-04-05

### Changed

- chore: bump version to 1.5.3

## [1.5.2] — 2026-03-25

### Changed

- v1.5.2: clean up deprecated integration code

## [1.5.1] — 2026-02-28

### Changed

- Bump version to 1.5.1
