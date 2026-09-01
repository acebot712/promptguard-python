# Both SDKs are held in step by a shared contract, not by convention

The Python and Node SDKs are deliberate mirrors, and "keep them the same" is
not a thing code review can be trusted to enforce across two repositories. So
the behaviour that must match is written once as a file of cases —
`tests/guard-contract.json` — which both SDKs vendor and both test suites run.
Adding a feature to one SDK means adding its case to the contract first.

The copy in each repo is vendored rather than fetched at test time, so the
suite stays offline and deterministic. That reintroduces the drift problem one
level down, which is what `tests/guard-contract.lock.json` is for: it records
the published contract's version and sha256, and the contract test checks the
vendored file against that digest. Editing either file alone fails the suite.
Hand-editing the contract to make a test pass is therefore not available, which
is the point.

## Consequences

The contract covers wire-facing behaviour — decision parsing, redaction
enforcement, message conversion per provider, the blocked-error shape, the
request payload. It does **not** cover either SDK's auto-instrumentation
introspection surface, and those have already drifted: this SDK exposes
`patched_sdks()` and `is_active()`, while Node exposes `getAppliedPatches()`
and `detectedUnpatched()`, with the detected-unpatched helper private here and
public there. Nothing failed, because nothing checks.

So the contract's coverage is the real boundary of the mirroring guarantee.
Anything outside it matches only as long as someone remembers, and the honest
options are to extend the contract or to stop claiming parity for that surface.

The counterpart of this ADR lives in `promptguard-node`.
