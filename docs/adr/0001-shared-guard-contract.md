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

The contract's coverage is the real boundary of the mirroring guarantee.
Anything outside it matches only as long as someone remembers, which is not a
guarantee at all.

Writing that down is what surfaced the first breach. Through v1.5.1 the
contract covered wire-facing behaviour only — decision parsing, redaction
enforcement, message conversion per provider, the blocked-error shape, the
request payload — and said nothing about what auto-instrumentation reports
about itself. That surface had already drifted: this SDK named the Bedrock
patch `boto3-bedrock` where Node named it `bedrock`, so the same health check
answered differently per language. Nothing failed, because nothing checked.

**v1.6.0 closed it.** An `instrumentation_introspection` section now pins the
report's key set, the advice URL, the provider-patch name vocabulary and two
invariants, and this SDK took the `bedrock` rename. The gap that motivated this
paragraph is the reason the section exists.

What remains deliberately unpinned is the *accessor* naming: `patched_sdks()`
here against `getAppliedPatches()` in Node, `is_active()` only here,
`detectedUnpatched()` public there and private here. Those are public API in
two published packages, so aligning them costs a major version in both. The
contract pins the facts the two SDKs report rather than the names they report
them under, which is the part a caller's assertion actually depends on.

The counterpart of this ADR lives in `promptguard-node`.
