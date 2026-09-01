# Fail open by default when the Guard API is unreachable

When a scan cannot reach the Guard API, the call proceeds unscanned and a
warning is logged. This is the default for both auto-instrumentation and the
Guard client, and it applies on the response path as well as the request path.
Callers who want the opposite ask for it explicitly with `fail_open=False`,
which turns an unreachable Guard API into a raised `GuardApiError`.

A security SDK that defaults to allowing traffic looks wrong at first glance.
The alternative is worse for almost everyone who installs this: PromptGuard
would become a hard dependency of the customer's application, and a Guard API
outage — or a network blip, or a too-tight timeout — would take down their
product rather than degrade its security. Very few teams will accept that
trade to adopt a drop-in SDK, and the ones who genuinely need it are the ones
who will read the flag and set it.

## Consequences

Retries happen before the fail-open policy governs, so a transient 429 or 5xx
does not silently become unscanned traffic. A malformed response body is
surfaced as a `GuardApiError` rather than defaulting to "allow", precisely so
the caller's explicit policy decides instead of a parsing accident.

The visible cost is that "PromptGuard is installed" does not by itself mean
"every call was scanned". Anything reporting on coverage has to account for
fail-open, and changing this default later would silently change the security
posture of every existing installation — so it is a major-version decision,
not a tuning knob.
