# Auto-instrumentation scans out of band; the provider call still goes direct

`init()` wraps the provider SDK's methods so that each call is scanned by the
Guard API before it goes out, and optionally again on the response. The
provider call itself is left alone: the original method runs with the
application's own client, against the provider's own endpoint. PromptGuard is
never in the data path.

The obvious alternative was to have `init()` repoint the patched client at the
PromptGuard proxy — which is what the CLI's static transform does, so the
machinery exists and is proven. It was not taken here. Putting the proxy in
the data path of every patched call means PromptGuard has to faithfully
forward every provider feature the application uses — streaming, tools,
structured outputs, whatever ships next — and any gap becomes a broken
customer application rather than a missed scan. Scanning out of band keeps the
blast radius of a PromptGuard problem at "not scanned" instead of "not
working", which is also what makes fail-open a coherent default.

## Consequences

Each scanned call costs an extra round trip, and a response scan costs a
second one — this is latency the application pays serially, which is why
response scanning is opt-in rather than always on.

Because the provider call is untouched, a redact decision has to rewrite the
outgoing arguments before the original method runs. When the call's shape
cannot be rewritten safely, there is nothing to fall back to except stopping
the call, so enforce mode escalates that redact to a block rather than sending
unredacted content.

Proxy mode remains the other way to adopt the SDK, and there PromptGuard *is*
in the request path. The two are different products of this trade-off, not two
implementations of one.
