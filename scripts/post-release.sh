#!/usr/bin/env bash
# Post one Release to Slack #releases. Never fails the release it reports.
#
#   SLACK_RELEASES_WEBHOOK_URL=... scripts/release/post-release.sh \
#     <product> <version> <notes-url> [outcome]
#
# ADR-0038: every product's release workflow tells the team it shipped:
# product, version, where the notes are and, for the platform, what production
# is serving after the run. The canonical copy is in promptguard
# (scripts/release/post-release.sh, tested by test_announce.py); the six
# sibling repos carry copies. Change them together.
#
# A post that fails is a warning, never a failure: the Release already
# happened, and a red run for a Slack outage would send someone to "fix" a
# deploy that is fine. So every path, including a missing secret and a usage
# error, exits 0 and says why on a ::warning:: line.
set -uo pipefail

warn() { echo "::warning::#releases: $*"; exit 0; }

[ "$#" -ge 3 ] || warn "usage: $0 <product> <version> <notes-url> [outcome]; nothing posted"
product="$1" version="$2" notes="$3" outcome="${4:-}"

[ -n "${SLACK_RELEASES_WEBHOOK_URL:-}" ] \
  || warn "SLACK_RELEASES_WEBHOOK_URL is not set; nobody was told about $product $version"

if [ -n "$outcome" ]; then
  text="*$product $version*: $outcome"$'\n'"Notes: $notes"
else
  text="*$product $version* released"$'\n'"Notes: $notes"
fi
payload=$(jq -n --arg text "$text" '{text: $text, unfurl_links: false}') \
  || warn "could not build the message (is jq installed?); nothing posted"

if ! curl -sS --fail --max-time 15 -H 'Content-Type: application/json' \
    --data "$payload" "$SLACK_RELEASES_WEBHOOK_URL" >/dev/null; then
  warn "Slack refused or did not answer; $product $version was not announced"
fi
echo "#releases: announced $product $version"
