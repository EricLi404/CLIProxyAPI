# Codex Compact Adapter

This standalone proxy keeps OpenAI-compatible traffic transparent while serving
Codex remote compaction locally for selected upstream models.

## Behavior

- `gpt-*` and other non-matching models are forwarded unchanged.
- Matching models (by default `deepseek-*`, `glm-*`, `mimo-*`, and `minimax-*`)
  have `compaction_trigger` requests summarized through a normal Responses call.
- The generated checkpoint is returned as one `type=compaction` output item.
- Adapter-owned compact envelopes are decoded on later requests before they are
  sent to the upstream proxy.
- `/responses/compact` is handled locally for matching models as well.

The envelope is a compatibility format, not an OpenAI encrypted compact state.
Keep the adapter in front of the same upstream for the lifetime of a session.

## Run

Build from the CPA source checkout:

```bash
GOWORK=off go build -o /tmp/codex-compact-adapter ./cmd/codex-compact-adapter
```

Start it in front of the existing CPA on port `8317`:

```bash
CODEX_COMPACT_TARGET=http://127.0.0.1:8317 \
CODEX_COMPACT_LISTEN=127.0.0.1:8320 \
/tmp/codex-compact-adapter
```

To use it from Codex App, point the existing `OpenAI` provider `base_url` at
`http://127.0.0.1:8320/v1`. Restore `8317` to roll back.

Override the model allowlist with a comma-separated wildcard list:

```bash
CODEX_COMPACT_MODELS='deepseek-*,glm-*'
```

## Verify

```bash
curl http://127.0.0.1:8320/health
GOWORK=off go test ./cmd/codex-compact-adapter
```
