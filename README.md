# Mel — customer-support agent for Empório da Música

Mel is a text-based support agent for a fictional musical-instrument store in Campo Grande, MS (Brazil). It answers store-info questions, looks up product prices and availability, applies the store policies (returns, payment, shipping, promotions, warranty) and checks order status — through **function calling** over an **open model served by NVIDIA** (hosted API or a local NIM container).

> **Status of live behavior.** The agent has been run against the **NVIDIA hosted API** with `mistralai/mistral-large-4-1t-a52b`; the five transcripts in `examples/` were produced that way by `scripts/run_examples.py`. It has **never been run against a local NIM container**. Only a few manual conversations were checked, and tool-calling reliability has not been measured. Data cleaning, BM25 search, all business tools, history and the agent loop (with scripted fake responses) are covered by offline tests: `python -m pytest -q` → 61 passed.

Mel always replies **in English**, even if the customer writes in Portuguese. Tool output and policy text stay in Portuguese and are translated by the model.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate          # Linux/macOS
# .\.venv\Scripts\Activate.ps1     # Windows PowerShell
pip install -r requirements.txt
cp .env.example .env        # then edit it
python -m pytest -q         # offline, no key needed
python -m emporio.cli       # chat (history saved in sessions/)
python scripts/run_examples.py   # regenerate examples/*.md with the real model
```

CLI options: `--session NAME` (separate conversation history), `--reset` (clear it).

### Option A — NVIDIA hosted API (build.nvidia.com)

1. Create a key at <https://build.nvidia.com> (starts with `nvapi-`).
2. In `.env`:
   ```
   NVIDIA_API_KEY=nvapi-...
   NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1
   EMPORIO_MODEL=mistralai/mistral-large-4-1t-a52b
   ```

**Choosing a model.** `meta/llama-3.1-70b-instruct` is end-of-life on the hosted API (HTTP 410, EOL 2026-08-26). Many models listed by `GET /v1/models` return HTTP 404 for a given account, so the list is not proof of access. Of 59 candidates tested with tool calling, only 9 worked — for example `mistralai/mistral-large-4-1t-a52b` (the default), `nvidia/nemotron-3-ultra-550b-a55b`, `minimaxai/minimax-m3.1-flash` and `poolside/laguna-xs-2.1`. Before switching `EMPORIO_MODEL`, probe the model with a single request containing one user message and a `tools` array, and check that it returns HTTP 200 with a `tool_calls` response.

### Option B — local NVIDIA NIM (not tested)

Needs an NVIDIA GPU, Docker with the NVIDIA Container Toolkit and an NGC API key. This path has not been run for this project. The example below is illustrative: check the model card for the exact image name, tag and hardware requirements — the Llama 3.1 70B NIM may also be discontinued. Whatever you serve, `EMPORIO_MODEL` must match the model id the container exposes (see its `GET /v1/models`).

```bash
export NGC_API_KEY=...
docker run -it --rm --gpus all --shm-size=16g -p 8000:8000 \
  -e NGC_API_KEY -v ~/.cache/nim:/opt/nim/.cache \
  nvcr.io/nim/meta/llama-3.1-70b-instruct:latest
```

Then in `.env`:
```
NVIDIA_API_KEY=dummy          # ignored by a local NIM
NVIDIA_BASE_URL=http://localhost:8000/v1
EMPORIO_MODEL=meta/llama-3.1-70b-instruct   # must match the model served by the container
```

| Variable | Default | Meaning |
|---|---|---|
| `NVIDIA_API_KEY` | – | API key (dummy value is fine for a local NIM) |
| `NVIDIA_BASE_URL` | `https://integrate.api.nvidia.com/v1` | OpenAI-compatible endpoint |
| `EMPORIO_MODEL` | `mistralai/mistral-large-4-1t-a52b` | Model id (must support tool calling) |

## Architecture and decisions

```
customer ─▶ cli.py ─▶ Agent.reply() ──▶ OpenAI-compatible chat API (NVIDIA)
                         │   ▲  tool_calls / tool messages (loop, max 6 rounds)
                         ▼   │
                       Tools.run() ─▶ data.py (cleaned CSVs) · policies.py (BM25) · escalation log
                         │
                       history.py (sessions/<id>.json)
```

- **Approach: function calling (hybrid), not free-form RAG.** Prices, promotions, installments, shipping, store hours and return windows are computed by deterministic Python tools, so the LLM never does arithmetic or recalls a price from memory. Free-text policy questions go through **BM25 retrieval** over the policy manual, chunked by heading (the manual is ~8 pages, so lexical search over accent-stripped Portuguese is accurate, dependency-free and explainable; embeddings would add cost without clear gain).
- **Why NVIDIA + an open model.** The OpenAI-compatible API means the same code runs against the hosted catalog (free credits for prototyping) and a self-hosted NIM, which suits a store that may not want customer data leaving its infrastructure. Switching between them is only a change of `NVIDIA_BASE_URL` and `EMPORIO_MODEL`. The default, **`mistralai/mistral-large-4-1t-a52b`**, was chosen because it answered tool-calling requests on the hosted API for this account (see [Choosing a model](#option-a--nvidia-hosted-api-buildnvidiacom)); model availability on the hosted catalog changes over time, so probe before relying on any id. The agent handles parallel tool calls and also works if the model emits them one at a time.
- **Prompt strategy.** Short prompt, tool-usage rules first, then business rules, then two few-shot examples, because open models follow long prompts less reliably than Claude. Temperature 0.2.
- **Robustness.** Malformed JSON arguments or unknown tools return an error string to the model so it can retry; a failed turn is rolled back from history and the customer gets a fallback message; `MAX_TOOL_ROUNDS = 6`.
- **History.** One JSON file per session; the window sent to the model always starts on a user turn so it never begins with an orphan `tool` message or splits an assistant `tool_calls` from its `tool` replies.
- **Privacy.** Order data is only returned when the order number **and** the e-mail or phone on that order both match. The tool output only exposes the customer's first name.
- **Escalation.** `escalate_to_human` appends a ticket to `sessions/escalations.jsonl` (a stand-in for a real helpdesk integration).

## Data inconsistencies and assumptions

`data/politicas.md` is a cleaned copy of the PDF manual (headings normalized, layout noise removed). Where the sources disagree, I chose as follows:

| Finding | Decision |
|---|---|
| Phone: cover/footer `(67) 3341-4444` vs WhatsApp `(67) 3321-4500` | Both kept: store phone vs WhatsApp channel |
| E-mail: `contato@emporiodamusica.com.br` (cover) vs `contato@empóriodamusica.com.br` (section 1.2, accent is invalid in practice) | Use the ASCII one |
| Credit table says minimum installment R$ 100 for 12x, section 3.1 gives tiers (≤3x: ≥R$ 50; 4–6x: ≥R$ 80; 7–12x: ≥R$ 100) | Tiers from 3.1 are implemented |
| Order 3: total R$ 3.450,00 vs items R$ 3.498,00; order 20: R$ 1.400,00 vs R$ 1.488,00 | Reported by `data_issues()`; order total is shown as stored |
| Product names vs descriptions mismatch (e.g. "Music Man Bass 1X" described as a Fender Jazz Bass; "Korg Synth 1 Pro" described as a Roland with 88 keys but specs say 61) | Product **name** is the source of truth; descriptions are only used for search |
| Product 96 is `active` with stock 0; 130 `coming_soon` and 136 `discontinued` have stock | Only `active` products with stock > 0 are sellable; others are flagged, never offered |
| `Cedr`, `Cedar`, etc. spelling noise in names | Kept as-is (names must match the store's records) |
| No delivery date in orders, only `estimated_delivery` | Used as the receipt date to count the 7-day regret window |
| Defect exchange is 30 days (4.2) while legal warranty is 90 days (8.1) | Both reported; 30 days counted from purchase, 90 from receipt |
| "Region metropolitana de Campo Grande" is not defined | Only the city "Campo Grande" gets metro shipping (free above R$ 500,00; R$ 35,00 flat fee otherwise; exactly R$ 500,00 pays the fee) |
| No orders channel column | All orders are treated as online purchases (right of withdrawal applies) |
| Holidays are not in the data | `open_now` uses the regular schedule only (Campo Grande time, UTC-4) |

Other assumptions: promotions with `is_active = 0` are expired and never offered; when several active promotions exist for one product the highest wins; PIX 5% is never applied on top of a promotion (manual 6.2).

## Known limitations

- **Never tested against a local NIM**, and there is **no systematic evaluation**: the hosted API was used for the five example scenarios and a few manual conversations only. Tool-calling reliability, parallel calls and output formatting across many conversations are unmeasured.
- Open models may skip a tool call and answer from memory, or follow instructions less reliably than Claude. Mitigations: tool-first prompt, error feedback on bad calls, low temperature. There is no automatic check that a tool was called, and no automatic language check.
- In the recorded runs Mel answered in English to customers who wrote in Portuguese — this is the intended behavior. English-only answers are a requirement of this project, not a store policy.
- Outside Campo Grande the shipping fee cannot be quoted (it needs ZIP code, weight and size); Mel lists PAC/SEDEX/Jadlog deadlines and says so.
- Tracking codes come from the CSV; there is no carrier integration. Escalation is a local log.
- Conversation memory is a plain truncation window (last 40 messages), no summarization.
- Customer identity is a weak check (order number + e-mail/phone); real use would need proper authentication.

**With more time:** run a small eval set against the live model (tool selection, language, rule adherence), add a language-guard retry, add a second model fallback, replace the escalation log with a ticketing API, and add ZIP-based freight quoting.

## Example conversations

`scripts/run_examples.py` runs five scenarios through the real agent and writes `examples/*.md` (including the tool calls). They cover catalog search, store info, price + PIX vs promotion, out-of-scope requests and — the non-trivial one — a return request that needs identity verification and the 7-day regret rule. The script refuses to run without `NVIDIA_API_KEY` (unless `NVIDIA_BASE_URL` points to a local NIM) and refuses to write a transcript if the model call failed, so transcripts are never fabricated. The current files were generated against the NVIDIA hosted API with `mistralai/mistral-large-4-1t-a52b` (the model id is written at the top of each transcript).

## AI assistants

This project was built with **Claude Code** (Anthropic's CLI agent). Workflow: I wrote the requirements (the case statement, the manual and the CSVs) as a long prompt; Claude Code planned the structure, then implemented module by module (data → policies → tools → history → prompt/agent → CLI/docs), writing the tests alongside the code and running them after each step. Business rules were reviewed against the original manual text, and decisions on ambiguous points are listed above. The Claude assistant used web search only to check NVIDIA's NIM function-calling documentation. This README was written and later updated by Claude Code's `devops` agent.

## Project layout

```
emporio/   data.py policies.py tools.py history.py prompts.py agent.py cli.py
data/      CSV tables + politicas.md
scripts/   run_examples.py
tests/     offline tests (FakeClient returns OpenAI-shaped responses)
examples/  generated conversations
```
