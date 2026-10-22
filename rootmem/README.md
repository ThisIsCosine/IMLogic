# RootMem construction and usage

`build_rootmem.py` constructs RootMem incrementally from existing per-user
memory files. It preserves the system prompt and batch-update prompt from
the supplied active LightMem implementation verbatim. `use_rootmem.py` reads
its output and implements query-time routing and MCQ answering using the
separately supplied usage script's router and answering prompts verbatim.
These standalone scripts do not modify the existing `eval/` runners.

## Configuration and execution

Install the API client:

```bash
pip install openai
```

Fill the configuration block at the beginning of `build_rootmem.py`:

| Variable | Meaning |
| --- | --- |
| `INPUT_DIR` | Directory containing one memory JSON file per user |
| `OUTPUT_DIR` | Separate output directory; use a fresh directory for each run |
| `LOG_DIR` | Separate directory for per-user logs |
| `INPUT_PATTERN` | File pattern, for example `conv-*.json` or `*.json` |
| `LLM_API_KEY` | API key for an OpenAI-compatible endpoint |
| `LLM_BASE_URL` | API base URL |
| `LLM_MODEL_NAME` | Model identifier |
| `BATCH_SIZE` | Number of memories per incremental update |
| `MAX_WORKERS` | Number of users processed concurrently |
| `TEMPERATURE` | Sampling temperature supported by the chosen model |
| `SAVE_EVERY` | Save an additional numbered snapshot every N batches |
| `TEST_LIMIT` | Optional memory limit per user; `None` processes all memories |

The reference implementation used `BATCH_SIZE=60`, `MAX_WORKERS=5`,
`TEMPERATURE=0.1`, and `SAVE_EVERY=20`. These values are documented here for
reproducibility, but are not filled into the script automatically. The endpoint
must support chat completions with `response_format={"type": "json_object"}`.

From the `IMLogic` directory:

```bash
python rootmem/build_rootmem.py
```

## Input and output

Each file must contain memories belonging to a single user. Accepted JSON
layouts include a list of memory points, a list of objects with `memory_points`,
an object with a `personas` list, or an object with `memory_points`:

```json
{
  "memory_points": [
    {
      "memory_content": "The user has an active cat that climbs furniture.",
      "timestamp": "2026-01-15"
    }
  ]
}
```

The script reads `memory_content` and `timestamp` (falling back to `time_t`).
As in the reference, it reduces recognizable year-month timestamps to `YYYY-MM`,
sorts the formatted memory strings, and omits memories starting with
`User's name is` or ending with `gender is Male`. String sorting is not a general
date parser; normalize input timestamps to ISO format for chronological order.
The benchmark's raw dialogue JSONL is not a direct input to this constructor.

For each input, the constructor starts with `{"agent_skills": []}` and processes
batches sequentially, providing the previous matrix and the next batch to the
model. Different user files can run concurrently. No query or QA answer is used
during construction.

Each successful batch updates `OUTPUT_DIR/<input-filename>` (using the
original input filename, including its existing `.json` extension). Additional snapshots
use `<input-stem>.batch-<N>.json`. The model returns an `agent_skills` array with
`Skill_Domain`, `Execution_Rules`, and `Logic_Aware_Decision_Region` string fields.
Logs record progress, token usage for successful updates, runtime, and status.

A failed request, invalid JSON, or invalid output schema stops that user's
construction instead of silently skipping memories. The last successful output
remains a **partial checkpoint**, and the run exits with an error if any user
fails. Existing output files are not overwritten by a new run. Automatic
resumption is not implemented.

## Using the constructed RootMem

Install `openai` and `requests`, then fill the configuration block in
`use_rootmem.py` and run:

```bash
python rootmem/use_rootmem.py
```

The script evaluates **one user's QA file per run**. `ROOTMEM_FILE` points to
that user's output from `build_rootmem.py`, `QA_FILE` points to a JSON list of
questions (or an object containing `questions`), and `FIXED_USER_ID` identifies
the same user in the retrieval service. Use a complete construction result,
not a partial checkpoint from a failed run. Set `OUTPUT_FILE` to a new path.
There is no hardcoded person or embedded personal memory matrix.

Configure the router and answer model separately using `ROUTER_API_KEY`,
`ROUTER_BASE_URL`, `ROUTER_MODEL_NAME`, `ROUTER_TEMPERATURE` and the corresponding
`ANSWER_*` variables. Both use OpenAI-compatible chat completions; the router
also requires JSON-object response support. The supplied script's external
`llms.llm_request` implementation was not provided, so the answer API transport
is implemented directly here. Its unspecified model/decoding settings must be
filled by the user; the MCQ prompt itself is unchanged.

The retrieval adapter follows the supplied MemOS HTTP contract. Configure
`RETRIEVAL_URL` with the **complete search endpoint**, `RETRIEVAL_API_KEY`, and
`RETRIEVAL_AUTH_SCHEME` (the reference used `Token`). It posts `query`, `user_id`,
`memory_limit_number`, and `preference_limit_number`, then reads
`data.memory_detail_list[*].memory_value` followed by
`data.preference_detail_list[*].preference`. Memories must already be ingested
in that service. To use another memory system, replace `retrieve_memories`
with an adapter returning a ranked list of strings. No ingestion or local
semantic retriever is implemented by this script.

Set `MODES` to any unique subset of these names:

| Mode | RootMem supplied to answer model | Retrieved memories |
| --- | --- | --- |
| `EMPTY` | None | None |
| `RAG` | None | Up to `MAX_TOTAL_ITEMS` |
| `MATRIX_RAG` | All skills | Up to budget minus all skills |
| `DYNAMIC_MATRIX_RAG` | Skills activated by the query | Up to budget minus activated skills |

As in the reference, each skill or retrieved memory consumes one **item** of
the budget, not a fixed token count. All skills must fit within the configured
budget. The original settings were `TOP_K=20`, `PREF_TOP_K=10`,
`MAX_TOTAL_ITEMS=30`, `MAX_WORKERS=30`, and `ROUTER_TEMPERATURE=0.0`.
Set a positive `REQUEST_TIMEOUT` in seconds, and optionally an integer
`SHUFFLE_SEED` for reproducible shuffling. Numeric settings are left unfilled
in code. `EMPTY` alone does not require retrieval or RootMem configuration.

Routing receives only the query plus each skill's domain and execution rules.
It does not receive retrieved memories, candidate options, or reference answers.
Activated skills are injected in full, including their
`Logic_Aware_Decision_Region`. The original answering prompt uses the label
`Logic_Aware_Constraints`; that wording is preserved even though the JSON field
is named `Logic_Aware_Decision_Region`. Query text is passed verbatim; a separate
`query_time` field is not automatically prepended. Within a question, all modes
share the same shuffled options and retrieval results.

Router failures activate all skills, as in the supplied implementation, and
are explicitly recorded as `fallback_to_all`. Retrieval and answer errors are
recorded rather than silently treated as successful runs. Output contains
`config`, `summary`, and `results` in original question order, with option
mappings, responses, active domains, retrieval budgets, timings, and error
metadata. Failed questions remain in the accuracy denominator, and any
retrieval/answer failures make the process exit with an error after saving.
Router fallback is reported separately from a genuine routing decision.
