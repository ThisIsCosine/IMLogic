# IMLogic

IMLogic is a benchmark for evaluating whether a personalized assistant can use user-specific memories to resolve **implicit logical relevance** in query answering. The benchmark is designed around cases where a user query appears reasonable on the surface, but a different, previously stored memory should constrain or redirect the final answer.

This repository contains the benchmark data, the generation and verification prompts, and reference evaluation scripts for both **memory-level** and **conversation-level** settings. It supports both **multiple-choice QA (MCQ)** and **open-ended QA** evaluation.

It also provides standalone scripts for **RootMem construction, query-time
routing, and MCQ evaluation**. See [RootMem: Construction and Usage](#rootmem-construction-and-usage)
for setup and [the RootMem guide](rootmem/README.md) for detailed input and output formats.

For definitions, proportions, and examples of the seven categories identified
through post hoc analysis, see [Analysis of Logical Associations](docs/mined_logical_associations.md).

## What Is Being Evaluated

IMLogic focuses on the following question:

> Given a current user query, can a model retrieve the **right past memory** and use it to produce a response that is logically aligned with the user's actual situation, rather than following surface-level preferences or generic advice?

The benchmark is built around pairs of memories:

- `memory_s`: exhibits relatively high semantic overlap with 𝑞, but alone is insufficient to support the most appropriate personalized response. 
- `memory_l`: semantically distant from 𝑞 but contains information that is logically important.

## Repository Contents

```text
IMLogic/
|- docs/
|  `- mined_logical_associations.md
|- dataset/
|  |- dialogue/
|  |  `- HaluMem-Medium.jsonl
|  `- qa/
|     |- qa_user00.json
|     |- qa_user01.json
|     `- ...
|- generate/
|  |- memory_lagging.py
|  |- implicit_logical_memory_pair_mining.py
|  `- qa_generating.py
|- eval/
|  |- evaluate_common.py
|  |- evaluate_on_memory.py
|  |- evaluate_on_conversation.py
|  `- judge_openended.py
|- rootmem/
|  |- build_rootmem.py
|  |- use_rootmem.py
|  `- README.md
`- README.md
```

Current release summary:

- `dataset/dialogue/HaluMem-Medium.jsonl` contains `20` dialogue records
- `dataset/qa/qa_userXX.json` contains `20` aligned QA files, one per dialogue record

The alignment rule is:

- dialogue item `0` corresponds to `qa_user00.json`
- dialogue item `1` corresponds to `qa_user01.json`
- and so on

## Benchmark Data Format

### Dialogue File

`dataset/dialogue/HaluMem-Medium.jsonl` is a JSONL file. Each line is one user trajectory and contains:

- `uuid`: user identifier
- `persona_info`: summary of persona metadata
- `sessions`: a list of multi-turn sessions

Each `session` typically includes:

- `start_time`, `end_time`
- `memory_points`: stored memories extracted or maintained by the source system
- `dialogue`: the original user-assistant interaction turns
- `questions`: auxiliary session-level questions in the source data

Each `memory_point` typically contains:

- `memory_content`
- `memory_lype`
- `timestamp`
- `importance`
- `memory_source`
- `index`

Minimal schema sketch:

```json
{
  "uuid": "user-uuid",
  "persona_info": "...",
  "sessions": [
    {
      "start_time": "...",
      "end_time": "...",
      "memory_points": [
        {
          "index": 1,
          "memory_content": "...",
          "memory_lype": "...",
          "timestamp": "...",
          "importance": 0.75
        }
      ],
      "dialogue": [
        {
          "role": "user",
          "content": "...",
          "timestamp": "...",
          "dialogue_turn": 0
        }
      ]
    }
  ]
}
```

### QA File

Each `dataset/qa/qa_userXX.json` file is a list of benchmark instances for the corresponding user.

Each QA item contains:

- `memory_s`: semantic memory but alone is insufficient to support the most appropriate personalized response
- `memory_l`: logical constraint and supports the most appropriate personalized response
- `query`: the user query to answer
- `options`: the MCQ candidates
- `query_time`: timestamp associated with the query
- `category`: evaluation category

The `options` object contains:

- `Correct`: the logically correct answer when `memory_l` is used properly
- `Trap_Preference`: answer based on the semantic memory only
- `Trap_Fabrication`: answer with hallucinated justification
- `Trap_Generic`: generic advice without relevant memory grounding

Minimal schema sketch:

```json
{
  "memory_s": "...",
  "memory_l": "...",
  "query_type": "Advice",
  "query": "...",
  "options": {
    "Correct": "...",
    "Trap_Preference": "...",
    "Trap_Fabrication": "...",
    "Trap_Generic": "..."
  },
  "query_time": "...",
  "category": "Goal/Mission Alignment"
}
```

For open-ended judgment, `memory_l` is used as **Memory l** in the judge prompt.

## Benchmark Construction Pipeline

The benchmark construction process is organized into three stages under `generate/`.

### 1. Memory Tagging

File: `generate/memory_lagging.py`

This script assigns each memory a top-level semantic tag using an LLM-based taxonomy. The current taxonomy includes:

- `Personal_Background`
- `Assets`
- `Past_Experience`
- `States`
- `Preferences`
- `Opinions`
- `Goals`
- `Plans`
- `Social_Relationships`
- `Others`

### 2. Implicit Logical Pair Mining

File: `generate/implicit_logical_memory_pair_mining.py`

This script mines candidate `(L, S)` pairs where:

- `L` is the logically constraining memory
- `S` is a competing or distractor memory

The mining prompt asks whether `S` conflicts with or is logically dominated by `L`, rather than merely being topically unrelated.

### 3. QA Generation and Verification

File: `generate/qa_generating.py`

This script generates benchmark questions and four candidate options. It also contains:

- a **generator prompt**
- a **judger prompt**
- a **curator prompt**

These prompts are used to create and verify high-quality adversarial benchmark instances with:

- strong alignment to the distractor memory
- zero leakage of the true constraint into the query
- balanced candidate options

## Evaluation Settings

IMLogic supports two evaluation settings.

### 1. Memory-Level Evaluation

File: `eval/evaluate_on_memory.py`

This setting assumes a memory bank already exists. The evaluation pipeline is:

1. build a memory bank from stored `memory_points`
2. retrieve relevant memories for the current query
3. answer the QA using an LLM

The script provides a **user integration point** for the retriever or memory
system being evaluated. Connect your method through the following function:

```python
def retrieve_memories(query, memory_bank, qa_item, top_k):
    # User integration point: connect the retriever or memory system under evaluation.
    # Return up to top_k memory dictionaries with a memory_content field.
    return []
```

The default returns an empty list and supplies no retrieved memories to the
answer model. Replace this behavior with your method to evaluate retrieval.

### 2. Conversation-Level Evaluation

File: `eval/evaluate_on_conversation.py`

This setting starts from raw dialogue instead of a ready-made memory bank. The pipeline is:

1. extract memories from dialogue
2. retrieve relevant extracted memories
3. answer the QA using an LLM

The script provides two **user integration points** for connecting a memory
system's dialogue ingestion/extraction and query-time retrieval:

```python
def extract_memories_from_dialogue(dialogue_item):
    # User integration point: connect your memory system's dialogue ingestion/extraction.
    # Return memory dictionaries with a memory_content field.
    return []

def retrieve_memories(query, memory_bank, qa_item, top_k):
    # User integration point: connect your memory system's query-time retrieval.
    # Return up to top_k memory dictionaries with a memory_content field.
    return []
```

Both functions return empty lists by default. Connect both stages to evaluate
your memory system; otherwise, the answer model receives no retrieved memories.
Additional memory metadata, such as timestamps, may be included in the returned
dictionaries. These integration points let different methods share the same
benchmark answering and scoring protocol.

## QA Modes

The repository supports two answer modes.

### MCQ Mode

In MCQ mode, the model is given the query, retrieved context, and four candidate responses. It must output only the best option letter.

Implementation:

- prompt template: `eval/evaluate_common.py`
- runner: `eval/evaluate_on_memory.py` or `eval/evaluate_on_conversation.py`
- CLI flag: `--answer-mode multiple_choice`

### Open-Ended Mode

In open-ended mode, the model is given the query and retrieved context and must generate one concise answer sentence.

Implementation:

- prompt template: `eval/evaluate_common.py`
- runner: `eval/evaluate_on_memory.py` or `eval/evaluate_on_conversation.py`
- CLI flag: `--answer-mode open_ended`

## LLM Answering Prompts

The repository currently includes two evaluation-time prompts:

- **Cognitive Evaluation Agent** for MCQ selection
- **Cognitive Generative Agent** for open-ended generation

Both prompts are implemented in:

- `eval/evaluate_common.py`

They share the same basic structure:

- `Active_Root_Memory_Units`
- `global_state`
- retrieved context
- timestamped user query

The evaluation scripts expect an **OpenAI-compatible API**.

## Open-Ended LLM-as-Judge Evaluation

File: `eval/judge_openended.py`

This script evaluates open-ended model outputs using an LLM judge. For each QA item, it uses:

- `memory_l` as `Memory l`
- `options["Correct"]` as `Reference Answer`
- the original `query` as `User Question`
- the generated answer as `Model Prediction`

The judge prompt is intentionally strict:

- the answer must explicitly reflect the relevant constraint
- the answer must align with the logical direction of the reference answer
- generic answers should be marked incorrect

### Judge Output

The judge returns JSON of the form:

```json
{
  "is_correct": true,
  "reasoning": "..."
}
```

The saved result file includes:

- per-question judge result
- `memory_l`
- `reference_answer`
- `model_prediction`
- aggregate accuracy statistics

## How to Run the Benchmark

Run the commands below from the `IMLogic` directory. Replace all angle-bracket
placeholders with your own model names and paths. Configure the API credentials
and endpoint for your service. Connect the evaluation integration points
described above before evaluating your retrieval method or memory system.

### Prerequisites

The evaluation scripts assume:

- Python 3.10+
- `openai` Python package
- access to an OpenAI-compatible API endpoint

Typical environment variables:

```bash
export OPENAI_API_KEY="<your-api-key>"
export OPENAI_BASE_URL="<your-api-base-url>"
export OPENAI_MODEL="<your-model-name>"
```

On Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="<your-api-key>"
$env:OPENAI_BASE_URL="<your-api-base-url>"
$env:OPENAI_MODEL="<your-model-name>"
```

### 1. Memory-Level MCQ Evaluation

```bash
python eval/evaluate_on_memory.py \
  --answer-mode multiple_choice \
  --dialogue-file "<path-to-dialogue-jsonl>" \
  --qa-dir "<path-to-qa-directory>" \
  --output "<path-to-mcq-results-json>" \
  --model "<your-answer-model>"
```

### 2. Memory-Level Open-Ended Evaluation

```bash
python eval/evaluate_on_memory.py \
  --answer-mode open_ended \
  --dialogue-file "<path-to-dialogue-jsonl>" \
  --qa-dir "<path-to-qa-directory>" \
  --output "<path-to-open-ended-results-json>" \
  --model "<your-answer-model>"
```

### 3. Conversation-Level MCQ Evaluation

```bash
python eval/evaluate_on_conversation.py \
  --answer-mode multiple_choice \
  --dialogue-file "<path-to-dialogue-jsonl>" \
  --qa-dir "<path-to-qa-directory>" \
  --output "<path-to-mcq-results-json>" \
  --model "<your-answer-model>"
```

### 4. Conversation-Level Open-Ended Evaluation

```bash
python eval/evaluate_on_conversation.py \
  --answer-mode open_ended \
  --dialogue-file "<path-to-dialogue-jsonl>" \
  --qa-dir "<path-to-qa-directory>" \
  --output "<path-to-open-ended-results-json>" \
  --model "<your-answer-model>"
```

### 5. Judge Open-Ended Predictions

```bash
python eval/judge_openended.py \
  --predictions-file "<path-to-open-ended-results-json>" \
  --qa-dir "<path-to-qa-directory>" \
  --output "<path-to-judge-results-json>" \
  --model "<your-judge-model>"
```

### Useful Optional Flags

- `--user-index`: evaluate only one aligned user trajectory
- `--question-limit`: evaluate only the first `N` questions per user
- `--top-k`: maximum number of retrieved memories passed to the answer model
- `--global-state-file`: override the automatically constructed global state block
- `--output`: set a custom result file path

## RootMem: Construction and Usage

RootMem construction and usage are separate stages:

1. Prepare per-user memory JSON files using your memory system.
2. Run `rootmem/build_rootmem.py` to incrementally construct RootMem for each user.
3. Run `rootmem/use_rootmem.py` to route the constructed RootMem for a query,
   combine activated skills with retrieved memories, and answer MCQs.

These scripts are standalone. The `eval/` runners retain their user integration
points and do not automatically load or invoke RootMem. Open-ended evaluation
is available through `eval/`; the standalone RootMem usage script currently
implements MCQ evaluation only.

### Setup

Use Python 3.10+ and install:

```bash
pip install openai requests
```

**Fill the global configuration block at the beginning of each RootMem script
before running it.** Paths, API keys, API endpoints, model identifiers, and
required numeric settings are intentionally unfilled. Choose the construction,
routing, and answering models yourself. These scripts read their configuration
from those global variables; the `OPENAI_*` environment variables shown for the
benchmark runners are not automatically loaded by the RootMem scripts.

### 1. Construct RootMem

Configure [rootmem/build_rootmem.py](rootmem/build_rootmem.py):

| Configuration | What to fill |
| --- | --- |
| `INPUT_DIR` | Directory of memory JSON files, one user per file |
| `INPUT_PATTERN` | Input filename pattern, such as `*.json` |
| `OUTPUT_DIR` | Separate directory for constructed RootMem JSON files |
| `LOG_DIR` | Separate directory for per-user construction logs |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL_NAME` | Your construction model credentials, endpoint, and identifier |
| `BATCH_SIZE`, `MAX_WORKERS` | Positive integers for memories per batch and concurrent users |
| `TEMPERATURE` | Sampling temperature supported by your model |
| `SAVE_EVERY` | Positive integer for the numbered snapshot interval |
| `TEST_LIMIT` | Optional memory limit per user; leave `None` to process all memories |

Each input file contains a list of memory points or an object with
`memory_points`. The supported nested `personas` layout and a JSON example are
documented in [the RootMem guide](rootmem/README.md). Memory points contain
`memory_content` and a `timestamp` or `time_t`. The raw benchmark dialogue JSONL
must first be converted into per-user memory files; it is not a direct input.

```bash
python rootmem/build_rootmem.py
```

The constructor processes batches sequentially for each user, updating the
previous RootMem with new memories. It uses no evaluation queries or reference
answers. Each output retains the input filename and contains an `agent_skills`
array with `Skill_Domain`, `Execution_Rules`, and `Logic_Aware_Decision_Region`.
Use a fresh output directory for each run. A failed construction may leave a
partial checkpoint; use completed results for the next stage.

### 2. Route RootMem and Answer Queries

Configure [rootmem/use_rootmem.py](rootmem/use_rootmem.py):

| Configuration | What to fill |
| --- | --- |
| `QA_FILE` | One user's QA JSON file containing queries and MCQ options |
| `ROOTMEM_FILE` | That user's completed JSON output from the construction stage |
| `OUTPUT_FILE` | New output path for evaluation results |
| `FIXED_USER_ID` | The same user's identifier in your retrieval service |
| `ROUTER_API_KEY`, `ROUTER_BASE_URL`, `ROUTER_MODEL_NAME`, `ROUTER_TEMPERATURE` | Your routing model configuration |
| `ANSWER_API_KEY`, `ANSWER_BASE_URL`, `ANSWER_MODEL_NAME`, `ANSWER_TEMPERATURE` | Your answering model configuration |
| `RETRIEVAL_URL`, `RETRIEVAL_API_KEY`, `RETRIEVAL_AUTH_SCHEME` | Complete search endpoint, credentials, and authorization scheme for your memory service |
| `TOP_K`, `PREF_TOP_K` | Maximum facts and preferences requested from the retrieval service |
| `MAX_TOTAL_ITEMS` | Combined budget for RootMem skills and retrieved memories |
| `MAX_WORKERS`, `REQUEST_TIMEOUT` | Concurrent questions and request timeout in seconds |
| `MODES` | A nonempty list of evaluation modes from the table below |
| `SHUFFLE_SEED` | Optional integer for reproducible MCQ option shuffling |

`QA_FILE`, `ROOTMEM_FILE`, and `FIXED_USER_ID` must refer to the same user.
The script runs one user's QA file at a time. The retrieval adapter follows the
MemOS search API contract described in [the RootMem guide](rootmem/README.md).
For that adapter, memories must already exist in the service. To connect a
different memory system, replace `retrieve_memories()` with an adapter that
returns a ranked list of memory strings.

| Mode | RootMem context | Retrieval budget |
| --- | --- | --- |
| `EMPTY` | None | None |
| `RAG` | None | `MAX_TOTAL_ITEMS` |
| `MATRIX_RAG` | All skills | `MAX_TOTAL_ITEMS` minus the number of skills |
| `DYNAMIC_MATRIX_RAG` | Query-activated skills | `MAX_TOTAL_ITEMS` minus the number of activated skills |

Select modes by filling `MODES`, for example `["DYNAMIC_MATRIX_RAG"]` to use
query-time routing. The budget counts items, not tokens. `EMPTY` alone does not
require RootMem, router, or retrieval-service configuration.

```bash
python rootmem/use_rootmem.py
```

Routing receives the query and each skill's domain and execution rules. It
does not receive MCQ options or reference answers. The answer model receives
the activated skills, retrieved memories, query, and shuffled options. Router
failures fall back to all skills and are recorded explicitly. Retrieval and
answering errors are also recorded.

The output contains `config`, `summary`, and `results`, including per-mode MCQ
accuracy, errors, retrieved memories, activated domains, and option mappings.
Failed questions remain in the denominator. Resolve any reported evaluation
errors before using the results in an experiment report.

## Result Files

The evaluation scripts save JSON result files containing:

- `config`
- `summary`
- `users`

Per-user records include:

- question metadata
- model outputs
- candidate options for MCQ
- retrieved memories
- answer-level correctness for MCQ

Judge result files additionally include:

- LLM-as-judge verdicts for open-ended answers
- reasoning from the judge

## Reproducibility Notes

To reproduce the full IMLogic pipeline, researchers should specify:

1. the memory extraction method
2. the retrieval method
3. the answer model
4. the answer mode (`multiple_choice` or `open_ended`)
5. the judge model for open-ended evaluation

At the moment, the repository provides:

- benchmark instances
- generation scripts
- verification prompts
- evaluation scripts
- LLM-as-judge support for open-ended QA
- standalone RootMem construction, routing, and MCQ evaluation scripts

The **memory extraction** and **memory retrieval** functions are intentional
user integration points for the method being evaluated. Their default empty
outputs do not run an extractor or retriever. Users should connect their own
implementations before reporting results for a retrieval method or memory system.

## Citation and Usage

If you use IMLogic in your experiments, please cite the corresponding paper once the bibliographic entry is released. If you adapt the benchmark with a custom extractor, retriever, or judge model, we recommend documenting those components explicitly for reproducibility.
