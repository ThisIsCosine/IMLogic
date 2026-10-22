"""Construct RootMem incrementally from per-user memory JSON files.

Fill the configuration below, install ``openai``, and run this file.
One input file must contain memories for one user. Accepted JSON layouts:
a list of memory points, a list of objects with ``memory_points``, an object
with ``personas``, or an object with ``memory_points``.

Example numeric settings from the reference implementation:
BATCH_SIZE = 60, MAX_WORKERS = 5, TEMPERATURE = 0.1, SAVE_EVERY = 20.
INPUT_PATTERN can be "conv-*.json" or "*.json". TEST_LIMIT=None uses all
memories. The prompts, memory filtering, month normalization, and string
sorting follow the active implementation supplied with the project.
"""

# ================= User configuration (fill before running) =================
INPUT_DIR = ""
OUTPUT_DIR = ""
LOG_DIR = ""
INPUT_PATTERN = ""

LLM_API_KEY = ""
LLM_BASE_URL = ""
LLM_MODEL_NAME = ""

BATCH_SIZE = None
MAX_WORKERS = None
TEMPERATURE = None
SAVE_EVERY = None
TEST_LIMIT = None  # Optional positive integer; None means all memories.

# ================= Imports and original prompts =================
import json
import math
import re
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

SYSTEM_PROMPT = """# Role: Cognitive Memory Architect

You are an advanced AI Cognitive Memory Architect. Your task is to transform raw, chronological user memory logs containing implicit associations into a highly structured, logic-driven `Cognitive Decision Matrix` in JSON format.

Your ultimate goal is to equip an AI Agent with the ability to make rational, fact-based decisions and prevent "Preference Bias", avoiding scenarios where the AI blindly caters to a user's emotional impulses, past habits, or outdated data.

# Core Directive: Global Timeline Synthesis & Trajectory Tracking
Before generating the JSON, you MUST scan the entire memory array. 
Users evolve in many aspects. `STATE_ANCHORS` must capture this evolution by demonstrating the **Temporal Flow/Trajectory** (e.g., what it was in the past -> the turning point -> what it is NOW). However, your `Execution_Rules` must ALWAYS be strictly governed by their current/latest state.

# Matrix Components & Content Rules:

### 1. Skill_Domain
Do not fragment domains into micro-categories. Dynamically analyze the user data and cluster them into a MAXIMUM of 10 macro-level domains depending on the core pillars of the user's life (e.g., Career & Life Goals, Financial & Asset Management, Health & Stress Management, Social Dynamics, Lifestyle & Consumption Ethics, Entertainment, Pet Interactions, and Travel).

### 2. Execution_Rules
Use fluid, highly contextual, and empathetic prose. Describe specific scenarios based on the user's past vulnerabilities or outdated habits. When defining interventions, you MUST explicitly ground your reasoning in objective realities whenever available (e.g., vetoing high-cost travel specifically because have no money, vetoing sport exercise because of injury). Use commanding yet protective language (e.g., "gently but firmly intervene", "absolutely veto", "actively persuade", "forcibly redirect").

### 3. Logic_Aware_Decision_Region
You must strictly use the following prefix format with square brackets:
- `STATE_ANCHORS: [...]`: Map the flow of changes, including timestamps or logical progression, ending with the absolute current state. (e.g., `[Trajectory: A (Past) -> B (Turning Point) -> C (Current). Current core focus: ...]`)
- `HARD_FACTS: [...]`: Insurmountable physiological limits, severe allergies/phobias, ethical bottom lines, or uncompromising core life goals.

# [Few-Shot Examples]

Example 1:
{
  "Skill_Domain": "Financial & Consumption Mindset",
  "Execution_Rules": "All requests regarding shopping, luxury goods, or lifestyle upgrades must be strictly scrutinized through her current principles of 'minimalism' and 'financial rebuilding phase'. If she is stimulated by social media or nostalgia and impulsively wants to buy designer bags or high-end electronics, you must gently intervene immediately. Point out that this behavior violates the minimalist discipline she established to get out of debt, and actively persuade her to shift her attention to digital asset cleanup or free nature experiences.",
  "Logic_Aware_Decision_Region": "STATE_ANCHORS: [Financial & Consumption Trajectory: Heavy luxury consumer/high debt (2020-2022) -> Bankruptcy crisis -> Currently in financial rebuilding and minimalist practice phase (2024-present). Current available savings: Minimal, reserved for emergencies only.] HARD_FACTS: [Absolutely forbidden to fall into new consumer debt; Life focus has shifted from material possession to spiritual minimalism.]"
}

Example 2:
{
  "Skill_Domain": "Health & Dietary Management",
  "Execution_Rules": "When handling suggestions related to dining, gatherings, or emotional eating, her 'clinical-grade gluten allergy' and 'current strict veganism' must be applied as the highest filtering standards. When the user tries to seek out previously loved traditional baked desserts or high-calorie fast food to vent due to high stress, you must intercept with an uncompromising attitude. Remind her that this will trigger a severe health crisis, and forcibly redirect her to choose gluten-free alternatives that align with her current plant-based anti-inflammatory principles.",
  "Logic_Aware_Decision_Region": "STATE_ANCHORS: [Dietary Trajectory: Standard meat-free diet (2022) -> Diagnosed with severe Celiac disease and shifted to strict plant-based diet (2024-present). Current focus: Gluten-free and anti-inflammatory recovery.] HARD_FACTS: [Clinical-grade gluten allergy (ingestion causes anaphylaxis); Unshakable belief in animal protection.]"
}

# Output Format:
Generate a strictly valid JSON object containing an `agent_skills` array. 
Output the JSON content matching the tone, macro-domain clustering, and temporal accuracy of the examples. 

Now, dynamically cluster the domains, execute timeline synthesis (capturing state flow), eradicate the influence of outdated rules, and generate the JSON Cognitive Decision Matrix based strictly on the provided user memories.
"""


def build_user_prompt(current_matrix, memory_batch, batch_index):
    memories_str = "\n".join(memory_batch)
    user_prompt = f"""
# CURRENT MATRIX STATE (Before Batch {batch_index}):
{json.dumps(current_matrix, indent=2, ensure_ascii=False)}

# NEW BATCH OF MEMORIES TO INTEGRATE:
{memories_str}

Analyze the new memories. Check for temporal evolutions or goal/preference conflicts. OVERWRITE the CURRENT MATRIX to absorb these new insights perfectly. Strictly rewrite to keep the total JSON length from increasing. Aggressively merge and rewrite existing rules to keep the JSON as concise as possible, prioritizing information density over descriptive prose to keep the output aggressively shorter. Output ONLY the updated JSON!
"""
    return user_prompt


def validate_config():
    """Fail before creating files or making API requests if setup is incomplete."""
    required = {
        "INPUT_DIR": INPUT_DIR,
        "OUTPUT_DIR": OUTPUT_DIR,
        "LOG_DIR": LOG_DIR,
        "INPUT_PATTERN": INPUT_PATTERN,
        "LLM_API_KEY": LLM_API_KEY,
        "LLM_BASE_URL": LLM_BASE_URL,
        "LLM_MODEL_NAME": LLM_MODEL_NAME,
    }
    missing = [name for name, value in required.items()
               if not isinstance(value, str) or not value.strip()]
    if missing:
        raise ValueError("Fill configuration values: " + ", ".join(missing))
    for name, value in (("BATCH_SIZE", BATCH_SIZE), ("MAX_WORKERS", MAX_WORKERS),
                        ("SAVE_EVERY", SAVE_EVERY)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if TEST_LIMIT is not None and (type(TEST_LIMIT) is not int or TEST_LIMIT <= 0):
        raise ValueError("TEST_LIMIT must be None or a positive integer.")
    if (isinstance(TEMPERATURE, bool) or not isinstance(TEMPERATURE, (int, float))
            or not math.isfinite(TEMPERATURE) or not 0 <= TEMPERATURE <= 2):
        raise ValueError("TEMPERATURE must be a number between 0 and 2.")
    source = Path(INPUT_DIR).expanduser().resolve()
    output = Path(OUTPUT_DIR).expanduser().resolve()
    logs = Path(LOG_DIR).expanduser().resolve()
    if not source.is_dir():
        raise ValueError("INPUT_DIR must be an existing directory.")
    if len({source, output, logs}) != 3:
        raise ValueError("INPUT_DIR, OUTPUT_DIR, and LOG_DIR must be distinct.")
    if Path(INPUT_PATTERN).name != INPUT_PATTERN or "**" in INPUT_PATTERN:
        raise ValueError("INPUT_PATTERN must select files directly inside INPUT_DIR.")


def extract_memories(file_path):
    with Path(file_path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if isinstance(data, list):
        personas = data
    elif isinstance(data, dict) and "personas" in data:
        personas = data["personas"]
    elif isinstance(data, dict) and "memory_points" in data:
        personas = [data]
    else:
        raise ValueError("Expected per-user memory points or personas JSON.")
    if not isinstance(personas, list):
        raise ValueError("personas must be a list.")
    memories = []
    for persona in personas:
        if not isinstance(persona, dict):
            raise ValueError("Each memory/persona must be an object.")
        points = persona.get("memory_points", [persona])
        if not isinstance(points, list):
            raise ValueError("memory_points must be a list.")
        for point in points:
            if not isinstance(point, dict):
                raise ValueError("Each memory point must be an object.")
            content = (point.get("memory_content") or "").strip()
            raw_timestamp = str(point.get("timestamp") or point.get("time_t") or "unknown").strip()
            match = re.search(r"(\d{4}[-/]\d{2})", raw_timestamp)
            month_timestamp = match.group(1).replace("/", "-") if match else raw_timestamp
            if content and not content.startswith("User's name is") and not content.endswith("gender is Male"):
                memories.append(f"[{month_timestamp}] {content}")
    memories.sort()
    return memories


def write_log(log_file, message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with Path(log_file).open("a", encoding="utf-8") as handle:
        handle.write(f"[{timestamp}] {message}\n")


def save_json(path, value):
    """Replace a checkpoint only after the complete JSON has been written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        temporary.replace(path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def update_matrix_with_llm(client, current_matrix, memory_batch, batch_index, log_file):
    write_log(log_file, f"Starting batch {batch_index}.")
    response = client.chat.completions.create(
        model=LLM_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(current_matrix, memory_batch, batch_index)},
        ],
        response_format={"type": "json_object"},
        temperature=TEMPERATURE,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("The model returned no JSON content.")
    matrix = json.loads(content.strip())
    if not isinstance(matrix, dict) or not isinstance(matrix.get("agent_skills"), list):
        raise ValueError("The model must return a JSON object with an agent_skills array.")
    for skill in matrix["agent_skills"]:
        if not isinstance(skill, dict) or any(
            not isinstance(skill.get(key), str)
            for key in ("Skill_Domain", "Execution_Rules", "Logic_Aware_Decision_Region")
        ):
            raise ValueError("Each skill must contain the three string fields from the prompt.")
    usage = getattr(response, "usage", None)
    stats = {
        "prompt": getattr(usage, "prompt_tokens", 0) or 0,
        "completion": getattr(usage, "completion_tokens", 0) or 0,
        "total": getattr(usage, "total_tokens", 0) or 0,
    }
    write_log(log_file, f"Batch {batch_index} succeeded. Tokens: {stats}")
    return matrix, stats


def process_single_conversation(input_file, client_factory):
    input_file = Path(input_file)
    conv_id = input_file.stem
    output_file = Path(OUTPUT_DIR).expanduser() / input_file.name
    log_file = Path(LOG_DIR).expanduser() / f"{conv_id}.log"
    log_file.write_text(f"RootMem construction: {conv_id}\n", encoding="utf-8")
    start = time.monotonic()
    totals = {"prompt": 0, "completion": 0, "total": 0}
    completed = 0
    try:
        memories = extract_memories(input_file)
        if TEST_LIMIT is not None:
            memories = memories[:TEST_LIMIT]
        if not memories:
            raise ValueError("No usable memories in this input file.")
        matrix = {"agent_skills": []}
        batch_count = (len(memories) + BATCH_SIZE - 1) // BATCH_SIZE
        write_log(log_file, f"Memories: {len(memories)}; batches: {batch_count}")
        with client_factory(api_key=LLM_API_KEY, base_url=LLM_BASE_URL) as client:
            for offset in range(0, len(memories), BATCH_SIZE):
                batch_index = offset // BATCH_SIZE + 1
                matrix, stats = update_matrix_with_llm(
                    client, matrix, memories[offset:offset + BATCH_SIZE], batch_index, log_file,
                )
                completed += 1
                for key in totals:
                    totals[key] += stats[key]
                save_json(output_file, matrix)
                if completed % SAVE_EVERY == 0:
                    snapshot = output_file.parent / f"{conv_id}.batch-{completed}.json"
                    save_json(snapshot, matrix)
        write_log(log_file, "Status: complete")
        return conv_id
    except Exception as exc:
        # Do not silently skip failed batches or log API exception bodies/credentials.
        write_log(log_file, f"Status: failed ({type(exc).__name__}); successful batches: {completed}.")
        raise
    finally:
        write_log(log_file, f"Runtime (s): {time.monotonic() - start:.2f}; successful calls: {completed}; tokens: {totals}")


def main():
    validate_config()
    files = sorted(path for path in Path(INPUT_DIR).expanduser().glob(INPUT_PATTERN)
                   if path.is_file() and path.suffix.lower() == ".json")
    if not files:
        raise ValueError("No input JSON files match INPUT_PATTERN.")
    # Lazy import: configuration and input errors are reported without an API client.
    from openai import OpenAI

    Path(OUTPUT_DIR).expanduser().mkdir(parents=True, exist_ok=True)
    Path(LOG_DIR).expanduser().mkdir(parents=True, exist_ok=True)
    # A new run must not mix old successful outputs with incomplete results.
    for source in files:
        if (Path(OUTPUT_DIR).expanduser() / source.name).exists():
            raise ValueError("An output already exists. Use a fresh OUTPUT_DIR for this run.")
    print(f"Constructing RootMem for {len(files)} files with {MAX_WORKERS} workers.")
    failures = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(process_single_conversation, path, OpenAI): path for path in files}
        for count, future in enumerate(as_completed(futures), start=1):
            path = futures[future]
            try:
                future.result()
                print(f"[{count}/{len(files)}] Complete: {path.name}")
            except Exception as exc:
                failures += 1
                print(f"[{count}/{len(files)}] Failed: {path.name} ({type(exc).__name__}); see its log.")
    if failures:
        raise RuntimeError(f"{failures} file(s) failed; outputs for failed files may be partial.")
    print(f"Completed all {len(files)} files.")


if __name__ == "__main__":
    main()
