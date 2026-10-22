"""Route constructed RootMem and evaluate MCQs with the original usage prompts.

Fill the configuration below. ROOTMEM_FILE, QA_FILE, and FIXED_USER_ID must
refer to the same user. ROOTMEM_FILE accepts build_rootmem.py's JSON output.
The reference modes are EMPTY, RAG, MATRIX_RAG, DYNAMIC_MATRIX_RAG.
No API clients are created or requests made when importing this module.
"""

# ================= User configuration (fill before running) =================
QA_FILE = ""
ROOTMEM_FILE = ""
OUTPUT_FILE = ""
FIXED_USER_ID = ""

# Retrieval endpoint must implement the reference MemOS search/memory contract.
RETRIEVAL_URL = ""
RETRIEVAL_API_KEY = ""
RETRIEVAL_AUTH_SCHEME = ""  # Reference service used "Token".

ROUTER_API_KEY = ""
ROUTER_BASE_URL = ""
ROUTER_MODEL_NAME = ""
ROUTER_TEMPERATURE = None

ANSWER_API_KEY = ""
ANSWER_BASE_URL = ""
ANSWER_MODEL_NAME = ""
ANSWER_TEMPERATURE = None

TOP_K = None
PREF_TOP_K = None
MAX_TOTAL_ITEMS = None
MAX_WORKERS = None
REQUEST_TIMEOUT = None
MODES = []  # Fill with one or more of the four reference mode names.
SHUFFLE_SEED = None  # Optional integer for reproducible option shuffling.

import copy
import json
import math
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import ExitStack
from pathlib import Path

try:
    from .build_rootmem import save_json
except ImportError:
    from build_rootmem import save_json

ROUTER_PROMPT = """
You are a Cognitive Router for an AI Agent.
Your objective is to perform reasonable skill recall based ONLY on the User's Query. You must select ALL Cognitive Skills that will exert logical constraints or act as boundaries for the user's request.

# AVAILABLE SKILLS:
{skills_meta}

# USER QUERY:
"{query}"

# INSTRUCTION (ROUTING DECISION PROTOCOL):
1. Deeply evaluate the user's query to identify not only its literal needs but also the potential implications the request might trigger (e.g., does it involve resource consumption, energy allocation, value judgments, or risk assumption?).
2. Systematically compare the analyzed intent against each of the provided [AVAILABLE SKILLS]. The core evaluation criterion is: Would this skill's `Execution_Rules` logically constrain, modify, or provide a protective boundary for the current response?
3. Complex queries often span multiple domains (e.g., a question about a "vacation" typically involves both financial planning and psychological relaxation). 
4. Strictly return a valid JSON object containing ONLY the list of selected `Skill_Domain` names. Do not output any analytical process, reasoning, or explanatory text. 
5. You are encouraged to recall multiple relevant skill domains (even little relevance) to ensure a comprehensive response.

Format:
{{
  "activated_domains": ["Skill_Domain_1", "Skill_Domain_2", ...]
}}
"""
PROMPT_MCQ = """
You are an advanced conversational agent equipped with a Cognitive Skill system for deep personalization and logical reasoning.

# ACTIVE_SKILLS:
Dynamically activated cognitive skills providing the logical framework for time-aware personalization.
- `Execution_Rules`: The deterministic control laws and conflict resolution protocols.
- `Logic_Aware_Constraints`: The temporal parameter matrix bounding the Agent's feasible decision region.

{global_state}

# RETRIEVED_CONTEXT:
The user's historical memories.
{context}

# USER QUERY:
The `[Timestamp]` at the beginning of the query indicates the exact time of the user's current request.
"{query}"

# CANDIDATE RESPONSES:
{options_str}

# INSTRUCTION:
Evaluate the candidate responses and select the optimal choice by adhering to the following execution protocol:

1. Logic-Aware Personalization: Incorporate the `Execution_Rules` and `Logic_Aware_Constraints` defined in ACTIVE_SKILLS to deeply contextualize and tailor the response (bypass this step if empty).
2. Evidence-Grounded Alignment: Select the response that maximally aligns with the user's actual needs, relying thoughtfully on the provided evidence.

# OUTPUT FORMAT:
Output ONLY the letter of the best option (e.g., A, B, C, or D). Do not provide any explanation.
"""

SUPPORTED_MODES = ("EMPTY", "RAG", "MATRIX_RAG", "DYNAMIC_MATRIX_RAG")


def validate_config():
    if not MODES or len(set(MODES)) != len(MODES) or any(mode not in SUPPORTED_MODES for mode in MODES):
        raise ValueError("MODES must contain unique names from SUPPORTED_MODES.")
    required = {
        "QA_FILE": QA_FILE, "OUTPUT_FILE": OUTPUT_FILE,
        "ANSWER_API_KEY": ANSWER_API_KEY, "ANSWER_BASE_URL": ANSWER_BASE_URL,
        "ANSWER_MODEL_NAME": ANSWER_MODEL_NAME,
    }
    if any(mode != "EMPTY" for mode in MODES):
        required.update(FIXED_USER_ID=FIXED_USER_ID, RETRIEVAL_URL=RETRIEVAL_URL,
                        RETRIEVAL_API_KEY=RETRIEVAL_API_KEY,
                        RETRIEVAL_AUTH_SCHEME=RETRIEVAL_AUTH_SCHEME)
    if any(mode in MODES for mode in ("MATRIX_RAG", "DYNAMIC_MATRIX_RAG")):
        required["ROOTMEM_FILE"] = ROOTMEM_FILE
    if "DYNAMIC_MATRIX_RAG" in MODES:
        required.update(ROUTER_API_KEY=ROUTER_API_KEY, ROUTER_BASE_URL=ROUTER_BASE_URL,
                        ROUTER_MODEL_NAME=ROUTER_MODEL_NAME)
    missing = [key for key, value in required.items() if not isinstance(value, str) or not value.strip()]
    if missing:
        raise ValueError("Fill configuration values: " + ", ".join(missing))
    for name, value in (("MAX_WORKERS", MAX_WORKERS), ("MAX_TOTAL_ITEMS", MAX_TOTAL_ITEMS)):
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer.")
    if any(mode != "EMPTY" for mode in MODES):
        for name, value in (("TOP_K", TOP_K), ("PREF_TOP_K", PREF_TOP_K)):
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer.")
    if (isinstance(REQUEST_TIMEOUT, bool) or not isinstance(REQUEST_TIMEOUT, (int, float))
            or not math.isfinite(REQUEST_TIMEOUT) or REQUEST_TIMEOUT <= 0):
        raise ValueError("REQUEST_TIMEOUT must be a positive number of seconds.")
    temperatures = [("ANSWER_TEMPERATURE", ANSWER_TEMPERATURE)]
    if "DYNAMIC_MATRIX_RAG" in MODES:
        temperatures.append(("ROUTER_TEMPERATURE", ROUTER_TEMPERATURE))
    for name, value in temperatures:
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 2:
            raise ValueError(f"{name} must be a number between 0 and 2.")
    if SHUFFLE_SEED is not None and type(SHUFFLE_SEED) is not int:
        raise ValueError("SHUFFLE_SEED must be None or an integer.")
    for name in ("QA_FILE", "ROOTMEM_FILE"):
        if name in required and not Path(required[name]).expanduser().is_file():
            raise ValueError(f"{name} must be an existing file.")
    if Path(OUTPUT_FILE).expanduser().exists():
        raise ValueError("OUTPUT_FILE already exists. Choose a new output path.")


def load_skills(path):
    matrix = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    if not isinstance(matrix, dict) or not isinstance(matrix.get("agent_skills"), list):
        raise ValueError("ROOTMEM_FILE must contain an agent_skills array.")
    skills = matrix["agent_skills"]
    names = set()
    for skill in skills:
        if not isinstance(skill, dict) or any(
            not isinstance(skill.get(key), str) or not skill[key].strip()
            for key in ("Skill_Domain", "Execution_Rules", "Logic_Aware_Decision_Region")
        ):
            raise ValueError("Invalid RootMem skill fields.")
        if skill["Skill_Domain"] in names:
            raise ValueError("RootMem Skill_Domain names must be unique.")
        names.add(skill["Skill_Domain"])
    # The reference includes every activated skill and budgets RAG around it.
    if len(skills) > MAX_TOTAL_ITEMS:
        raise ValueError("MAX_TOTAL_ITEMS must accommodate all RootMem skills.")
    return skills


def load_questions(path):
    raw = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    questions = raw if isinstance(raw, list) else raw.get("questions") if isinstance(raw, dict) else None
    if not isinstance(questions, list) or not questions:
        raise ValueError("QA_FILE must contain a nonempty list or a questions list.")
    for item in questions:
        if not isinstance(item, dict) or not isinstance(item.get("query"), str) or not item["query"].strip():
            raise ValueError("Each QA item must have a nonempty query.")
        options = item.get("options")
        if not isinstance(options, dict) or "Correct" not in options or not 2 <= len(options) <= 26:
            raise ValueError("Each QA item needs 2–26 options including Correct.")
        if any(not isinstance(value, str) or not value.strip() for value in options.values()):
            raise ValueError("Option text must be nonempty strings.")
    return questions


def format_options(options_dict, rng):
    items = list(options_dict.items())
    rng.shuffle(items)
    lines, mapping, correct_letter = [], {}, None
    for idx, (type_key, text) in enumerate(items):
        letter = chr(65 + idx)
        lines.append(f"{letter}. {text}")
        mapping[letter] = type_key
        if type_key == "Correct":
            correct_letter = letter
    return "\n".join(lines), mapping, correct_letter


def get_active_skills(query, skills, client):
    router_meta = [{"Skill_Domain": s["Skill_Domain"], "Execution_Rules": s["Execution_Rules"]} for s in skills]
    prompt = ROUTER_PROMPT.format(
        skills_meta=json.dumps(router_meta, indent=2, ensure_ascii=False), query=query,
    )
    known = [skill["Skill_Domain"] for skill in skills]
    try:
        response = client.chat.completions.create(
            model=ROUTER_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=ROUTER_TEMPERATURE,
        )
        data = json.loads(response.choices[0].message.content)
        domains = data["activated_domains"]
        if not isinstance(domains, list) or any(not isinstance(d, str) or d not in known for d in domains):
            raise ValueError("Invalid activated_domains returned by router.")
        return list(dict.fromkeys(domains)), {"fallback_to_all": False, "error_type": None}
    except Exception as exc:
        # Preserve the reference fallback, but expose it in the saved results.
        return known, {"fallback_to_all": True, "error_type": type(exc).__name__}


def retrieve_memories(query, post):
    response = post(
        RETRIEVAL_URL,
        json={"query": query, "user_id": FIXED_USER_ID,
              "memory_limit_number": TOP_K, "preference_limit_number": PREF_TOP_K},
        headers={"Content-Type": "application/json",
                 "Authorization": f"{RETRIEVAL_AUTH_SCHEME} {RETRIEVAL_API_KEY}"},
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    body = response.json()
    data = body.get("data") if isinstance(body, dict) else None
    if not isinstance(data, dict):
        raise ValueError("Retrieval response must contain a data object.")
    memories = []
    for list_key, text_key in (("memory_detail_list", "memory_value"), ("preference_detail_list", "preference")):
        items = data.get(list_key, []) or []
        if not isinstance(items, list):
            raise ValueError("Invalid retrieval result list.")
        for item in items:
            if not isinstance(item, dict):
                raise ValueError("Invalid retrieval result item.")
            text = item.get(text_key)
            if text:
                if not isinstance(text, str):
                    raise ValueError("Retrieved memory must be text.")
                memories.append(text)
    return memories


def build_mode_context(mode, skills, active_domains, retrieved):
    if mode == "EMPTY":
        return "No active skills triggered.", "No retrieved context.", [], 0
    if mode == "RAG":
        active = []
    elif mode == "MATRIX_RAG":
        active = skills
    elif mode == "DYNAMIC_MATRIX_RAG":
        active = [skill for skill in skills if skill["Skill_Domain"] in active_domains]
    else:
        raise ValueError("Unsupported evaluation mode.")
    budget = max(0, MAX_TOTAL_ITEMS - len(active))
    # MATRIX_RAG retains an empty JSON object/array even if skills is empty.
    state = (json.dumps({"agent_skills": active}, indent=2, ensure_ascii=False)
             if active or mode == "MATRIX_RAG" else "No active skills triggered.")
    context = "\n- ".join(retrieved[:budget]) if retrieved else "No specific past memories found."
    return state, context, active, budget


def search_and_ask(item, index, skills, client_factory, post):
    query = item["query"]  # Preserve original query text; do not append a timestamp.
    rng = random.Random(None if SHUFFLE_SEED is None else SHUFFLE_SEED + index)
    options, mapping, correct_letter = format_options(item["options"], rng)
    domains, router_info = [], {"fallback_to_all": False, "error_type": None}
    retrieved, retrieval_error, route_ms, search_ms = [], None, 0.0, 0.0
    mode_results = {}
    with ExitStack() as stack:
        answer_client = stack.enter_context(client_factory(
            api_key=ANSWER_API_KEY, base_url=ANSWER_BASE_URL, timeout=REQUEST_TIMEOUT,
        ))
        if "DYNAMIC_MATRIX_RAG" in MODES:
            router_client = stack.enter_context(client_factory(
                api_key=ROUTER_API_KEY, base_url=ROUTER_BASE_URL, timeout=REQUEST_TIMEOUT,
            ))
            start = time.monotonic()
            domains, router_info = get_active_skills(query, skills, router_client)
            route_ms = (time.monotonic() - start) * 1000
        if any(mode != "EMPTY" for mode in MODES):
            start = time.monotonic()
            try:
                retrieved = retrieve_memories(query, post)
            except Exception as exc:
                retrieval_error = type(exc).__name__
            search_ms = (time.monotonic() - start) * 1000
        for mode in MODES:
            state, context, active, budget = build_mode_context(mode, skills, domains, retrieved)
            result = {"model_choice": "N/A", "is_correct": False, "status": "error",
                      "active_domains": [s["Skill_Domain"] for s in active],
                      "rag_budget": budget, "retrieved_count_used": min(budget, len(retrieved))}
            if mode == "DYNAMIC_MATRIX_RAG":
                result["routed_domains"] = domains
            if mode != "EMPTY" and retrieval_error:
                result.update(error_stage="retrieval", error_type=retrieval_error)
                mode_results[mode] = result
                continue
            prompt = PROMPT_MCQ.format(global_state=state, context=context, query=query, options_str=options)
            try:
                response = answer_client.chat.completions.create(
                    model=ANSWER_MODEL_NAME, messages=[{"role": "user", "content": prompt}],
                    temperature=ANSWER_TEMPERATURE,
                )
                raw = response.choices[0].message.content or ""
                choice = raw.strip()[0].upper() if raw.strip() else "N/A"
                result["raw_output"] = raw
                if choice not in mapping:
                    raise ValueError("Answer did not start with a valid option letter.")
                result.update(model_choice=choice, is_correct=choice == correct_letter, status="success")
            except Exception as exc:
                result.update(error_stage="answer", error_type=type(exc).__name__)
            mode_results[mode] = result
    data = copy.deepcopy(item)
    data["ablation_eval"] = {
        "retrieved_memories": retrieved, "shuffled_options_map": mapping,
        "correct_letter_ground_truth": correct_letter, "mode_results": mode_results,
        "router": router_info, "retrieval_error_type": retrieval_error,
        "route_ms": route_ms, "search_ms": search_ms,
    }
    return data


def summarize(results):
    summaries = {}
    for mode in MODES:
        correct = errors = 0
        for item in results:
            record = item.get("ablation_eval", {}).get("mode_results", {}).get(mode)
            if record is None or record["status"] != "success":
                errors += 1
            elif record["is_correct"]:
                correct += 1
        summaries[mode] = {"correct": correct, "total": len(results), "errors": errors,
                           "accuracy_percent": 100 * correct / len(results) if results else 0.0}
    return summaries


def main():
    validate_config()
    questions = load_questions(QA_FILE)
    skills = load_skills(ROOTMEM_FILE) if any(m in MODES for m in ("MATRIX_RAG", "DYNAMIC_MATRIX_RAG")) else []
    from openai import OpenAI
    if any(mode != "EMPTY" for mode in MODES):
        import requests
        post = requests.post
    else:
        post = None
    results = [None] * len(questions)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(search_and_ask, item, index, skills, OpenAI, post): index
                   for index, item in enumerate(questions)}
        for count, future in enumerate(as_completed(futures), start=1):
            index = futures[future]
            try:
                results[index] = future.result()
            except Exception as exc:
                results[index] = copy.deepcopy(questions[index])
                results[index]["evaluation_error"] = {"error_type": type(exc).__name__}
            print(f"Processed {count}/{len(questions)} questions.")
    summary = summarize(results)
    save_json(Path(OUTPUT_FILE).expanduser(), {
        "config": {"qa_file": QA_FILE, "rootmem_file": ROOTMEM_FILE,
                   "user_id": FIXED_USER_ID, "modes": MODES,
                   "router_model": ROUTER_MODEL_NAME, "answer_model": ANSWER_MODEL_NAME,
                   "router_temperature": ROUTER_TEMPERATURE, "answer_temperature": ANSWER_TEMPERATURE,
                   "top_k": TOP_K, "pref_top_k": PREF_TOP_K, "max_total_items": MAX_TOTAL_ITEMS,
                   "shuffle_seed": SHUFFLE_SEED},
        "summary": summary, "results": results,
    })
    for mode, stats in summary.items():
        print(f"{mode}: {stats['correct']}/{stats['total']} ({stats['accuracy_percent']:.2f}%); errors: {stats['errors']}")
    if any(stats["errors"] for stats in summary.values()):
        raise RuntimeError("Evaluation contains errors; inspect saved results before reporting accuracy.")


if __name__ == "__main__":
    main()
