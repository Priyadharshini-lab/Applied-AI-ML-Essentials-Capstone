"""
=====================================================================
 PART 4 — LLM-Powered Feature
 TRACK CHOSEN: (C) Model Prediction Explanation Pipeline
 Input : best_model.pkl (produced by Part 3's part3_ensembles.py)
=====================================================================
Run with:  python3 part4_llm_explain.py

Requires an environment variable holding your LLM API key, e.g.:
    export LLM_API_KEY="sk-or-..."
Optionally override the endpoint / model:
    export LLM_API_URL="https://openrouter.ai/api/v1/chat/completions"
    export LLM_MODEL="openai/gpt-oss-20b:free"

Produces:
  - console output for every task
  - results/pii_guardrail_test.csv
  - results/temperature_comparison.csv
  - results/prediction_explanations.csv
"""

import os
import re
import json
import glob
import time
import joblib
import requests
import pandas as pd
from jsonschema import validate, ValidationError

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 20)

RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)


def find_file(filename):
    candidates = [filename, f"/mnt/user-data/uploads/{filename}", f"/content/{filename}",
                  f"data/{filename}"]
    for path in candidates:
        matches = glob.glob(path, recursive=True)
        if matches:
            return matches[0]
    matches = glob.glob(f"**/{filename}", recursive=True)
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Could not find '{filename}'.")


# =====================================================================
# STEP 0 — LLM API CONNECTION
# =====================================================================
print("=" * 70)
print("STEP 0: LLM API CONNECTION SETUP")
print("=" * 70)

# API key is NEVER hardcoded -- it is read from an environment variable.
API_KEY = os.environ.get("LLM_API_KEY")
API_URL = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL_NAME = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b:free")

if not API_KEY:
    print("WARNING: LLM_API_KEY environment variable is not set. "
          "Set it before running this script, e.g.:\n"
          "    export LLM_API_KEY='your-key-here'\n"
          "The script will still run, but every call_llm() call will fail fast "
          "and return None until a real key is provided.")


def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=1200, max_retries=3):
    """
    Reusable function to call any OpenAI-compatible chat-completions endpoint
    (OpenRouter, OpenAI, etc.) via a plain HTTP POST with the `requests` library.

    Retries on rate-limit (429) and transient server errors (5xx), since free-tier
    models on shared endpoints are frequently overloaded. Returns the assistant's
    text content on success, or None if every attempt fails.
    """
    if not API_KEY:
        print("call_llm() aborted: LLM_API_KEY is not set.")
        return None

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    # Optional: for reasoning models (e.g. openai/gpt-oss-*, deepseek-r1, etc.) that spend part
    # of max_tokens on a hidden reasoning trace before the final answer, lowering reasoning
    # effort leaves more of the token budget for the actual JSON output. Only applied if set,
    # since not every provider/model recognizes this field (harmless to include otherwise).
    reasoning_effort = os.environ.get("LLM_REASONING_EFFORT")
    if reasoning_effort:
        payload["reasoning"] = {"effort": reasoning_effort}
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"call_llm() network error (attempt {attempt}/{max_retries}): {e}")
            time.sleep(2 * attempt)
            continue

        if response.status_code == 200:
            try:
                body = response.json()
            except json.JSONDecodeError as e:
                print(f"call_llm() could not parse response body as JSON: {e}\nRaw text: {response.text[:500]}")
                return None

            choices = body.get("choices")
            if not choices:
                # Model returned 200 but no choices -- print the full body so the
                # cause (content filter, empty generation, provider error passthrough) is visible.
                print(f"call_llm() got HTTP 200 but no 'choices' in body: {body}")
                return None

            content = choices[0].get("message", {}).get("content")
            if not content:
                print(f"call_llm() got an empty message content. Full choice: {choices[0]}")
                return None
            return content

        elif response.status_code in (429, 500, 502, 503, 504):
            print(f"call_llm() got status {response.status_code} (attempt {attempt}/{max_retries}): "
                  f"{response.text[:300]} -- retrying...")
            time.sleep(3 * attempt)
            continue
        else:
            print(f"call_llm() failed with status code {response.status_code}: {response.text[:500]}")
            return None

    print(f"call_llm() gave up after {max_retries} attempts.")
    return None


def call_llm_english(system_prompt, user_prompt, temperature=0.0, max_tokens=1200):
    """
    Wraps call_llm() with an English-output check. Some free-tier / higher-temperature
    generations can drift into another language mid-response (observed with Cyrillic
    tokens at temperature=0.7). Since downstream consumers are English-speaking plant
    operators and the output must be strict JSON, we detect non-ASCII content and
    retry once with a stronger reminder before giving up and returning what we got.
    """
    response = call_llm(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)

    if response is not None and not response.isascii():
        print("call_llm_english() detected non-ASCII (likely non-English) output -- retrying once.")
        reinforced_system_prompt = (
            system_prompt
            + " IMPORTANT: Respond ONLY in English. Do not use words, phrases, or "
              "characters from any other language, anywhere in the response."
        )
        retry_response = call_llm(
            reinforced_system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens
        )
        if retry_response is not None and retry_response.isascii():
            return retry_response
        # Fall back to whichever response we have; caller/validator will flag it if
        # it still isn't valid JSON, but we don't silently lose the original result.
        print("call_llm_english() retry still non-ASCII or failed -- returning best available response.")
        return retry_response if retry_response is not None else response

    return response


# --- Demonstrate call_llm with a simple test prompt ---
print("\nTest call: 'Reply with only the word: hello'")
test_output = call_llm(
    system_prompt="You are a terse assistant that follows instructions exactly.",
    user_prompt="Reply with only the word: hello",
    temperature=0.0,
    max_tokens=200,
)
print(f"Test call output: {test_output!r}")

# =====================================================================
# STEP 1 — PII GUARDRAIL
# =====================================================================
print("\n" + "=" * 70)
print("STEP 1: PII GUARDRAIL")
print("=" * 70)


def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))


def call_llm_guarded(system_prompt, user_prompt, temperature=0.0, max_tokens=1200):
    """Wraps call_llm() with the mandatory PII guardrail, run BEFORE every call."""
    if has_pii(user_prompt):
        print("Input blocked: PII detected.")
        return None
    return call_llm(system_prompt, user_prompt, temperature=temperature, max_tokens=max_tokens)


guardrail_tests = [
    ("Contains email", "Please summarize this record for john.doe@example.com and flag any issues."),
    ("Clean input", "Please summarize this record and flag any issues."),
]

guardrail_rows = []
for label, text in guardrail_tests:
    blocked = has_pii(text)
    print(f"\n[{label}] input: {text!r}")
    print(f"has_pii() -> {blocked}")
    if blocked:
        print("Input blocked: PII detected.")
        outcome = "Blocked (PII detected)"
    else:
        result = call_llm_guarded(
            system_prompt="You are a helpful assistant.",
            user_prompt=text,
            temperature=0.0,
            max_tokens=300,
        )
        outcome = f"Passed to LLM (response: {result!r})"
    guardrail_rows.append((label, text, blocked, outcome))

guardrail_table = pd.DataFrame(guardrail_rows, columns=["Test", "Input", "PII Detected", "Outcome"])
print("\nGuardrail test summary:")
print(guardrail_table.to_string(index=False))
guardrail_table.to_csv(f"{RESULTS_DIR}/pii_guardrail_test.csv", index=False)
print(f"Saved: {RESULTS_DIR}/pii_guardrail_test.csv")

# =====================================================================
# STEP 2 — LOAD BEST MODEL FROM PART 3
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: LOAD best_model.pkl")
print("=" * 70)

model_path = find_file("best_model.pkl")
print(f"Loading: {model_path}")
best_model = joblib.load(model_path)
print(f"Loaded model type: {type(best_model)}")

# The exact column order the pipeline was fit on (Part 3: SimpleImputer ->
# StandardScaler -> RandomForestClassifier, fit on un-scaled X_train which
# was: numeric weather/time features + label-encoded DAY_PERIOD +
# one-hot SOURCE_KEY_GEN with drop_first=True, so inverter
# "1BY6WEcLGh8j5v7" -- the first category alphabetically -- has no
# column of its own and is represented by all dummy columns = 0).
FEATURE_ORDER = list(best_model.feature_names_in_)
DAY_PERIOD_ORDER = {"Night": 0, "Morning": 1, "Afternoon": 2, "Evening": 3}
SOURCE_KEY_DUMMY_COLS = [c for c in FEATURE_ORDER if c.startswith("SOURCE_KEY_GEN_")]
# The inverter ID encoded in each dummy column name, e.g. "SOURCE_KEY_GEN_1IF53ai7Xc0U56Y" -> "1IF53ai7Xc0U56Y"
KNOWN_INVERTERS = ["1BY6WEcLGh8j5v7"] + [c.replace("SOURCE_KEY_GEN_", "") for c in SOURCE_KEY_DUMMY_COLS]


def encode_record(features: dict) -> pd.DataFrame:
    """
    Preprocess a raw feature dict into the single-row DataFrame the
    pipeline expects, with columns in FEATURE_ORDER.

    Expected keys in `features`:
        AMBIENT_TEMPERATURE (float)
        MODULE_TEMPERATURE  (float)
        IRRADIATION          (float)
        HOUR                  (int, 0-23)
        DAY_PERIOD           (str: "Night"|"Morning"|"Afternoon"|"Evening")
        SOURCE_KEY_GEN       (str: one of the 22 known inverter IDs)
    """
    inverter = features["SOURCE_KEY_GEN"]
    if inverter not in KNOWN_INVERTERS:
        raise ValueError(
            f"Unknown SOURCE_KEY_GEN '{inverter}'. Must be one of: {KNOWN_INVERTERS}"
        )

    row = {
        "AMBIENT_TEMPERATURE": float(features["AMBIENT_TEMPERATURE"]),
        "MODULE_TEMPERATURE": float(features["MODULE_TEMPERATURE"]),
        "IRRADIATION": float(features["IRRADIATION"]),
        "HOUR": int(features["HOUR"]),
        "DAY_PERIOD_ENC": DAY_PERIOD_ORDER[features["DAY_PERIOD"]],
    }
    for dummy_col in SOURCE_KEY_DUMMY_COLS:
        row[dummy_col] = 1.0 if dummy_col == f"SOURCE_KEY_GEN_{inverter}" else 0.0

    return pd.DataFrame([row], columns=FEATURE_ORDER)


# =====================================================================
# STEP 3 — THREE HAND-CRAFTED FEATURE-VECTOR INPUTS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 3: HAND-CRAFTED FEATURE-VECTOR INPUTS -> PREDICT / PREDICT_PROBA")
print("=" * 70)

hand_crafted_inputs = [
    {  # Clear daytime, high irradiation, healthy-looking reading
        "AMBIENT_TEMPERATURE": 32.0,
        "MODULE_TEMPERATURE": 48.5,
        "IRRADIATION": 0.85,
        "HOUR": 12,
        "DAY_PERIOD": "Afternoon",
        "SOURCE_KEY_GEN": "1IF53ai7Xc0U56Y",
    },
    {  # Low irradiation, early morning -- plausible underperformance candidate
        "AMBIENT_TEMPERATURE": 24.0,
        "MODULE_TEMPERATURE": 25.0,
        "IRRADIATION": 0.05,
        "HOUR": 7,
        "DAY_PERIOD": "Morning",
        "SOURCE_KEY_GEN": "adLQvlD726eNBSB",
    },
    {  # Moderate irradiation, evening, mid-range conditions
        "AMBIENT_TEMPERATURE": 28.0,
        "MODULE_TEMPERATURE": 33.0,
        "IRRADIATION": 0.35,
        "HOUR": 17,
        "DAY_PERIOD": "Evening",
        "SOURCE_KEY_GEN": "z9Y9gH1T5YWrNuG",
    },
]

encoded_rows = []
predictions = []
probabilities = []
for i, feats in enumerate(hand_crafted_inputs, start=1):
    encoded = encode_record(feats)
    pred = best_model.predict(encoded)[0]
    proba = best_model.predict_proba(encoded)[0, 1]
    encoded_rows.append(encoded)
    predictions.append(int(pred))
    probabilities.append(float(proba))
    print(f"\nInput {i}: {feats}")
    print(f"  Predicted class (1=underperforming): {pred}")
    print(f"  Predicted probability of underperformance: {proba:.4f}")

# =====================================================================
# STEP 4 — SCHEMA + PROMPT DESIGN FOR EXPLANATIONS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 4: PROMPT DESIGN AND JSON SCHEMA")
print("=" * 70)

EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": [
        "prediction_label",
        "confidence_level",
        "top_reason",
        "second_reason",
        "next_step",
    ],
}

SYSTEM_PROMPT = (
    "You are a solar-plant monitoring assistant. You are given the feature "
    "values used by a machine-learning classifier, the classifier's predicted "
    "class, and its predicted probability for a single inverter reading. The "
    "model flags a reading as class 1 ('underperforming') when AC power output "
    "falls in the bottom 20% of all genuine daytime readings (IRRADIATION > 0). "
    "Explain the prediction in plain language for a plant operator. "
    "If the primary driver of a low-output prediction is naturally low "
    "irradiation (e.g. very early morning, late evening, or overcast "
    "conditions), say the low output may be expected given current conditions, "
    "and recommend inspecting the panels/inverter only if similarly low output "
    "continues once irradiation rises -- do not jump straight to recommending a "
    "physical inspection when the conditions themselves plausibly explain it. "
    "Respond ONLY in English. Do not use words, phrases, or characters from any "
    "other language, anywhere in the response. "
    "Respond with ONLY a single valid JSON object -- no markdown code fences, "
    "no commentary before or after it -- containing exactly these five scalar "
    "fields: "
    '{"prediction_label": string, "confidence_level": "low"|"medium"|"high", '
    '"top_reason": string, "second_reason": string, "next_step": string}. '
    "Do not include any other keys."
)

USER_PROMPT_TEMPLATE = (
    "Feature values:\n"
    "  Ambient temperature (C): {AMBIENT_TEMPERATURE}\n"
    "  Module temperature (C): {MODULE_TEMPERATURE}\n"
    "  Irradiation (normalized): {IRRADIATION}\n"
    "  Hour of day: {HOUR}\n"
    "  Day period: {DAY_PERIOD}\n"
    "  Inverter ID: {SOURCE_KEY_GEN}\n"
    "Predicted class: {predicted_class} (1 = underperforming, 0 = normal)\n"
    "Predicted probability of underperformance: {predicted_proba:.4f}\n\n"
    "Return the JSON explanation now."
)

print("SYSTEM_PROMPT (verbatim):")
print(SYSTEM_PROMPT)
print("\nUSER_PROMPT_TEMPLATE (verbatim, with placeholders):")
print(USER_PROMPT_TEMPLATE)
print(
    "\nRationale for temperature=0: this is a structured-extraction task where "
    "we need the SAME five JSON fields, validly formatted, every time -- "
    "temperature=0 makes the model always pick the highest-probability next "
    "token, which maximizes determinism and JSON-validity consistency. Any "
    "creative variation in wording is a liability here, not a benefit, since "
    "downstream code parses the output as strict JSON."
)


def build_user_prompt(feats, pred, proba):
    return USER_PROMPT_TEMPLATE.format(
        AMBIENT_TEMPERATURE=feats["AMBIENT_TEMPERATURE"],
        MODULE_TEMPERATURE=feats["MODULE_TEMPERATURE"],
        IRRADIATION=feats["IRRADIATION"],
        HOUR=feats["HOUR"],
        DAY_PERIOD=feats["DAY_PERIOD"],
        SOURCE_KEY_GEN=feats["SOURCE_KEY_GEN"],
        predicted_class=pred,
        predicted_proba=proba,
    )


def parse_and_validate(raw_response):
    """Strip whitespace, parse as JSON, validate against EXPLANATION_SCHEMA.
    Returns (parsed_dict_or_fallback, status_string)."""
    fallback = {
        "prediction_label": None,
        "confidence_level": None,
        "top_reason": None,
        "second_reason": None,
        "next_step": None,
    }
    if raw_response is None:
        return fallback, "fail (no response from call_llm)"

    try:
        parsed = json.loads(raw_response.strip())
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return fallback, f"fail (JSONDecodeError: {e})"

    try:
        validate(instance=parsed, schema=EXPLANATION_SCHEMA)
    except ValidationError as e:
        print(f"Schema validation error: {e.message}")
        return fallback, f"fail (ValidationError: {e.message})"

    return parsed, "pass"


# =====================================================================
# STEP 5 — RUN THE PIPELINE END-TO-END ON THE THREE INPUTS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: END-TO-END PIPELINE (guardrail -> LLM -> validate)")
print("=" * 70)

demo_rows = []
for i, feats in enumerate(hand_crafted_inputs, start=1):
    user_prompt = build_user_prompt(feats, predictions[i - 1], probabilities[i - 1])
    print(f"\n--- Input {i} ---")
    print(f"Feature input: {feats}")
    print(f"Predicted class: {predictions[i - 1]}   Probability: {probabilities[i - 1]:.4f}")

    blocked = has_pii(user_prompt)
    if blocked:
        print("Input blocked: PII detected.")
        raw_response = None
        pass_block = "Blocked"
    else:
        time.sleep(2)  # brief pacing to avoid free-tier rate limits between successive calls
        raw_response = call_llm_english(SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=1200)
        pass_block = "Passed"

    print(f"Raw LLM response: {raw_response!r}")
    parsed, status = parse_and_validate(raw_response)
    print(f"Validation outcome: {status}")
    print(f"Parsed/fallback explanation: {parsed}")

    demo_rows.append({
        "Feature Input": json.dumps(feats),
        "Predicted Class": predictions[i - 1],
        "Probability": round(probabilities[i - 1], 4),
        "Explanation JSON": json.dumps(parsed),
        "Validation Status": status,
        "Guardrail Result": pass_block,
    })

demo_table = pd.DataFrame(demo_rows)
print("\n3-row demonstration table:")
print(demo_table.to_string(index=False))
demo_table.to_csv(f"{RESULTS_DIR}/prediction_explanations.csv", index=False)
print(f"Saved: {RESULTS_DIR}/prediction_explanations.csv")

# =====================================================================
# STEP 6 — TEMPERATURE A/B COMPARISON (temp=0 vs temp=0.7)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 6: TEMPERATURE COMPARISON (temp=0 vs temp=0.7)")
print("=" * 70)

temp_rows = []
for i, feats in enumerate(hand_crafted_inputs, start=1):
    user_prompt = build_user_prompt(feats, predictions[i - 1], probabilities[i - 1])

    if has_pii(user_prompt):
        out_t0, out_t7 = None, None
    else:
        out_t0 = call_llm_english(SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=1200)
        time.sleep(2)
        out_t7 = call_llm_english(SYSTEM_PROMPT, user_prompt, temperature=0.7, max_tokens=1200)
        time.sleep(2)

    print(f"\nInput {i} @ temp=0.0: {out_t0!r}")
    print(f"Input {i} @ temp=0.7: {out_t7!r}")

    key_diff = (
        "Not evaluated (no LLM response -- check LLM_API_KEY / connectivity)"
        if (out_t0 is None or out_t7 is None)
        else ("Identical wording" if out_t0.strip() == out_t7.strip() else "Differing wording/phrasing")
    )
    temp_rows.append({
        "Input": json.dumps(feats),
        "Output at temp=0": out_t0,
        "Output at temp=0.7": out_t7,
        "Key difference": key_diff,
    })

temp_table = pd.DataFrame(temp_rows)
print("\nTemperature comparison table:")
print(temp_table.to_string(index=False))
temp_table.to_csv(f"{RESULTS_DIR}/temperature_comparison.csv", index=False)
print(f"Saved: {RESULTS_DIR}/temperature_comparison.csv")
print(
    "\nNOTE (for README): at temperature=0 the model deterministically selects the "
    "single highest-probability next token at every step, so re-running the exact "
    "same prompt produces the same (or nearly identical) output every time -- ideal "
    "for a structured field we need to parse reliably. At temperature=0.7 the model "
    "samples from a wider slice of the next-token probability distribution, so "
    "wording, emphasis, and sometimes field ordering can vary between runs even "
    "though the underlying prediction and probability fed into the prompt are "
    "unchanged -- useful for more natural, varied prose but riskier for anything "
    "downstream code must parse deterministically."
)

print("\nDONE. All tables saved in ./results/.")