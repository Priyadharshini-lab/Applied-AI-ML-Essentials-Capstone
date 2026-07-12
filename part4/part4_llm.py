"""
=====================================================================
 PART 4 — LLM-Powered Feature
 TRACK CHOSEN: (C) Model Prediction Explanation Pipeline
 Input : best_model.pkl              (Part 3's part3_ensembles.py)
         expected_power_model.pkl    (Part 2's part2_models.py)
=====================================================================
Run with:  python3 part4_llm_explain.py

Requires an environment variable holding your LLM API key. Two ways to
provide it:

  (a) A .env file in the same folder as this script, containing:
          LLM_API_KEY=sk-or-...
      (requires `pip install python-dotenv`; this script loads it
      automatically via load_dotenv() below -- just creating the .env
      file is NOT enough on its own, it has to actually be loaded into
      the process environment, which is what load_dotenv() does)

  (b) Setting the environment variable directly in your shell, e.g.:
          export LLM_API_KEY="sk-or-..."       (macOS/Linux)
          set LLM_API_KEY=sk-or-...            (Windows cmd)
          $env:LLM_API_KEY="sk-or-..."         (Windows PowerShell)

Optionally override the endpoint / model (same .env or shell mechanism):
    LLM_API_URL=https://openrouter.ai/api/v1/chat/completions
    LLM_MODEL=openai/gpt-oss-20b:free

SECURITY: never commit a real .env file or a real API key to git. Add
`.env` to .gitignore and commit only a `.env.example` with a placeholder
value (e.g. LLM_API_KEY=your-key-here). If a real key was ever exposed
(committed, screenshotted, pasted in chat, etc.), treat it as compromised:
revoke/delete it in your provider's dashboard and generate a new one.

Produces:
  - console output for every task
  - results/pii_guardrail_test.csv
  - results/temperature_comparison.csv
  - results/prediction_explanations.csv

=====================================================================
FIX APPLIED IN THIS VERSION (vs the previous run's log)
---------------------------------------------------------------------
Input 3 in the previous run showed the LLM returning
    prediction_label = "normal"
alongside reasoning text that described a clearly bad Performance Ratio
(0.469) and recommended an inspection -- i.e. the LLM's own wording
contradicted the label it put in the very same JSON object, and that
label happened to disagree with the classifier's actual predicted class.

The fix has two layers:
  1. PROMPT FIX: SYSTEM_PROMPT now contains an explicit, unambiguous rule
     that prediction_label MUST mirror the classifier's predicted class
     (not the Performance Ratio, not the LLM's own judgment).
  2. CODE FIX (the one that actually guarantees correctness):
     parse_and_validate() now takes the classifier's `pred` and force-
     overwrites `prediction_label` to match it after parsing, regardless
     of what the LLM wrote. A prompt instruction is a request; the code
     override is a guarantee. If the LLM disagreed, that disagreement is
     logged and preserved in a new "needs_review" column instead of being
     silently lost -- a case where the hard classifier decision and the
     continuous Performance Ratio disagree (as in Input 3) is genuinely
     useful information for an operator, so it's surfaced, not overridden
     away or hidden.
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

try:
    from dotenv import load_dotenv
    # Reads a .env file (if present) in the current working directory and
    # injects its KEY=VALUE lines into os.environ, so os.environ.get(...)
    # below can actually see them. Without this call, a .env file sitting
    # on disk has no effect -- Python never reads it automatically.
    load_dotenv()
except ImportError:
    print(
        "NOTE: python-dotenv is not installed, so a .env file (if you have one) "
        "will NOT be loaded automatically. Run: pip install python-dotenv "
        "-- or export LLM_API_KEY directly in your shell instead."
    )

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

# API key is NEVER hardcoded -- it is read from the environment (populated
# either by a real shell `export`/`set`, or by load_dotenv() above reading
# a local .env file).
API_KEY = os.environ.get("LLM_API_KEY")
API_URL = os.environ.get("LLM_API_URL", "https://openrouter.ai/api/v1/chat/completions")
MODEL_NAME = os.environ.get("LLM_MODEL", "openai/gpt-oss-20b:free")

if not API_KEY:
    print(
        "WARNING: LLM_API_KEY is not set in the environment. Checked both a "
        "local .env file (via load_dotenv()) and the shell environment -- "
        "neither has it. Set it one of these ways before running this script:\n"
        "    (a) create a .env file next to this script containing:\n"
        "            LLM_API_KEY=your-key-here\n"
        "    (b) export LLM_API_KEY='your-key-here'   (shell)\n"
        "The script will still run, but every call_llm() call will fail fast "
        "and return None until a real key is provided."
    )
else:
    # Masked confirmation only -- NEVER print the full key (to console,
    # logs, screenshots, or committed files). This just confirms the key
    # was actually loaded and roughly what it looks like.
    masked = API_KEY[:8] + "..." + API_KEY[-4:] if len(API_KEY) > 12 else "***"
    print(f"LLM_API_KEY loaded successfully (masked): {masked}")


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
# STEP 2 — LOAD BEST CLASSIFIER (Part 3) AND EXPECTED-POWER MODEL (Part 2)
# =====================================================================
print("\n" + "=" * 70)
print("STEP 2: LOAD best_model.pkl AND expected_power_model.pkl")
print("=" * 70)

model_path = find_file("best_model.pkl")
print(f"Loading: {model_path}")
best_model = joblib.load(model_path)
print(f"Loaded classifier type: {type(best_model)}")

# expected_power_model.pkl is the regression pipeline saved at the end of
# Part 2. It was fit on the SAME leak-free feature set as the classifier
# (weather + time + inverter identity, no AC_POWER lag/rolling features),
# so it can be used to estimate what AC power a reading "should" produce
# given its conditions -- the "Expected Power" half of the Performance
# Ratio used in the LLM explanation below.
power_model_path = find_file("expected_power_model.pkl")
print(f"Loading: {power_model_path}")
expected_power_model = joblib.load(power_model_path)
print(f"Loaded regression model type: {type(expected_power_model)}")

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
    pipeline expects, with columns in FEATURE_ORDER. This same encoded
    row is used for BOTH the classifier (best_model) and the regression
    model (expected_power_model), since they share the identical feature
    set and encoding.

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
print("STEP 3: HAND-CRAFTED FEATURE-VECTOR INPUTS -> PREDICT / PREDICT_PROBA / EXPECTED POWER")
print("=" * 70)

# Each input carries a hand-crafted ACTUAL_AC_POWER value (kW) -- a plausible
# measured reading paired with that synthetic scenario. This lets the
# pipeline compute Expected Power (from expected_power_model), Actual Power
# (hand-crafted), and Performance Ratio = Actual / Expected, which are the
# quantities a plant operator actually cares about.
hand_crafted_inputs = [
    {  # Clear daytime, high irradiation, healthy-looking reading -- actual
       # power close to what conditions would predict (ratio near 1.0)
        "AMBIENT_TEMPERATURE": 32.0,
        "MODULE_TEMPERATURE": 48.5,
        "IRRADIATION": 0.85,
        "HOUR": 12,
        "DAY_PERIOD": "Afternoon",
        "SOURCE_KEY_GEN": "1IF53ai7Xc0U56Y",
        "ACTUAL_AC_POWER": 940.0,
    },
    {  # Low irradiation, early morning -- low actual power that is
       # expected given the conditions
        "AMBIENT_TEMPERATURE": 24.0,
        "MODULE_TEMPERATURE": 25.0,
        "IRRADIATION": 0.05,
        "HOUR": 7,
        "DAY_PERIOD": "Morning",
        "SOURCE_KEY_GEN": "adLQvlD726eNBSB",
        "ACTUAL_AC_POWER": 42.0,
    },
    {  # Moderate irradiation, evening -- actual power deliberately well
       # below what conditions would predict; this is the borderline case
       # (classifier said class 0 at proba 0.37, but Performance Ratio is
       # a poor 0.469) that exposed the labeling bug -- kept in on purpose
       # as a regression test for the fix below.
        "AMBIENT_TEMPERATURE": 28.0,
        "MODULE_TEMPERATURE": 33.0,
        "IRRADIATION": 0.35,
        "HOUR": 17,
        "DAY_PERIOD": "Evening",
        "SOURCE_KEY_GEN": "z9Y9gH1T5YWrNuG",
        "ACTUAL_AC_POWER": 210.0,
    },
]

encoded_rows = []
predictions = []
probabilities = []
expected_powers = []
performance_ratios = []
for i, feats in enumerate(hand_crafted_inputs, start=1):
    encoded = encode_record(feats)
    pred = best_model.predict(encoded)[0]
    proba = best_model.predict_proba(encoded)[0, 1]

    expected_power = float(expected_power_model.predict(encoded)[0])
    expected_power = max(expected_power, 0.0)  # power can't be negative; clip a small-magnitude regression undershoot
    actual_power = feats["ACTUAL_AC_POWER"]
    # Guard against division by (near) zero at full-night / zero-irradiation
    # edge cases; a near-zero expected power with non-trivial actual power
    # is itself notable and reported as such rather than raising an error.
    performance_ratio = actual_power / expected_power if expected_power > 1.0 else None

    encoded_rows.append(encoded)
    predictions.append(int(pred))
    probabilities.append(float(proba))
    expected_powers.append(expected_power)
    performance_ratios.append(performance_ratio)

    print(f"\nInput {i}: {feats}")
    print(f"  Predicted class (1=underperforming): {pred}")
    print(f"  Predicted probability of underperformance: {proba:.4f}")
    print(f"  Expected AC Power (regression model): {expected_power:.1f} kW")
    print(f"  Actual AC Power (hand-crafted): {actual_power:.1f} kW")
    ratio_str = f"{performance_ratio:.3f}" if performance_ratio is not None else "N/A (expected power ~0)"
    print(f"  Performance Ratio (actual / expected): {ratio_str}")

# =====================================================================
# TASK: DETERMINE "NEEDS_REVIEW" IN CODE (NOT BY THE LLM)
# =====================================================================
# A borderline/disagreement flag computed directly from the classifier and
# the Performance Ratio, independent of anything the LLM says. This is what
# actually explains cases like Input 3: predicted class 0 (not flagged) but
# a Performance Ratio of 0.469 (clearly poor) -- worth a human's attention
# even though the hard classifier decision didn't cross its threshold.
NEEDS_REVIEW_RATIO_THRESHOLD = 0.75
needs_review_flags = []
for pred, ratio in zip(predictions, performance_ratios):
    disagreement = (pred == 0) and (ratio is not None) and (ratio < NEEDS_REVIEW_RATIO_THRESHOLD)
    needs_review_flags.append(bool(disagreement))

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
    "class, its predicted probability for a single inverter reading, and three "
    "measured/estimated power quantities: Expected AC Power (what a separate "
    "regression model, trained only on weather/time/inverter-identity features, "
    "predicts this reading 'should' produce), Actual AC Power (what was really "
    "measured), and Performance Ratio (Actual / Expected). The classifier flags "
    "a reading as class 1 ('underperforming') when AC power output falls in the "
    "bottom 20% of all genuine daytime readings (IRRADIATION > 0). "
    "Explain the prediction in plain language for a plant operator, and ground "
    "your explanation in the Performance Ratio rather than only the raw "
    "environmental conditions: a Performance Ratio near 1.0 means the reading is "
    "close to what conditions predict, even if absolute output is low (e.g. early "
    "morning or overcast) -- in that case say the low output is expected given "
    "current conditions. A Performance Ratio meaningfully below 1.0 (roughly "
    "under 0.85) means the inverter produced notably less than conditions "
    "predict, which is a genuine deviation worth mentioning in your reasoning "
    "even if it doesn't change the label below. If Performance Ratio is not "
    "available (expected power is near zero, e.g. deep night), rely on the raw "
    "conditions instead and say so. "
    # --- Hard constraint added to fix the label/class contradiction bug ---
    "The field prediction_label MUST always match the classifier's Predicted "
    "class exactly and is NOT a judgment call for you to make: if Predicted "
    "class = 1, prediction_label MUST be the literal string 'underperforming'; "
    "if Predicted class = 0, prediction_label MUST be the literal string "
    "'normal'. Never let the Performance Ratio, or your own reasoning, lead you "
    "to write a different label than the classifier's Predicted class -- use "
    "the Performance Ratio only to inform top_reason/second_reason/next_step, "
    "never to override prediction_label. If the Performance Ratio and the "
    "Predicted class seem to disagree (e.g. class is 'normal' but the ratio is "
    "poor), keep prediction_label as the classifier's class, note the "
    "discrepancy in top_reason or second_reason, and set next_step to "
    "recommend a closer look even though the hard classification is 'normal'. "
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
    "Predicted probability of underperformance: {predicted_proba:.4f}\n"
    "Expected AC Power (kW, from regression model): {expected_power:.1f}\n"
    "Actual AC Power (kW, measured): {actual_power:.1f}\n"
    "Performance Ratio (Actual / Expected): {performance_ratio_str}\n\n"
    "Reminder: prediction_label MUST exactly match Predicted class above "
    "('underperforming' if Predicted class is 1, otherwise 'normal').\n\n"
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


def build_user_prompt(feats, pred, proba, expected_power, actual_power, performance_ratio):
    ratio_str = f"{performance_ratio:.3f}" if performance_ratio is not None else "N/A (expected power near zero)"
    required_label = "underperforming" if pred == 1 else "normal"
    return (
        "Feature values:\n"
        f"  Ambient temperature (C): {feats['AMBIENT_TEMPERATURE']}\n"
        f"  Module temperature (C): {feats['MODULE_TEMPERATURE']}\n"
        f"  Irradiation (normalized): {feats['IRRADIATION']}\n"
        f"  Hour of day: {feats['HOUR']}\n"
        f"  Day period: {feats['DAY_PERIOD']}\n"
        f"  Inverter ID: {feats['SOURCE_KEY_GEN']}\n"
        f"Predicted class: {pred} (1 = underperforming, 0 = normal)\n"
        f"Predicted probability of underperformance: {proba:.4f}\n"
        f"Expected AC Power (kW, from regression model): {expected_power:.1f}\n"
        f"Actual AC Power (kW, measured): {actual_power:.1f}\n"
        f"Performance Ratio (Actual / Expected): {ratio_str}\n\n"
        f"Reminder: prediction_label MUST be '{required_label}' for this record, "
        f"matching Predicted class above exactly.\n\n"
        "Return the JSON explanation now."
    )


def parse_and_validate(raw_response, pred):
    """
    Strip whitespace, parse as JSON, validate against EXPLANATION_SCHEMA, then
    force prediction_label to match the classifier's actual predicted class.

    THE FIX: prediction_label is descriptive text coming out of the LLM, not a
    second vote on the classification. The prompt asks the LLM to keep it
    consistent, but only this hard override in code actually guarantees it on
    every run regardless of model behavior/temperature/drift.

    Returns (parsed_dict_or_fallback, status_string, label_overridden_bool).
    """
    expected_label = "underperforming" if pred == 1 else "normal"
    fallback = {
        "prediction_label": expected_label,
        "confidence_level": None,
        "top_reason": None,
        "second_reason": None,
        "next_step": None,
    }
    if raw_response is None:
        return fallback, "fail (no response from call_llm)", False

    try:
        parsed = json.loads(raw_response.strip())
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return fallback, f"fail (JSONDecodeError: {e})", False

    try:
        validate(instance=parsed, schema=EXPLANATION_SCHEMA)
    except ValidationError as e:
        print(f"Schema validation error: {e.message}")
        return fallback, f"fail (ValidationError: {e.message})", False

    # --- THE ACTUAL FIX: never trust the LLM's own prediction_label ---
    llm_label = str(parsed.get("prediction_label", "")).strip().lower()
    overridden = llm_label != expected_label
    if overridden:
        print(
            f"NOTE: LLM prediction_label ({parsed.get('prediction_label')!r}) "
            f"disagreed with classifier's predicted class ({pred}) -- "
            f"overriding to '{expected_label}'."
        )
        parsed["prediction_label"] = expected_label

    return parsed, "pass", overridden


# =====================================================================
# STEP 5 — RUN THE PIPELINE END-TO-END ON THE THREE INPUTS
# =====================================================================
print("\n" + "=" * 70)
print("STEP 5: END-TO-END PIPELINE (guardrail -> LLM -> validate -> enforce label)")
print("=" * 70)

demo_rows = []
for i, feats in enumerate(hand_crafted_inputs, start=1):
    pred = predictions[i - 1]
    user_prompt = build_user_prompt(
        feats, pred, probabilities[i - 1],
        expected_powers[i - 1], feats["ACTUAL_AC_POWER"], performance_ratios[i - 1],
    )
    print(f"\n--- Input {i} ---")
    print(f"Feature input: {feats}")
    print(f"Predicted class: {pred}   Probability: {probabilities[i - 1]:.4f}")
    print(f"Expected power: {expected_powers[i - 1]:.1f} kW   "
          f"Actual power: {feats['ACTUAL_AC_POWER']:.1f} kW   "
          f"Performance ratio: {performance_ratios[i - 1]}")
    print(f"needs_review (code-computed, class vs ratio disagreement): {needs_review_flags[i - 1]}")

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
    parsed, status, label_overridden = parse_and_validate(raw_response, pred)
    print(f"Validation outcome: {status}")
    print(f"Label overridden to match classifier: {label_overridden}")
    print(f"Parsed/fallback explanation: {parsed}")

    demo_rows.append({
        "Feature Input": json.dumps(feats),
        "Predicted Class": pred,
        "Probability": round(probabilities[i - 1], 4),
        "Expected Power (kW)": round(expected_powers[i - 1], 1),
        "Actual Power (kW)": feats["ACTUAL_AC_POWER"],
        "Performance Ratio": (
            round(performance_ratios[i - 1], 3) if performance_ratios[i - 1] is not None else None
        ),
        "Needs Review (code)": needs_review_flags[i - 1],
        "Explanation JSON": json.dumps(parsed),
        "Label Overridden": label_overridden,
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
    pred = predictions[i - 1]
    user_prompt = build_user_prompt(
        feats, pred, probabilities[i - 1],
        expected_powers[i - 1], feats["ACTUAL_AC_POWER"], performance_ratios[i - 1],
    )

    if has_pii(user_prompt):
        out_t0, out_t7 = None, None
    else:
        out_t0 = call_llm_english(SYSTEM_PROMPT, user_prompt, temperature=0.0, max_tokens=1200)
        time.sleep(2)
        out_t7 = call_llm_english(SYSTEM_PROMPT, user_prompt, temperature=0.7, max_tokens=1200)
        time.sleep(2)

    print(f"\nInput {i} @ temp=0.0: {out_t0!r}")
    print(f"Input {i} @ temp=0.7: {out_t7!r}")

    # Even in this raw A/B comparison (pre-override), check whether either
    # temperature setting produced a prediction_label inconsistent with the
    # classifier -- useful evidence for the README about why the code-level
    # override in Step 5 is necessary rather than optional.
    expected_label = "underperforming" if pred == 1 else "normal"

    def _label_consistent(raw):
        if raw is None:
            return None
        try:
            lbl = str(json.loads(raw.strip()).get("prediction_label", "")).strip().lower()
            return lbl == expected_label
        except json.JSONDecodeError:
            return None

    consistent_t0 = _label_consistent(out_t0)
    consistent_t7 = _label_consistent(out_t7)

    key_diff = (
        "Not evaluated (no LLM response -- check LLM_API_KEY / connectivity)"
        if (out_t0 is None or out_t7 is None)
        else ("Identical wording" if out_t0.strip() == out_t7.strip() else "Differing wording/phrasing")
    )
    temp_rows.append({
        "Input": json.dumps(feats),
        "Predicted Class": pred,
        "Output at temp=0": out_t0,
        "Output at temp=0.7": out_t7,
        "Key difference": key_diff,
        "Label consistent @temp=0 (pre-override)": consistent_t0,
        "Label consistent @temp=0.7 (pre-override)": consistent_t7,
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
    "unchanged. The 'Label consistent (pre-override)' columns above show whether "
    "the raw model output already agreed with the classifier before Step 5's code "
    "override is applied -- any 'False' there is exactly the failure mode that "
    "prompted the fix in this version of the script, which is why the pipeline "
    "no longer trusts the LLM's prediction_label at face value."
)

print("\nDONE. All tables saved in ./results/.")






































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































































