# Part 4 — LLM-Powered Feature

**Track chosen: (C) Model Prediction Explanation Pipeline**

## 1. Overview
This part loads the best classifier from Part 3 (`best_model.pkl`) and the expected-power regression model from Part 2 (`expected_power_model.pkl`). For three hand-crafted feature-vector inputs it calls `.predict()` / `.predict_proba()`, computes a Performance Ratio (Actual AC Power ÷ Expected AC Power), and asks an LLM (via OpenRouter, over plain HTTP POST with `requests`) to produce a structured JSON explanation of each prediction for a plant operator.

## 2. `call_llm` Function
```python
def call_llm(system_prompt, user_prompt, temperature=0.0, max_tokens=1200, max_retries=3):
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
    if response.status_code == 200:
        return response.json()["choices"][0]["message"]["content"]
    print(f"call_llm() failed with status code {response.status_code}: {response.text[:500]}")
    return None
```
(The actual implementation additionally retries on `429`/`5xx` transient errors up to `max_retries` times, since the free-tier model used — `openai/gpt-oss-20b:free` — is shared and occasionally rate-limited upstream; see Section 8.)

**API key handling**: `API_KEY = os.environ.get("LLM_API_KEY")` — read from the environment only, via either a real shell `export`/`set` or a local `.env` file loaded with `python-dotenv`'s `load_dotenv()`. It is never hardcoded anywhere in the script, and only a masked preview (`sk-or-v1...537f`) is ever printed.

**Test call demonstration**:
```
Test call: 'Reply with only the word: hello'
Test call output: 'hello'
```

## 3. Prompt Design

**System prompt (verbatim)**:
```
You are a solar-plant monitoring assistant. You are given the feature values used by a machine-learning classifier, the classifier's predicted class, its predicted probability for a single inverter reading, and three measured/estimated power quantities: Expected AC Power (what a separate regression model, trained only on weather/time/inverter-identity features, predicts this reading 'should' produce), Actual AC Power (what was really measured), and Performance Ratio (Actual / Expected). The classifier flags a reading as class 1 ('underperforming') when AC power output falls in the bottom 20% of all genuine daytime readings (IRRADIATION > 0). Explain the prediction in plain language for a plant operator, and ground your explanation in the Performance Ratio rather than only the raw environmental conditions: a Performance Ratio near 1.0 means the reading is close to what conditions predict, even if absolute output is low (e.g. early morning or overcast) -- in that case say the low output is expected given current conditions. A Performance Ratio meaningfully below 1.0 (roughly under 0.85) means the inverter produced notably less than conditions predict, which is a genuine deviation worth mentioning in your reasoning even if it doesn't change the label below. If Performance Ratio is not available (expected power is near zero, e.g. deep night), rely on the raw conditions instead and say so. The field prediction_label MUST always match the classifier's Predicted class exactly and is NOT a judgment call for you to make: if Predicted class = 1, prediction_label MUST be the literal string 'underperforming'; if Predicted class = 0, prediction_label MUST be the literal string 'normal'. Never let the Performance Ratio, or your own reasoning, lead you to write a different label than the classifier's Predicted class -- use the Performance Ratio only to inform top_reason/second_reason/next_step, never to override prediction_label. If the Performance Ratio and the Predicted class seem to disagree (e.g. class is 'normal' but the ratio is poor), keep prediction_label as the classifier's class, note the discrepancy in top_reason or second_reason, and set next_step to recommend a closer look even though the hard classification is 'normal'. Respond ONLY in English. Do not use words, phrases, or characters from any other language, anywhere in the response. Respond with ONLY a single valid JSON object -- no markdown code fences, no commentary before or after it -- containing exactly these five scalar fields: {"prediction_label": string, "confidence_level": "low"|"medium"|"high", "top_reason": string, "second_reason": string, "next_step": string}. Do not include any other keys.
```

**User prompt template (verbatim, with placeholders)**:
```
Feature values:
  Ambient temperature (C): {AMBIENT_TEMPERATURE}
  Module temperature (C): {MODULE_TEMPERATURE}
  Irradiation (normalized): {IRRADIATION}
  Hour of day: {HOUR}
  Day period: {DAY_PERIOD}
  Inverter ID: {SOURCE_KEY_GEN}
Predicted class: {predicted_class} (1 = underperforming, 0 = normal)
Predicted probability of underperformance: {predicted_proba:.4f}
Expected AC Power (kW, from regression model): {expected_power:.1f}
Actual AC Power (kW, measured): {actual_power:.1f}
Performance Ratio (Actual / Expected): {performance_ratio_str}

Reminder: prediction_label MUST be '{required_label}' for this record, matching Predicted class above exactly.

Return the JSON explanation now.
```

**Why `temperature=0`**: this is a structured-extraction task — we need the same five JSON fields, validly formatted, every time. `temperature=0` makes the model always pick the highest-probability next token, maximizing determinism and JSON-validity consistency. Any creative variation in wording is a liability here, not a benefit, since downstream code parses the output as strict JSON.

## 4. JSON Output Schema
```python
EXPLANATION_SCHEMA = {
    "type": "object",
    "properties": {
        "prediction_label": {"type": "string"},
        "confidence_level": {"type": "string", "enum": ["low", "medium", "high"]},
        "top_reason": {"type": "string"},
        "second_reason": {"type": "string"},
        "next_step": {"type": "string"},
    },
    "required": ["prediction_label", "confidence_level", "top_reason", "second_reason", "next_step"],
}
```
5 required scalar fields, as required by the assignment.

**Validation flow**: after each `call_llm()` response, the string is stripped and parsed with `json.loads()` inside a `try/except json.JSONDecodeError`; the parsed object is then validated against `EXPLANATION_SCHEMA` with `jsonschema.validate()` inside a `try/except jsonschema.ValidationError`. On either failure, a fallback dict (all fields `None` except `prediction_label`, which is still set deterministically from the classifier — see Section 5) is returned and the error is logged.

## 5. Design decision: `prediction_label` is enforced in code, not trusted from the LLM

An earlier run surfaced a case where the LLM's own JSON contradicted itself: it wrote `prediction_label: "normal"` while its `top_reason`/`next_step` text described a clearly poor Performance Ratio and recommended an inspection — and that label also disagreed with the classifier's actual predicted class for that row. A prompt instruction ("always match the classifier's class") reduces how often this happens but doesn't guarantee it, since the LLM is still free-generating text.

The fix applied here: `parse_and_validate(raw_response, pred)` **force-overwrites** `prediction_label` to `"underperforming"`/`"normal"` based on the classifier's actual `pred`, regardless of what string the LLM produced. Any disagreement is logged and recorded in a `Label Overridden` column rather than silently dropped. A separate, purely code-computed `needs_review` flag (`pred == 0` and `Performance Ratio < 0.75`) also surfaces exactly the borderline cases (hard classification says "normal", but the continuous ratio says otherwise) that are operationally useful to a plant operator, without letting that disagreement corrupt the reported label.

## 6. PII Guardrail
```python
def has_pii(text):
    email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    phone_pattern = r'\b\d{10}\b|\b\d{3}[-.\s]\d{3}[-.\s]\d{4}\b'
    return bool(re.search(email_pattern, text) or re.search(phone_pattern, text))
```
Run before **every** `call_llm()` call. If `has_pii()` returns `True`, the LLM is never called — the script prints `"Input blocked: PII detected."` and returns `None`.

**Guardrail test results**:

| Test | Input | PII Detected | Outcome |
|---|---|---|---|
| Contains email | "Please summarize this record for john.doe@example.com and flag any issues." | `True` | Blocked (PII detected) |
| Clean input | "Please summarize this record and flag any issues." | `False` | Passed to LLM (response returned) |

## 7. Three-Row Demonstration Table (Track C)

*The rows below are from an example run; regenerate `results/prediction_explanations.csv` with your own API key for the final submission, since exact LLM wording is not deterministic across accounts/sessions even at temperature=0.*

| Feature Input (abridged) | Predicted Class | Probability | Expected Power (kW) | Actual Power (kW) | Performance Ratio | Needs Review | Validation Status |
|---|---|---|---|---|---|---|---|
| Afternoon, IRRADIATION=0.85, inverter 1IF53...56Y | 0 | 0.0000 | 1091.0 | 940.0 | 0.862 | False | pass |
| Morning, IRRADIATION=0.05, inverter adLQv...BSB | 1 | 0.7907 | 104.3 | 42.0 | 0.403 | False | pass |
| Evening, IRRADIATION=0.35, inverter z9Y9g...NuG | 0 | 0.3700 | 447.9 | 210.0 | 0.469 | **True** | pass |

Row 3 is the case described in Section 5: the classifier's hard decision is "normal" (probability 0.37, under whatever threshold separates the classes) while the Performance Ratio (0.469) is poor — `needs_review=True` surfaces this without changing the reported `prediction_label`, which correctly stays `"normal"` to match the classifier.

Full explanation JSON for each row is saved in `results/prediction_explanations.csv`, including `top_reason`, `second_reason`, `next_step`, `confidence_level`, and whether the LLM's own label required overriding (`Label Overridden` column).

## 8. Temperature A/B Comparison (temp=0 vs temp=0.7)

| Input | Output at temp=0 | Output at temp=0.7 | Key difference | Label consistent @0 (pre-override) | Label consistent @0.7 (pre-override) |
|---|---|---|---|---|---|
| Input 1 (normal, ratio 0.862) | `{"prediction_label":"normal", ...}` | `{"prediction_label":"normal", ...}` (differently worded) | Differing wording/phrasing | True | True |
| Input 2 (underperforming, ratio 0.403) | `{"prediction_label":"underperforming", ...}` | *(may fail with HTTP 429 on a free-tier key under load — retried automatically, see Section 9)* | Not evaluated / differing wording | True | *varies* |
| Input 3 (normal, ratio 0.469 — borderline) | `{"prediction_label":"normal", ...}` | *(may fail with HTTP 429 similarly)* | Not evaluated / differing wording | True (after fix) | *varies* |

**Why temperature=0 is more deterministic**: at `temperature=0` the model always selects the single highest-probability next token at each step, so re-running the identical prompt produces the same (or nearly identical) output every time. At `temperature=0.7` the model samples from a wider slice of the next-token probability distribution, so wording, emphasis, and occasionally field ordering vary between runs even though the underlying classifier prediction and probability fed into the prompt are unchanged. The "Label consistent (pre-override)" columns record whether the *raw* LLM output (before the Section 5 code override is applied) already agreed with the classifier — any `False` there is exactly the failure mode the code-level fix in Section 5 was written to eliminate.

## 9. Rate Limits / Transient Errors
The free-tier model used here (`openai/gpt-oss-20b:free`) can return HTTP `429` under shared-endpoint load or once a daily free-request quota is exhausted. `call_llm()` retries automatically on `429` and `5xx` with backoff (`max_retries=3`); if all retries fail, it returns `None`, the guardrail/validation pipeline reports `"fail (no response from call_llm)"`, and the fallback dict (with a still-correct `prediction_label` derived from `pred`) is used — the pipeline degrades gracefully rather than crashing. This is not a code defect; it's expected behavior for a rate-limited free endpoint and is resolved by waiting for the quota reset or supplying a key with a paid tier / higher limits.

## 10. Environment Variables (no hardcoded keys)
```
LLM_API_KEY=sk-or-...              # required
LLM_API_URL=https://openrouter.ai/api/v1/chat/completions   # optional override
LLM_MODEL=openai/gpt-oss-20b:free  # optional override
LLM_REASONING_EFFORT=<...>         # optional, for reasoning-model providers
```
Provided either via a local `.env` file (loaded with `python-dotenv`) or directly exported in the shell. `.env` is git-ignored; only `.env.example` (placeholder values) is committed.

## 11. How to Run
```bash
pip install -r requirements.txt   # requests, pandas, jsonschema, joblib, python-dotenv
export LLM_API_KEY="sk-or-..."
python part4/part4_llm_explain.py
```
Expects `best_model.pkl` (Part 3) and `expected_power_model.pkl` (Part 2) discoverable via the same `find_file()` search used in earlier parts.

## Repository structure
```
part4/
    README.md
    part4_llm_explain.py
    .env.example
    results/
        pii_guardrail_test.csv
        prediction_explanations.csv
        temperature_comparison.csv
```