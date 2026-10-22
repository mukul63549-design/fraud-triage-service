
import json

from mock_llm import generate


ALLOWED_PRIORITIES = {"low", "medium", "high"}
ALLOWED_TEAMS = {
    "card_risk",
    "upi_risk",
    "customer_care",
    "merchant_ops",
}
ALLOWED_ACTIONS = {
    "hold_for_review",
    "block_and_notify",
}
REQUIRED_FIELDS = {
    "summary",
    "priority",
    "recommended_team",
    "recommended_action",
    "policy_id",
}

TEAM_BY_POLICY = {
    "P1": "card_risk",
    "P2": "upi_risk",
    "P3": "customer_care",
    "P4": "merchant_ops",
    "P5": "merchant_ops",
    "P6": "customer_care",
}


def build_prompt(txn_id, fraud_probability, amount_inr, policy_id):
    """Build a prompt with all four required labelled lines."""
    return (
        "Create a concise fraud triage case note as a JSON object only.\n"
        "Required fields: summary, priority, recommended_team, "
        "recommended_action, policy_id.\n"
        "Priority must be low, medium, or high.\n"
        "Team must be card_risk, upi_risk, customer_care, or merchant_ops.\n"
        "Action must be hold_for_review or block_and_notify.\n"
        "Never recommend approval when the policy requires a hold or block.\n"
        f"TXN_ID: {txn_id}\n"
        f"FRAUD_PROBABILITY: {float(fraud_probability)}\n"
        f"AMOUNT_INR: {float(amount_inr)}\n"
        f"POLICY_ID: {policy_id}\n"
    )


def validate_case_note(text, expected_policy_id):
    """Parse and validate a case note, raising ValueError if invalid."""
    try:
        note = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ValueError("Response is not valid JSON.") from exc

    if not isinstance(note, dict):
        raise ValueError("Response must be a JSON object.")

    missing = REQUIRED_FIELDS - note.keys()
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    for field in REQUIRED_FIELDS:
        if not isinstance(note[field], str) or not note[field].strip():
            raise ValueError(f"{field} must be a non-empty string.")

    if note["priority"] not in ALLOWED_PRIORITIES:
        raise ValueError("Invalid priority.")

    if note["recommended_team"] not in ALLOWED_TEAMS:
        raise ValueError("Invalid recommended_team.")

    if note["recommended_action"] not in ALLOWED_ACTIONS:
        raise ValueError("Invalid recommended_action.")

    if note["policy_id"] != expected_policy_id:
        raise ValueError(
            "The response policy_id does not match the retrieved policy."
        )

    # Enforce policy-specific safety requirements.
    if expected_policy_id == "P4":
        if note["recommended_action"] != "block_and_notify":
            raise ValueError("Policy P4 requires block_and_notify.")

    return note


def safe_fallback(txn_id, fraud_probability, amount_inr, policy_id):
    """Return a conservative structured note after repeated failures."""
    action = "block_and_notify" if policy_id == "P4" else "hold_for_review"
    team = TEAM_BY_POLICY.get(policy_id, "customer_care")
    probability = float(fraud_probability)
    priority = "high" if probability >= 0.8 else "medium"

    return {
        "summary": (
            f"Transaction {txn_id} for INR {float(amount_inr):,.0f} "
            f"requires manual review under policy {policy_id}; "
            "automated case-note generation was unavailable or invalid."
        ),
        "priority": priority,
        "recommended_team": team,
        "recommended_action": action,
        "policy_id": policy_id,
    }


def generate_case_note(
    txn_id,
    fraud_probability,
    amount_inr,
    policy_id,
    max_retries=3,
):
    """Generate, validate, and safely fall back to a structured case note."""
    if policy_id not in {f"P{i}" for i in range(1, 7)}:
        raise ValueError("policy_id must be P1 through P6.")

    if not 0 <= float(fraud_probability) <= 1:
        raise ValueError("fraud_probability must be between 0 and 1.")

    if int(max_retries) < 1:
        raise ValueError("max_retries must be at least 1.")

    prompt = build_prompt(
        txn_id,
        fraud_probability,
        amount_inr,
        policy_id,
    )

    for _ in range(int(max_retries)):
        try:
            response = generate(prompt, temperature=0.0)
            note = validate_case_note(response, policy_id)
            return note
        except (ValueError, ConnectionError):
            continue

    return safe_fallback(
        txn_id,
        fraud_probability,
        amount_inr,
        policy_id,
    )


def run_validation_check():
    """Verify that invalid priority values are rejected."""
    invalid_note = {
        "summary": "Test case",
        "priority": "very urgent",
        "recommended_team": "merchant_ops",
        "recommended_action": "hold_for_review",
        "policy_id": "P5",
    }

    try:
        validate_case_note(json.dumps(invalid_note), "P5")
    except ValueError as error:
        print("PASS: Invalid priority rejected:", error)
    else:
        print("FAIL: Invalid priority was accepted.")


if __name__ == "__main__":
    with open("case_scenarios.json", "r", encoding="utf-8") as file:
        scenarios = json.load(file)

    for scenario in scenarios:
        print(f"\nTransaction: {scenario['txn_id']}")

        # Temporary mapping for the supplied scenario test data.
        # The later service will pass the policy selected by retrieval.py.
        policy_id = {
            "T900001": "P1",
            "T900002": "P2",
            "T900003": "P3",
            "T900004": "P4",
            "T900005": "P5",
            "T900006": "P2",
            "T900007": "P1",
            "T900008": "P4",
            "T900009": "P3",
            "T900010": "P2",
        }[scenario["txn_id"]]

        result = generate_case_note(
            txn_id=scenario["txn_id"],
            fraud_probability=scenario["fraud_probability"],
            amount_inr=scenario["amount_inr"],
            policy_id=policy_id,
        )
        print(json.dumps(result, indent=2))

    print("\n--- Validation Check ---")
    run_validation_check()