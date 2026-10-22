
import requests

BASE_URL = "http://127.0.0.1:8000"

sample_transaction = {
    "txn_id": "SMOKE001",
    "customer_id": "C001",
    "amount_inr": 60000,
    "hour_of_day": 23,
    "txn_count_last_24h": 3,
    "avg_amount_30d_inr": 5000,
    "distance_from_home_km": None,
    "is_new_device": 1,
    "account_age_days": 30,
    "merchant_category": "Electronics",
    "city": "Delhi",
    "payment_mode": "UPI"
}

def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"{status}: {name}")
    return condition

def main():
    passed = 0
    total = 11

    try:
        response = requests.get(f"{BASE_URL}/health", timeout=10)
        passed += check(
            "Health endpoint",
            response.status_code == 200
            and response.json().get("status") == "ok"
        )

        response = requests.get(f"{BASE_URL}/metrics", timeout=10)
        passed += check(
            "Metrics endpoint",
            response.status_code == 200
            and "total_requests" in response.json()
            and "errors" in response.json()
        )

        response = requests.post(
            f"{BASE_URL}/score",
            json=sample_transaction,
            timeout=30
        )
        score_data = response.json() if response.status_code == 200 else {}
        passed += check("Score endpoint", response.status_code == 200)
        passed += check(
            "Fraud probability is between 0 and 1",
            0 <= score_data.get("fraud_probability", -1) <= 1
        )
        passed += check(
            "Threshold is between 0 and 1",
            0 <= score_data.get("threshold", -1) <= 1
        )
        passed += check(
            "Decision is present",
            score_data.get("decision") in {"review", "approve"}
        )
        passed += check(
            "High-risk override is reported",
            isinstance(score_data.get("high_risk_override"), bool)
        )

        response = requests.post(
            f"{BASE_URL}/case",
            json=sample_transaction,
            timeout=30
        )
        case_data = response.json() if response.status_code == 200 else {}
        passed += check("Case endpoint", response.status_code == 200)
        passed += check(
            "Case note contains required fields",
            all(
                key in case_data.get("case_note", {})
                for key in (
                    "summary",
                    "priority",
                    "recommended_team",
                    "recommended_action",
                    "policy_id"
                )
            )
        )

        invalid_transaction = sample_transaction.copy()
        invalid_transaction["amount_inr"] = -100

        response = requests.post(
            f"{BASE_URL}/score",
            json=invalid_transaction,
            timeout=10
        )
        passed += check(
            "Invalid amount is rejected",
            response.status_code in (400, 422)
        )

        response = requests.post(
            f"{BASE_URL}/score",
            json={},
            timeout=10
        )
        passed += check(
            "Missing required fields are rejected",
            response.status_code == 422
        )

    except requests.RequestException as error:
        print(f"FAIL: API connection error: {error}")
        print("Make sure the FastAPI server is running.")
        return

    print(f"\nSmoke test result: {passed}/{total} checks passed.")
    if passed != total:
        raise SystemExit(1)

if __name__ == "__main__":
    main()