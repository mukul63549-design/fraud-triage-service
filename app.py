
import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from casenote import generate_case_note
from retrieval import load_policies, build_retriever

BASE_DIR = Path(__file__).resolve().parent
LOG_PATH = BASE_DIR / "service.log"

logging.basicConfig(
    filename=LOG_PATH,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("fraud_triage")

app = FastAPI(title="Razorpay Merchant Risk Desk")
policies = load_policies()
retrieve_policy = build_retriever(policies)

model_bundle = None
request_metrics = {
    "total_requests": 0,
    "post_requests": 0,
    "errors": 0,
}


class Txn(BaseModel):
    txn_id: str = Field(min_length=1)
    customer_id: str = Field(min_length=1)
    amount_inr: float = Field(gt=0, le=10000000)
    hour_of_day: int = Field(ge=0, le=23)
    txn_count_last_24h: int = Field(ge=0)
    avg_amount_30d_inr: float = Field(gt=0)
    distance_from_home_km: float | None = Field(default=None, ge=0)
    is_new_device: int = Field(ge=0, le=1)
    account_age_days: int = Field(ge=0)
    merchant_category: str = Field(min_length=1)
    city: str = Field(min_length=1)
    payment_mode: str = Field(min_length=1)


class CaseIn(Txn):
    pass


def load_model():
    """Train the Part B model and prepare the bundle for scoring."""
    global model_bundle

    from train import (
        load_and_split,
        cross_validate_and_tune,
        select_threshold,
    )

    (
        X_train,
        X_val,
        _X_test,
        y_train,
        y_val,
        _y_test,
        median,
        columns,
    ) = load_and_split()

    final_model, _best_params = cross_validate_and_tune(
        X_train, y_train
    )
    final_model.fit(X_train, y_train)

    threshold = select_threshold(
        final_model, X_val, y_val
    )

    model_bundle = {
        "model": final_model,
        "columns": columns,
        "median_distance": float(median),
        "threshold": float(threshold),
    }


@app.on_event("startup")
def startup_event():
    load_model()


@app.middleware("http")
async def request_logger(request: Request, call_next):
    request_metrics["total_requests"] += 1

    if request.method.upper() == "POST":
        request_metrics["post_requests"] += 1

    started = time.perf_counter()
    status_code = 500

    try:
        response = await call_next(request)
        status_code = response.status_code

        if status_code >= 400:
            request_metrics["errors"] += 1

        return response
    except Exception:
        request_metrics["errors"] += 1
        raise
    finally:
        if request.method.upper() == "POST":
            elapsed_ms = (time.perf_counter() - started) * 1000
            logger.info(
                "POST %s status=%s duration_ms=%.2f",
                request.url.path,
                status_code,
                elapsed_ms,
            )


@app.exception_handler(Exception)
async def unexpected_error_handler(
    request: Request, exc: Exception
):
    logger.exception("Unhandled service error")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


@app.get("/health")
def health():
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    return {"status": "ok", "model_loaded": True}


def transaction_to_frame(txn: Txn) -> pd.DataFrame:
    if model_bundle is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded",
        )

    values = txn.model_dump()
    values.pop("txn_id", None)
    values.pop("customer_id", None)

    if values["distance_from_home_km"] is None:
        values["distance_from_home_km"] = (
            model_bundle["median_distance"]
        )

    values["is_night"] = int(
        values["hour_of_day"] >= 22
        or values["hour_of_day"] < 6
    )
    values["high_value"] = int(
        values["amount_inr"] >= 50000
    )
    values["amount_vs_avg_ratio"] = (
        values["amount_inr"]
        / max(values["avg_amount_30d_inr"], 1)
    )

    frame = pd.DataFrame([values])

    frame = pd.get_dummies(
        frame,
        columns=[
            "merchant_category",
            "city",
            "payment_mode",
        ],
        dtype=int,
    )

    frame = frame.reindex(
        columns=model_bundle["columns"],
        fill_value=0,
    )

    return frame


def score_transaction(txn: Txn) -> dict[str, Any]:
    if model_bundle is None:
        load_model()

    frame = transaction_to_frame(txn)
    model = model_bundle["model"]

    fraud_probability = float(
        model.predict_proba(frame)[0][1]
    )
    threshold = float(model_bundle["threshold"])

    high_risk_override = (
        txn.amount_inr >= 50000
        and txn.is_new_device == 1
    )
    flagged = (
        fraud_probability >= threshold
        or high_risk_override
    )

    return {
        "txn_id": txn.txn_id,
        "fraud_probability": round(fraud_probability, 6),
        "threshold": threshold,
        "flagged": bool(flagged),
        "decision": "review" if flagged else "approve",
        "high_risk_override": bool(high_risk_override),
    }


@app.post("/score")
def score(txn: Txn):
    return score_transaction(txn)


@app.post("/case")
def case(case_input: CaseIn):
    score_result = score_transaction(case_input)

    query = (
        f"Transaction amount INR {case_input.amount_inr}; "
        f"fraud probability {score_result['fraud_probability']}; "
        f"merchant category {case_input.merchant_category}; "
        f"city {case_input.city}; "
        f"payment mode {case_input.payment_mode}; "
        f"new device {case_input.is_new_device}; "
        f"decision {score_result['decision']}"
    )

    policy_result = retrieve_policy(query)

    if policy_result is None:
        policy_id = "P1"
        policy_text = "No close policy match; use manual review."
    else:
        policy = policy_result.get("policy", {})
        policy_id = policy.get(
            "policy_id",
            policy.get("id", "P1"),
        )
        policy_text = policy.get(
            "text",
            policy.get("description", str(policy)),
        )

    try:
        note = generate_case_note(
            case_input.txn_id,
            score_result["fraud_probability"],
            case_input.amount_inr,
            policy_id,
        )
    except ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Case-note provider unavailable",
        )

    return {
        "transaction": score_result,
        "policy_id": policy_id,
        "policy_text": policy_text,
        "case_note": note,
    }


@app.get("/metrics")
def metrics():
    return {
        **request_metrics,
        "fraud_threshold": (
            float(model_bundle["threshold"])
            if model_bundle is not None
            else None
        ),
    }