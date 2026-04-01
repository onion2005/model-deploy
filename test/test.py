import argparse
import json
import logging
import os
import time

import boto3
from botocore.exceptions import ClientError

logger = logging.getLogger(__name__)
sm_client = boto3.client("sagemaker")
sm_runtime_client = boto3.client("sagemaker-runtime")

# Bank Marketing dataset (bank-additional-full.csv) — features only, semicolon-delimited, no header.
# Columns: age;job;marital;education;default;housing;loan;contact;month;day_of_week;
#          duration;campaign;pdays;previous;poutcome;emp.var.rate;cons.price.idx;
#          cons.conf.idx;euribor3m;nr.employed
# Output: "yes" or "no" (subscribed a term deposit)

ACCURACY_THRESHOLD = 0.7   # pipeline fails if fewer than 70% of predictions are correct
LATENCY_WARNING_MS = 500   # log a warning if a single invocation exceeds this
VALID_PREDICTIONS = {"yes", "no", "y", "n"}

# 10 labeled rows taken directly from bank-additional-full.csv (5 yes, 5 no)
TEST_SAMPLES = [
    # "yes" cases — long call duration is the strongest signal
    ("41;blue-collar;divorced;basic.4y;unknown;yes;no;telephone;may;mon;1575;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191", "yes"),
    ("49;entrepreneur;married;university.degree;unknown;yes;no;telephone;may;mon;1042;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191", "yes"),
    ("49;technician;married;basic.9y;no;no;no;telephone;may;mon;1467;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191", "yes"),
    ("41;technician;married;professional.course;unknown;yes;no;telephone;may;mon;579;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191", "yes"),
    ("45;blue-collar;married;basic.9y;unknown;yes;no;telephone;may;mon;461;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191", "yes"),
    # "no" cases — short call, telephone contact, no prior contact history
    ("56;housemaid;married;basic.4y;no;no;no;telephone;may;mon;261;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191", "no"),
    ("57;services;married;high.school;unknown;no;no;telephone;may;mon;149;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191", "no"),
    ("37;services;married;high.school;no;yes;no;telephone;may;mon;226;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191", "no"),
    ("40;admin.;married;basic.6y;no;no;no;telephone;may;mon;151;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191", "no"),
    ("56;services;married;high.school;no;no;yes;telephone;may;mon;307;1;999;0;nonexistent;1.1;93.994;-36.4;4.857;5191", "no"),
]


def invoke_endpoint(endpoint_name):
    """
    Invokes the endpoint one row at a time with Bank Marketing feature data (semicolon-delimited CSV).
    Validates response format, checks per-sample latency, and asserts accuracy >= ACCURACY_THRESHOLD.
    Raises an exception (blocking pipeline promotion) if accuracy is too low.
    """
    results = []
    latencies_ms = []

    for payload, expected in TEST_SAMPLES:
        t0 = time.time()
        response = sm_runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType="text/csv",
            Body=payload,
        )
        latency_ms = (time.time() - t0) * 1000
        latencies_ms.append(latency_ms)

        prediction = response["Body"].read().decode("utf-8").strip().lower()

        if prediction not in VALID_PREDICTIONS:
            raise Exception(
                f"Unexpected endpoint response: '{prediction}'. Expected one of {VALID_PREDICTIONS}."
            )

        if latency_ms > LATENCY_WARNING_MS:
            logger.warning(f"High latency {latency_ms:.0f}ms for input: {payload[:50]}...")

        normalised = {"y": "yes", "n": "no"}.get(prediction, prediction)
        match = normalised == expected
        logger.info(
            f"Expected: {expected:3s} | Predicted: {prediction:3s} | "
            f"{'PASS' if match else 'FAIL'} | {latency_ms:.0f}ms"
        )
        results.append({
            "input": payload,
            "expected": expected,
            "predicted": prediction,
            "match": match,
            "latency_ms": round(latency_ms, 1),
        })

    passed = sum(r["match"] for r in results)
    total = len(results)
    accuracy = passed / total
    avg_latency_ms = sum(latencies_ms) / len(latencies_ms)

    logger.info(
        f"Accuracy: {passed}/{total} ({accuracy:.0%}) | "
        f"Avg latency: {avg_latency_ms:.0f}ms | Max latency: {max(latencies_ms):.0f}ms"
    )

    if accuracy < ACCURACY_THRESHOLD:
        raise Exception(
            f"Accuracy {accuracy:.0%} is below the required threshold of {ACCURACY_THRESHOLD:.0%} "
            f"({passed}/{total} correct). Blocking pipeline promotion."
        )

    return {
        "endpoint_name": endpoint_name,
        "success": True,
        "accuracy": round(accuracy, 4),
        "avg_latency_ms": round(avg_latency_ms, 1),
        "results": results,
    }


def test_endpoint(endpoint_name):
    """
    Describe the endpoint and ensure InSerivce, then invoke endpoint.  Raises exception on error.
    """
    error_message = None
    try:
        # Ensure endpoint is in service
        response = sm_client.describe_endpoint(EndpointName=endpoint_name)
        status = response["EndpointStatus"]
        if status != "InService":
            error_message = f"SageMaker endpoint: {endpoint_name} status: {status} not InService"
            logger.error(error_message)
            raise Exception(error_message)

        # Output if endpoint has data capture enbaled
        endpoint_config_name = response["EndpointConfigName"]
        response = sm_client.describe_endpoint_config(EndpointConfigName=endpoint_config_name)
        if "DataCaptureConfig" in response and response["DataCaptureConfig"]["EnableCapture"]:
            logger.info(f"data capture enabled for endpoint config {endpoint_config_name}")

        # Call endpoint to handle
        return invoke_endpoint(endpoint_name)
    except ClientError as e:
        error_message = e.response["Error"]["Message"]
        logger.error(error_message)
        raise Exception(error_message)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-level", type=str, default=os.environ.get("LOGLEVEL", "INFO").upper())
    parser.add_argument("--import-build-config", type=str, required=True)
    parser.add_argument("--export-test-results", type=str, required=True)
    args, _ = parser.parse_known_args()

    # Configure logging to output the line number and message
    log_format = "%(levelname)s: [%(filename)s:%(lineno)s] %(message)s"
    logging.basicConfig(format=log_format, level=args.log_level)

    # Load the build config
    with open(args.import_build_config, "r") as f:
        config = json.load(f)

    # Get the endpoint name from sagemaker project name
    endpoint_name = "{}-{}".format(
        config["Parameters"]["SageMakerProjectName"], config["Parameters"]["StageName"]
    )
    results = test_endpoint(endpoint_name)

    # Print results and write to file
    logger.debug(json.dumps(results, indent=4))
    with open(args.export_test_results, "w") as f:
        json.dump(results, f, indent=4)
