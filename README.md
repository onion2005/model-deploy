## MLOps for SageMaker Endpoint Deployment

This repository contains the deployment code for promoting approved ML models to real-time inference endpoints on AWS SageMaker. It is part of a SageMaker MLOps project and is triggered automatically by AWS CodePipeline when a new model package is approved in the Model Registry.

**Model:** Bank Marketing classification (UCI dataset) — predicts whether a client will subscribe to a term deposit (`yes` / `no`).  
**Input:** semicolon-delimited CSV row of 20 features (no header).  
**Output:** `yes` or `no`.

---

## Deploy Flow

```
 Model Registry (SageMaker)
         |
         | new ModelPackage approved
         v
 ┌─────────────────────┐
 │     CodePipeline    │
 │      triggered      │
 └─────────────────────┘
         |
         v
 ┌─────────────────────────────┐
 │         BUILD STAGE         │
 │  buildspec.yml + build.py   │
 │                             │
 │  - Fetch latest approved    │
 │    model package ARN        │
 │  - Merge stage configs with │
 │    runtime values (ARN,     │
 │    IAM role, S3 paths, tags)│
 │  - Export staging/prod      │
 │    config JSON artifacts    │
 │  - Package CloudFormation   │
 │    template to S3           │
 └─────────────────────────────┘
         |
         v
 ┌─────────────────────────────┐
 │       STAGING DEPLOY        │
 │  endpoint-config-template   │
 │  staging-config.json        │
 │                             │
 │  CloudFormation creates:    │
 │  - SageMaker Model          │
 │  - EndpointConfig           │
 │    (ml.m5.large x1,         │
 │     100% data capture)      │
 │  - Endpoint: <proj>-staging │
 │  - Auto-scaling (1–2 inst.) │
 │  - CloudWatch alarms        │
 │    (latency, 4XX, 5XX)      │
 └─────────────────────────────┘
         |
         v
 ┌─────────────────────────────┐
 │       STAGING TEST          │
 │  test/buildspec.yml         │
 │  test/test.py               │
 │                             │
 │  - Assert endpoint InService│
 │  - Confirm data capture on  │
 │  - Invoke endpoint with 10  │
 │    labeled samples (5 yes,  │
 │    5 no) one row at a time  │
 │  - Validate response format │
 │  - Check per-request latency│
 │  - Assert accuracy >= 70%   │
 │    (blocks pipeline if not) │
 └─────────────────────────────┘
         |
         v
 ┌─────────────────────────────┐
 │      MANUAL APPROVAL        │
 │   (AWS CodePipeline console)│
 └─────────────────────────────┘
         |
         v
 ┌─────────────────────────────┐
 │        PROD DEPLOY          │
 │  endpoint-config-template   │
 │  prod-config.json           │
 │                             │
 │  CloudFormation creates:    │
 │  - SageMaker Model          │
 │  - EndpointConfig           │
 │    (ml.m5.xlarge x2,        │
 │     20% data capture)       │
 │  - Endpoint: <proj>-prod    │
 │  - Auto-scaling (2–6 inst.) │
 │  - CloudWatch alarms        │
 │    (latency, 4XX, 5XX)      │
 │  - Model Monitor schedule   │
 │    (hourly, if baseline set)│
 └─────────────────────────────┘
```

### Key steps explained

| Step | File(s) | What it does |
|---|---|---|
| Detect approved model | `build.py` | Queries SageMaker Model Registry for the latest `Approved` package in the configured `ModelPackageGroup` |
| Build config artifacts | `build.py`, `buildspec.yml` | Merges stage configs with runtime values (model ARN, IAM role, S3 data capture path, project tags) and exports them as JSON |
| Package infrastructure | `buildspec.yml` | Runs `aws cloudformation package` on `endpoint-config-template.yml`, uploading assets to S3 |
| Deploy staging endpoint | `endpoint-config-template.yml`, `staging-config.json` | CloudFormation creates a `SageMaker::Endpoint` named `<project>-staging` with auto-scaling (1–2 instances) and CloudWatch alarms |
| Test staging endpoint | `test/buildspec.yml`, `test/test.py` | Asserts endpoint is `InService`, invokes it with 10 labeled Bank Marketing rows, checks response format and latency, **blocks pipeline** if accuracy < 70% |
| Manual approval gate | AWS CodePipeline console | A human must approve promotion before the prod deploy proceeds |
| Deploy prod endpoint | `endpoint-config-template.yml`, `prod-config.json` | Same template deployed as `<project>-prod` — `ml.m5.xlarge x2`, auto-scaling up to 6 instances, 20% data capture, optional Model Monitor |

---

## Repository Layout

| File | Purpose |
|---|---|
| `build.py` | Fetches latest approved model package ARN, merges and exports stage config files |
| `buildspec.yml` | CodeBuild spec for the Build stage — runs `build.py` and packages the CloudFormation template |
| `endpoint-config-template.yml` | CloudFormation template defining the endpoint, auto-scaling policy, CloudWatch alarms, and optional Model Monitor schedule |
| `staging-config.json` | Parameters for the staging endpoint (instance type/count, data capture, alarms, monitoring) |
| `prod-config.json` | Parameters for the prod endpoint |
| `test/buildspec.yml` | CodeBuild spec for the staging test stage |
| `test/test.py` | Validates the staging endpoint is healthy and runs accuracy/latency checks against 10 labeled samples |
| `test/data/bank-additional/bank-additional-full.csv` | Bank Marketing dataset used to source test samples |

---

## Configuration

### Staging (`staging-config.json`)

| Parameter | Value | Notes |
|---|---|---|
| `EndpointInstanceType` | `ml.m5.large` | Sufficient for staging load |
| `EndpointInstanceCount` | `1` | Minimum instances |
| `EndpointMaxInstanceCount` | `2` | Auto-scaling ceiling |
| `TargetInvocationsPerInstance` | `70` | Invocations/min/instance before scaling out |
| `SamplingPercentage` | `100` | Full data capture for staging observability |
| `AlarmEmail` | _(empty)_ | Set to receive CloudWatch alarm emails |
| `BaselineConstraintsUri` | _(empty)_ | Set after running a Model Monitor baseline job |
| `BaselineStatisticsUri` | _(empty)_ | Set after running a Model Monitor baseline job |
| `ModelMonitorImageUri` | us-east-1 default | Update for your deployment region — see [AWS docs](https://docs.aws.amazon.com/sagemaker/latest/dg/model-monitor-pre-built-container.html) |

### Prod (`prod-config.json`)

| Parameter | Value | Notes |
|---|---|---|
| `EndpointInstanceType` | `ml.m5.xlarge` | Larger instance for production throughput |
| `EndpointInstanceCount` | `2` | Minimum 2 for high availability |
| `EndpointMaxInstanceCount` | `6` | Auto-scaling ceiling |
| `TargetInvocationsPerInstance` | `70` | Invocations/min/instance before scaling out |
| `SamplingPercentage` | `20` | Reduced capture to balance cost and coverage |
| `AlarmEmail` | _(empty)_ | **Recommended:** set to your on-call email |
| `BaselineConstraintsUri` | _(empty)_ | **Recommended:** set to enable data drift detection |
| `BaselineStatisticsUri` | _(empty)_ | **Recommended:** set to enable data drift detection |
| `ModelMonitorImageUri` | us-east-1 default | Update for your deployment region |

---

## Model Monitor setup

The monitoring schedule is created automatically when both `BaselineConstraintsUri` and `BaselineStatisticsUri` are set. To generate the baseline files:

```python
from sagemaker.model_monitor import DefaultModelMonitor

monitor = DefaultModelMonitor(role=role)
monitor.suggest_baseline(
    baseline_dataset="s3://your-bucket/training-data/bank-additional-full.csv",
    dataset_format={"csv": {"header": True, "output_columns_position": "END"}},
    output_s3_uri="s3://your-bucket/baseline-output/",
)
```

Then set `BaselineConstraintsUri` and `BaselineStatisticsUri` in `prod-config.json` to the S3 paths of the generated `constraints.json` and `statistics.json` files.

The monitor runs **hourly** and writes violation reports to `{DataCaptureUploadPath}/monitoring-output/`.

---

## Staging test gate

The pipeline will **fail and block prod promotion** if the staging endpoint accuracy drops below 70%. The test sends 10 labeled rows (5 `yes`, 5 `no`) from the Bank Marketing dataset and compares predictions against known labels.

To tighten or loosen the gate, update `ACCURACY_THRESHOLD` in `test/test.py`:

```python
ACCURACY_THRESHOLD = 0.7  # require at least 70% correct
```

To add more test cases, append `(features_csv, expected_label)` tuples to `TEST_SAMPLES` in the same file. Features must be semicolon-delimited with no header row.
