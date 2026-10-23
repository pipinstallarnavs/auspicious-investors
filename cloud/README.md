# Vertex AI experiments on Cloud Run

The cloud runner uses Gemini through **Vertex AI with Application Default Credentials**.
The Cloud Run job runs as a dedicated service account; no Gemini API key or downloaded
service-account key is required. Switching off the local PC does not stop the job.

## First preliminary experiment

`examples/vertex-pilot.json` contains 2,400 exploratory prompts: five languages, four events,
both roles, twelve scenarios, the unspecified persona, one template, one cue, and five
conditions. It preserves scenario breadth while keeping the first run smaller than the full
persona/template design. Do not interpret it as a confirmatory experiment.

The selected model is `gemini-3.8-flash`, global endpoint, `LOW` thinking, temperature 1.0,
best-effort seed 42, and at most 1,536 output tokens. Both thinking and response tokens count
toward output limits and charges. The requested model, returned version, generation settings,
raw responses, usage, input materials, and analysis tables are retained.

References, verified 22 October 2026:

- [Model availability and thinking levels](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/gemini/3-8-flash)
- [Vertex/Agent Platform pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing)
- [Cloud Run job service identity](https://docs.cloud.google.com/run/docs/configuring/jobs/service-identity)

## Budget and execution

`cloud/run-template.json` is the run specification. It reserves a US$90 model budget within
the user's US$100 initial ceiling, leaving US$10 for infrastructure. At the recorded introductory
prices, the conservative model bound for the shipped pilot is US$55.44. The bound uses UTF-8
request bytes plus overhead as a conservative text-token estimate, maximum output tokens, and
both allowed HTTP attempts for every prompt. Caching discounts are ignored. This is a planning
guard rather than a Cloud Billing account spending limit; infrastructure and taxes are separate.
Verify prices again after 31 December 2026; the worker rejects an expired price specification.

Use **one task, parallelism one, no Cloud Run task retries**, 1 CPU, 1 GiB memory, and a three-hour
task timeout. The worker provides bounded concurrent Vertex requests internally. Disabling task
retries prevents automatic retries from multiplying the planned request budget.

Build the image with Cloud Build and deploy its immutable digest. The build identity needs
Cloud Build builder access; the runtime identity needs Vertex AI User on the dedicated project
and Storage Object User only on its results bucket. Use a private bucket with uniform access
and public-access prevention. Cloud Run and the storage bucket can use `us-central1`; model
requests use the explicitly recorded global endpoint.

Upload a concrete run specification before executing the job:

```powershell
gcloud storage cp artifacts/cloud-preparation/run.json gs://YOUR_BUCKET/runs/YOUR_RUN_ID/run.json --project=YOUR_PROJECT
gcloud run jobs create mahurat-pilot --image=YOUR_IMAGE_DIGEST --region=us-central1 --project=YOUR_PROJECT --service-account=YOUR_RUNNER_ACCOUNT --tasks=1 --parallelism=1 --max-retries=0 --task-timeout=3h --cpu=1 --memory=1Gi --args="--bucket,YOUR_BUCKET,--run-id,YOUR_RUN_ID"
gcloud run jobs execute mahurat-pilot --region=us-central1 --project=YOUR_PROJECT --async
```

Record `source_revision` and `container_image` in the run specification. Always use a fresh run
identifier when changing the design, generation settings, model, or prices. A completed run is
idempotent: executing that same identifier again returns its completed status without new calls.

Run a separate smoke experiment with `examples/pilot-controls.json` first. Confirm that its
ten responses are valid and that upload/download and checksums work before launching the pilot.

## Checkpoints and outputs

The worker writes a consistent SQLite backup and progress JSON to Cloud Storage at least every
30 seconds while results are being committed, and forces a final snapshot. SQLite runs on the
container's local filesystem; it is never placed on a Cloud Storage filesystem mount. Snapshots
are closed and converted to a self-contained database before upload, including committed WAL
pages. A later explicit resume can restore that checkpoint without repeating stored responses.

`runs/YOUR_RUN_ID/` in the bucket contains:

- `run.json`, `status.json`, `progress.json`, and `cost-plan.json`.
- `inputs/`: the design, stimuli, generated trials and review sidecar.
- `model/`: raw responses, metadata, summary, and SQLite journal.
- `analysis/`: paired effects, controls, language interactions, validity, bounds and return errors.
- `checksums.json` and `results.zip` for an integrity-checked local archive.

Completed responses, including invalid and blocked outputs, are preserved. Completion means
the design has been attempted, not that every response was valid. Inspect validity, coverage,
errors, and the model version before interpreting effect estimates.

A GCS lease prevents concurrent executions from writing to one run identifier. Graceful failures
release it. If an execution is killed before cleanup, verify that it is no longer running before
manually clearing its lease for an explicit resume. Cloud checkpoints can lag by one interval;
a force-killed task may require repeating the few requests not yet included in its checkpoint.

## Download to the local PC

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/Sync-CloudResults.ps1 -RunId YOUR_RUN_ID
```

The script waits for a terminal cloud status, downloads the whole run into `results/YOUR_RUN_ID`,
and verifies the published SHA-256 checksums. It can be rerun safely. A failed run's available
artifacts are also downloaded for diagnosis. A PC cannot receive files while powered off;
the private cloud copy remains available until the PC is back online.

For this initial deployment, a local scheduled download can run periodically and at the next
sign-in. Its task name is passed to the script so it removes only that task after a successful
download. The cloud experiment itself is independent of the download task.
