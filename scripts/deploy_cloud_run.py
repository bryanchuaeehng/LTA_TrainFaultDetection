"""Deploy the tested app to the current authenticated GCP project and verify HTTP health."""
from __future__ import annotations
import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path


def gcloud(*args, capture=True):
    command = ["gcloud", *args]
    return subprocess.run(command, check=True, text=True,
                          stdout=subprocess.PIPE if capture else None).stdout


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project")
    parser.add_argument("--region", default="asia-southeast1")
    parser.add_argument("--service", default="lta-nebulax-ps3")
    parser.add_argument("--memory", default="8Gi")
    args = parser.parse_args()
    if not shutil.which("gcloud"):
        sys.exit("Google Cloud CLI is unavailable. Run this script in an authenticated Google Cloud Shell or development terminal.")
    if not gcloud("auth", "list", "--filter=status:ACTIVE", "--format=value(account)").strip():
        sys.exit("No active GCP login. Run gcloud auth login in this terminal, then run this command again.")
    project = args.project or gcloud("config", "get-value", "project").strip()
    if not project or project == "(unset)":
        sys.exit("No GCP project selected. Run gcloud config set project YOUR_PROJECT_ID, then retry.")
    gcloud("projects", "describe", project, "--format=value(projectId)")
    root = Path(__file__).resolve().parents[1]
    # Repository tests are required before cloud mutation.
    subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=root, check=True)
    gcloud("services", "enable", "run.googleapis.com", "cloudbuild.googleapis.com",
           "artifactregistry.googleapis.com", "--project", project, "--quiet", capture=False)
    gcloud("run", "deploy", args.service, "--project", project, "--source", str(root),
           "--region", args.region, "--allow-unauthenticated", "--min-instances", "0",
           "--max-instances", "2", "--concurrency", "8", "--cpu", "2",
           "--memory", args.memory, "--timeout", "3600", "--port", "8080",
           "--session-affinity", "--quiet", capture=False)
    service = json.loads(gcloud("run", "services", "describe", args.service, "--project", project,
                                "--region", args.region, "--format=json"))
    status = service["status"]
    url = status["url"]
    for suffix in ["/_stcore/health", "/"]:
        for attempt in range(6):
            try:
                with urllib.request.urlopen(url + suffix, timeout=20) as response:
                    if response.status != 200:
                        raise RuntimeError(f"HTTP {response.status} for {suffix}")
                    response.read()
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2)
    logs = gcloud("run", "services", "logs", "read", args.service, "--project", project,
                  "--region", args.region, "--limit", "40", "--format=json")
    record = {"project_id": project, "region": args.region, "service": args.service,
              "revision": status.get("latestReadyRevisionName"), "url": url,
              "http_root": 200, "http_health": 200, "browser_and_live_inference": "still require verification"}
    out = root / "outputs"
    out.mkdir(exist_ok=True)
    (out / "deployment.json").write_text(json.dumps(record, indent=2) + "\n")
    (out / "cloud_run_logs.json").write_text(logs)
    print(json.dumps(record, indent=2))
    print("Open the URL and verify a real upload/inference/download before declaring deployment complete.")


if __name__ == "__main__":
    main()
