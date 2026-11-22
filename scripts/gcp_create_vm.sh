#!/usr/bin/env bash
# Run in Google Cloud Shell (shell.cloud.google.com), signed in to the account with credits.
# Creates a 16-vCPU CPU VM. No GPU quota needed: training takes ~30-60 min on CPU.
set -euo pipefail
ZONE=${ZONE:-us-central1-a}
gcloud services enable compute.googleapis.com run.googleapis.com \
  cloudbuild.googleapis.com artifactregistry.googleapis.com
gcloud compute instances create fraudgnn --zone="$ZONE" \
  --machine-type=n2-standard-16 --boot-disk-size=100GB \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --scopes=cloud-platform
echo "Next: gcloud compute scp ~/kaggle.json fraudgnn:~ --zone=$ZONE && gcloud compute ssh fraudgnn --zone=$ZONE"
