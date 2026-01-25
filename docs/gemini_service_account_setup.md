# Gemini Service Account Setup Guide

This guide walks through setting up Gemini with GCP service account authentication for image analysis in RAGtriever.

## Why Gemini with Service Account?

Using Gemini via service account provides:
- **Enterprise authentication** via service account JSON (no API keys in environment)
- **Advanced vision models** including Gemini 2.0 and future Gemini 3 models
- **Regional deployment** with data residency controls
- **Better rate limits** and quota management for production use

## Prerequisites

1. **Google Cloud Project** with billing enabled
2. **Vertex AI API** enabled (Gemini models are accessed via Vertex AI)
3. **Service Account** with appropriate permissions

## Step 1: Create a Service Account

```bash
# Set your project ID
PROJECT_ID="your-project-id"

# Create service account
gcloud iam service-accounts create ragtriever-sa \
    --display-name="RAGtriever Image Analysis" \
    --project=$PROJECT_ID

# Grant Vertex AI User role
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ragtriever-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

## Step 2: Create and Download Credentials

```bash
# Create JSON key
gcloud iam service-accounts keys create ~/.config/gcloud/ragtriever-key.json \
    --iam-account=ragtriever-sa@${PROJECT_ID}.iam.gserviceaccount.com

# Secure the file
chmod 600 ~/.config/gcloud/ragtriever-key.json
```

## Step 3: Configure RAGtriever

### Option A: Configuration File

Edit your `config.toml`:

```toml
[image_analysis]
provider = "gemini-service-account"

[gemini_service_account]
project_id = "your-project-id"
location = "global"  # or "us-central1", "europe-west4", etc.
credentials_file = "/path/to/ragtriever-key.json"
model = "gemini-2.0-flash-exp"
```

### Option B: Environment Variables

```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="$HOME/.config/gcloud/ragtriever-key.json"
```

Then in `config.toml`:

```toml
[image_analysis]
provider = "gemini-service-account"

[gemini_service_account]
# Will use environment variables
location = "global"
model = "gemini-2.0-flash-exp"
```

## Available Models

### Current (Gemini 2.0)
- `gemini-2.0-flash-exp` - Fast, experimental (recommended for testing)
- `gemini-2.0-flash` - Production-ready (when available)

### Region Availability
- **global**: Recommended, routes to best available region
- **us-central1**: US data residency
- **europe-west4**: EU data residency
- **asia-northeast1**: Asia-Pacific

Check current availability:
```bash
gcloud ai models list --region=global --project=$PROJECT_ID
```

## Testing the Setup

1. **Verify credentials:**
```bash
# Check file exists and is readable
test -r $GOOGLE_APPLICATION_CREDENTIALS && echo "✓ Credentials file OK"

# Test authentication
gcloud auth activate-service-account \
    --key-file=$GOOGLE_APPLICATION_CREDENTIALS

gcloud ai models list --region=global --project=$PROJECT_ID
```

2. **Test with RAGtriever:**
```bash
# Scan a vault with images
ragtriever scan --config config.toml --full

# Query image content
ragtriever query --config config.toml "describe image content"
```

## Troubleshooting

### Authentication Errors

**Error:** `Credentials file not found`
```bash
# Check path is absolute and file exists
ls -l $GOOGLE_APPLICATION_CREDENTIALS
```

**Error:** `Permission denied`
```bash
# Fix file permissions
chmod 600 $GOOGLE_APPLICATION_CREDENTIALS
```

### API Errors

**Error:** `TooManyRequests` (429)
- This is normal with multiple images
- Images are processed sequentially to respect rate limits
- Failed images will show warning but scan continues

**Error:** `PermissionDenied` (403)
```bash
# Ensure service account has correct role
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:ragtriever-sa@*"
```

**Error:** `Model not found`
- Check model availability in your region
- Try `location = "global"` for automatic routing

### Performance Tips

1. **Batch processing**: Images are processed during vault scanning, not query time
2. **Rate limits**: Gemini API handles ~60 images/minute with current settings
3. **Costs**: Monitor usage at [cloud.google.com/billing](https://cloud.google.com/billing)

## Security Best Practices

1. **Never commit credentials:**
   ```bash
   # .gitignore already includes:
   # *.json
   # credentials/
   # secrets/
   ```

2. **Restrict service account permissions:**
   - Only grant `roles/aiplatform.user`
   - Consider using workload identity for GKE deployments

3. **Rotate keys periodically:**
   ```bash
   # Create new key
   gcloud iam service-accounts keys create new-key.json \
       --iam-account=ragtriever-sa@${PROJECT_ID}.iam.gserviceaccount.com

   # Update config
   # Test with new key

   # Delete old key
   gcloud iam service-accounts keys delete OLD_KEY_ID \
       --iam-account=ragtriever-sa@${PROJECT_ID}.iam.gserviceaccount.com
   ```

4. **Monitor usage:**
   ```bash
   # View API usage
   gcloud logging read \
       "resource.type=aiplatform.googleapis.com/Endpoint" \
       --limit=50 \
       --project=$PROJECT_ID
   ```

## Next Steps

- See [troubleshooting.md](troubleshooting.md) for common issues
- Review [IMPROVEMENTS.md](../IMPROVEMENTS.md) for planned enhancements
- Check [CLAUDE.md](../CLAUDE.md) for architecture details
