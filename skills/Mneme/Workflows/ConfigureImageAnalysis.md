# ConfigureImageAnalysis Workflow

Configure image analysis providers for extracting content from images.

## Trigger Phrases
- "configure images"
- "setup gemini"
- "image analysis"
- "configure image extraction"

## Provider Options

| Provider | Auth | Latency | Use Case |
|----------|------|---------|----------|
| **tesseract** | None | ~100ms | Local OCR, text only |
| **gemini** | API key | ~500ms | Personal, simple setup |
| **gemini-service-account** | Service acct | ~500ms | Enterprise GCP, IAM |
| **aigateway** | Gateway key | ~1000ms | Enterprise Microsoft |
| **off** | N/A | 0ms | Text-only vaults |

## Procedure by Provider

### Tesseract (Local OCR)

**Requirements:**
```bash
# macOS
brew install tesseract

# Ubuntu/Debian
apt-get install tesseract-ocr

# Python package
pip install pytesseract
```

**Config:**
```toml
[image_analysis]
provider = "tesseract"
```

### Gemini API (Personal)

**Requirements:**
- GEMINI_API_KEY environment variable

**Config:**
```toml
[image_analysis]
provider = "gemini"
gemini_model = "gemini-2.0-flash"
```

**Setup:**
```bash
export GEMINI_API_KEY="your-api-key"
```

### Gemini Service Account (Enterprise)

**Requirements:**
- GCP project with Vertex AI enabled
- Service account with `roles/aiplatform.user`
- Credentials JSON file

**Config:**
```toml
[image_analysis]
provider = "gemini-service-account"

[gemini_service_account]
project_id = "your-gcp-project-id"
location = "global"
credentials_file = "/path/to/service-account.json"
model = "gemini-2.0-flash-exp"
```

**Setup:**
```bash
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/creds.json"
unset GEMINI_API_KEY  # Important: avoid conflicts
```

**Verification:**
```bash
# Check credentials
test -r $GOOGLE_APPLICATION_CREDENTIALS && echo "OK"

# Test authentication
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS
gcloud ai models list --region=global
```

### Disable Image Analysis

**Config:**
```toml
[image_analysis]
provider = "off"
```

## Performance Tuning

For API-based providers, configure workers in `config.toml`:

```toml
[indexing]
image_workers = 8  # Parallel API workers (default: 8)
```

**Recommendations:**
- 8-10 workers for <100 images
- Expect 429 errors with large image sets (normal behavior)
- gemini-2.5-flash is 3x faster than gemini-3-pro-preview

## Checklist

### Gemini Service Account Checklist
- [ ] Credentials file exists and is readable
- [ ] Use `location = "global"` for best model availability
- [ ] `GEMINI_API_KEY` is unset (avoid conflicts)
- [ ] Service account has `roles/aiplatform.user` permission

## Testing

After configuration:

```bash
# Run scan with a few images
mneme scan --config config.toml --full --log-file logs/scan_$(date +%Y%m%d_%H%M%S).log

# Check logs for image processing
grep -i "image" logs/scan*.log

# Query for image content
mneme query --config config.toml "diagram" --k 5
```
