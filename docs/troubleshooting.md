# Troubleshooting Guide

Common issues and solutions when using RAGtriever.

## Table of Contents
- [Office Temp Files](#office-temp-files)
- [Offline Mode Issues](#offline-mode-issues)
- [Gemini service account Authentication](#vertex-ai-authentication)
- [Image Analysis](#image-analysis)
- [Query Issues](#query-issues)
- [Performance](#performance)

---

## Office Temp Files

### Problem
PowerPoint/Word/Excel creates lock files when documents are open (e.g., `~$document.pptx`). These cause extraction errors.

### Symptoms
```
PackageNotFoundError: Package not found at '.../~$document.pptx'
```

### Solution
1. **Close all Office applications** before scanning
2. **Update ignore patterns** in `config.toml`:
   ```toml
   [vault]
   ignore = [
       ".git/**",
       ".obsidian/cache/**",
       "**/.DS_Store",
       "**/~$*",           # Office temp files
       "**/.~lock.*",      # LibreOffice lock files
       "**/~*.tmp"         # Generic temp files
   ]
   ```

3. **Remove existing temp files:**
   ```bash
   find /path/to/vault -name "~$*" -delete
   ```

### Prevention
- Close Office apps before indexing
- Use the updated ignore patterns from `examples/config.toml.example`

---

## Offline Mode Issues

### Problem
Config has `offline_mode = true` but embedding model isn't cached locally.

### Symptoms
```
OSError: We couldn't connect to 'https://huggingface.co' to load the files,
and couldn't find them in the cached files.
```

### Solution

**Option 1: Download model first**
```toml
# Step 1: Temporarily disable offline mode
[embeddings]
offline_mode = false
```
```bash
# Step 2: Run scan to download model
ragtriever scan --config config.toml

# Step 3: Re-enable offline mode
```

**Option 2: Use environment variable**
```bash
HF_OFFLINE_MODE=0 ragtriever scan --config config.toml
```

**Option 3: Use a cached model**
```bash
# Check what's cached
ls ~/.cache/huggingface/hub/

# Update config to use cached model
```

### Prevention
- Check cached models: `ls ~/.cache/huggingface/hub/`
- Download models before enabling offline mode
- Document which models your team uses

---

## Gemini service account Authentication

### Problem: Credentials Not Found

**Symptoms:**
```
Gemini service account credentials file not found
```

**Solution:**
```bash
# Check file exists
ls -l $GOOGLE_APPLICATION_CREDENTIALS

# Check path is absolute in config.toml
[gemini_service_account]
credentials_file = "/absolute/path/to/service-account.json"  # Not relative!
```

### Problem: Permission Denied

**Symptoms:**
```
PermissionDenied: Service account does not have permission
```

**Solution:**
```bash
# Verify service account has correct role
gcloud projects get-iam-policy $PROJECT_ID \
    --flatten="bindings[].members" \
    --filter="bindings.members:serviceAccount:*ragtriever*"

# Grant role if missing
gcloud projects add-iam-policy-binding $PROJECT_ID \
    --member="serviceAccount:ragtriever-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
    --role="roles/aiplatform.user"
```

### Problem: Invalid Credentials

**Symptoms:**
```
google.auth.exceptions.DefaultCredentialsError
```

**Solution:**
```bash
# Test credentials manually
gcloud auth activate-service-account \
    --key-file=/path/to/service-account.json

# Verify API access
gcloud ai models list --region=global
```

---

## Image Analysis

### Problem: Rate Limit Errors

**Symptoms:**
```
Gemini service account analysis failed: TooManyRequests
```

**Expected Behavior:**
- This is normal with multiple images
- RAGtriever processes images sequentially
- Failed images show warning but scan continues
- Most images are successfully analyzed

**Not a Problem If:**
- Scan completes with "Scan complete"
- Most images have analysis data (check with query)

**Solution if persistent:**
```bash
# Check quota limits in Google Cloud Console
# https://console.cloud.google.com/apis/api/aiplatform.googleapis.com/quotas

# Consider using a different region with higher quota
[gemini_service_account]
location = "us-central1"  # Try different regions
```

### Problem: Model Not Available

**Symptoms:**
```
Model 'gemini-2.0-flash-exp' not found in region 'europe-west4'
```

**Solution:**
```bash
# Check model availability in your region
gcloud ai models list --region=europe-west4

# Use 'global' for automatic routing
[gemini_service_account]
location = "global"
```

### Problem: Deprecated SDK Warning

**Symptoms:**
```
UserWarning: This feature is deprecated as of June 24, 2025
```

**Status:**
- This is a known warning from Gemini service account SDK
- Does not affect functionality
- Will be addressed in future updates (see IMPROVEMENTS.md)

---

## Query Issues

### Problem: FTS5 Syntax Errors

**Symptoms:**
```
fts5: syntax error near "-"
```

**Cause:**
- Special characters in queries (hyphens, slashes, etc.)

**Solution:**
- **Automatic** as of commit 28680e5
- Queries are automatically escaped and wrapped in quotes
- Technical terms like "T-DXd", "CDK4/6i" work correctly

**Example:**
```bash
# These all work correctly:
ragtriever query "HR+/HER2-low breast cancer"
ragtriever query "T-DXd treatment"
ragtriever query "CDK4/6 inhibitor"
```

### Problem: No Results

**Check these:**

1. **Index exists and is populated:**
   ```bash
   sqlite3 ~/.ragtriever/indexes/myvault/vaultrag.sqlite \
       "SELECT COUNT(*) FROM documents WHERE deleted=0;"
   ```

2. **Embeddings are present:**
   ```bash
   sqlite3 ~/.ragtriever/indexes/myvault/vaultrag.sqlite \
       "SELECT COUNT(*) FROM embeddings;"
   ```

3. **Query the right vault:**
   ```bash
   ragtriever query --config config.toml "search term"
   ```

4. **Try a broader query:**
   ```bash
   # Too specific
   ragtriever query "exact document title from memory"

   # Better
   ragtriever query "key concepts from document"
   ```

---

## Performance

### Problem: Slow Indexing

**Check:**

1. **Device setting:**
   ```toml
   [embeddings]
   device = "mps"  # For Mac M1/M2/M3
   # device = "cuda"  # For NVIDIA GPU
   # device = "cpu"  # Slowest
   ```

2. **Batch size:**
   ```toml
   [embeddings]
   batch_size = 64  # Increase for faster indexing (if you have RAM)
   ```

3. **Image analysis provider:**
   ```toml
   [image_analysis]
   provider = "tesseract"  # Fastest (local OCR)
   # provider = "gemini-service-account"  # Slower (API calls) but better quality
   ```

### Problem: High Memory Usage

**Solution:**

1. **Reduce batch size:**
   ```toml
   [embeddings]
   batch_size = 16  # Reduce if running out of memory
   ```

2. **Use smaller model:**
   ```toml
   [embeddings]
   model = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dims, lightweight
   # vs
   model = "BAAI/bge-base-en-v1.5"  # 768 dims, more accurate but heavier
   ```

---

## Getting Help

If you encounter an issue not covered here:

1. **Check logs:**
   ```bash
   # Run with verbose output
   ragtriever scan --config config.toml 2>&1 | tee scan.log
   ```

2. **Verify configuration:**
   ```bash
   # Check config is valid TOML
   python -c "import tomllib; tomllib.loads(open('config.toml').read())"
   ```

3. **Test components individually:**
   ```bash
   # Test embeddings only
   python -c "from sentence_transformers import SentenceTransformer; \
              SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

   # Test Gemini service account only
   gcloud ai models list --region=global
   ```

4. **Report issues:**
   - GitHub: [rartzi/RAGtriever/issues](https://github.com/rartzi/RAGtriever/issues)
   - Include: config (redacted), error message, RAGtriever version
