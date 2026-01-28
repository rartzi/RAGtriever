# Troubleshooting

## Issue: Office Temp Files (`~$*.pptx`)

**Symptom:** `PackageNotFoundError: Package not found at '~$document.pptx'`

**Solution:**
1. Close all Office applications before scanning
2. Add to ignore patterns in config.toml:
   ```toml
   [vault]
   ignore = [
       ".git/**",
       ".obsidian/cache/**",
       "**/.DS_Store",
       "**/~$*",           # Office temp files
       "**/.~lock.*"       # LibreOffice lock files
   ]
   ```
3. Remove existing temp files:
   ```bash
   find /path/to/vault -name "~$*" -delete
   ```

## Issue: Offline Mode with Uncached Model

**Symptom:** `OSError: We couldn't connect to 'https://huggingface.co'`

**Solution:**
```bash
# Option 1: Temporarily disable offline mode
export HF_OFFLINE_MODE=0
mneme scan --config config.toml

# Then re-enable offline mode
export HF_OFFLINE_MODE=1

# Option 2: Use a cached model
ls ~/.cache/huggingface/hub/
# Update config.toml to use one of the cached models
```

## Issue: Gemini service account Rate Limits (429 errors)

**Symptom:** `Gemini service account analysis failed: TooManyRequests`

**Expected Behavior:** This is normal with multiple images. The scan continues and processes other files.

**Solutions:**
- Most images will succeed despite some 429 errors
- Re-run scan - successful images are cached
- Consider using Gemini API or Tesseract for large image sets

## Issue: Gemini service account Authentication Errors

**Solution:**
1. Verify credentials file exists:
   ```bash
   test -r $GOOGLE_APPLICATION_CREDENTIALS && echo "OK"
   ```
2. Check environment variables:
   ```bash
   echo $GOOGLE_APPLICATION_CREDENTIALS
   echo $GOOGLE_CLOUD_PROJECT
   ```
3. Ensure `GEMINI_API_KEY` is **unset** (conflicts with Gemini service account):
   ```bash
   unset GEMINI_API_KEY
   ```
4. Test authentication:
   ```bash
   gcloud auth activate-service-account \
       --key-file=$GOOGLE_APPLICATION_CREDENTIALS
   gcloud ai models list --region=global
   ```

## Issue: FTS5 Syntax Errors with Special Characters

**Symptom:** `fts5: syntax error near "-"`

**Solution:** This is handled automatically as of recent versions. Queries with hyphens, slashes, etc. work correctly:
```bash
mneme query "T-DXd treatment"     # Works
mneme query "CDK4/6 inhibitor"    # Works
mneme query "HR+/HER2- cancer"    # Works
```

## Success Criteria

After running a scan, verify:
- [ ] Scan completes without fatal errors
- [ ] Rate limit errors (429) are logged but don't stop scan
- [ ] Database created: `~/.mneme/indexes/<vault>/vaultrag.sqlite`
- [ ] Query returns results
- [ ] Image content is searchable (if images in vault)
- [ ] Metadata includes analysis_provider and model name
