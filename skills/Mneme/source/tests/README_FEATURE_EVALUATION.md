# Feature Evaluation Tests

Comprehensive evaluation tests for Mneme's new retrieval features:
- Heading boost (H1, H2, H3)
- Tag boost (matching query terms to document tags)
- MMR diversity (limiting chunks per document)

## Running the Tests

The tests require network access to download embedding models on first run and are marked as `@pytest.mark.manual` for optional execution.

### Run all evaluation tests
```bash
pytest tests/test_feature_evaluation.py -v -m manual
```

### Run specific test
```bash
# Comprehensive comparison (baseline vs all features)
pytest tests/test_feature_evaluation.py::test_feature_comparison_comprehensive -v -m manual

# Test feature independence
pytest tests/test_feature_evaluation.py::test_feature_independence -v -m manual

# Test MMR diversity effectiveness
pytest tests/test_feature_evaluation.py::test_mmr_diversity_effectiveness -v -m manual
```

### Run without markers filter
```bash
pytest tests/test_feature_evaluation.py -v
```

## Test Structure

### test_feature_comparison_comprehensive
The main evaluation test that compares 5 configurations:

1. **Baseline**: All features disabled
2. **With Heading Boost**: Only heading boost enabled
3. **With Tag Boost**: Only tag boost enabled
4. **With MMR Diversity**: Only MMR enabled
5. **All Features**: All three enabled

**Metrics measured:**
- Precision@5, Precision@10
- Diversity@5, Diversity@10 (unique documents ratio)
- Relevance percentage

**Output:**
- Per-query metrics for each configuration
- Comparison table with % improvements
- Overall averages across all queries
- Assertions to verify improvements

### test_feature_independence
Verifies that features work together without conflicts:
- Enables all features (heading boost, tag boost, MMR)
- Verifies metadata is correctly added
- Checks diversity is maintained

### test_mmr_diversity_effectiveness
Specifically tests MMR diversity:
- Compares results with/without MMR
- Measures document diversity improvement
- Verifies `max_per_document` constraint is enforced

## Test Vault

The tests create a synthetic vault with 10 documents designed to test each feature:

| Document | Purpose |
|----------|---------|
| `kubernetes-architecture.md` | H1 title "Kubernetes Architecture" for heading boost |
| `ml-deployment.md` | Tags: #machine-learning #mlops for tag boost |
| `ml-training.md` | Tags: #machine-learning for tag boost |
| `api-design.md` | Multiple chunks with "API implementation details" for MMR |
| `api-security.md` | Multiple chunks for MMR diversity |
| `api-testing.md` | Multiple chunks for MMR diversity |
| `docker-guide.md` | H2 headings + #docker tag for combined boost |
| `database-design.md` | Long document with many chunks for MMR |
| `jenkins-cicd.md` | Irrelevant document (noise) |
| `lambda-serverless.md` | Irrelevant document (noise) |

## Expected Improvements

Based on the test design:

**Heading Boost:**
- Should improve "kubernetes architecture" query (H1 match)
- Should improve "docker container orchestration" query (H2 match)

**Tag Boost:**
- Should improve "machine learning deployment" query (#machine-learning)
- Should improve "docker container orchestration" query (#docker)

**MMR Diversity:**
- Should increase unique documents in results for "API implementation details"
- Should limit chunks per document to max_per_document (default: 2)
- Should improve diversity for "database schema design patterns" (long doc)

**All Features Combined:**
- Should show cumulative improvements
- P@5 should be ≥ baseline for all queries
- Overall average improvement should be positive

## Performance Notes

- First run downloads embedding model (~90MB for all-MiniLM-L6-v2)
- Full evaluation takes ~2-5 minutes (indexing + 5 queries × 5 configurations)
- Individual tests can be run separately for faster iteration

## Troubleshooting

### Model download fails
```bash
# Set offline_mode=False in config or allow network access
# Model will be cached in ~/.cache/huggingface/
```

### Tests fail with import errors
```bash
# Ensure dev dependencies installed
pip install -e ".[dev]"
```

### Baseline shows high precision
This is expected if the test vault is small and well-structured. The goal is to verify that features maintain or improve baseline, not necessarily show large improvements on synthetic data.
