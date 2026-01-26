#!/usr/bin/env python3
"""
Generate comparison report for parallel Gemini 3 Pro provider testing.

Parses stdout, logs, and profiling reports to compare:
- Service Account provider (gemini-service-account)
- AI Gateway provider (aigateway)
"""

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
import statistics


@dataclass
class ScanStats:
    """Statistics from scan stdout output."""
    files_scanned: int = 0
    files_indexed: int = 0
    files_failed: int = 0
    chunks_created: int = 0
    embeddings_created: int = 0
    images_processed: int = 0
    elapsed_seconds: float = 0.0


@dataclass
class ImageMetrics:
    """Per-image processing metrics from DEBUG logs."""
    latencies_ms: List[float]
    success_count: int = 0
    failure_count: int = 0
    retry_count: int = 0
    circuit_breaker_trips: int = 0
    errors_by_category: Dict[str, int] = None

    def __post_init__(self):
        if self.errors_by_category is None:
            self.errors_by_category = defaultdict(int)


@dataclass
class ProfilingStats:
    """Top functions from profiling report."""
    top_functions: List[Dict[str, Any]]
    total_time: float = 0.0


def parse_stdout_stats(stdout_path: Path) -> ScanStats:
    """Parse ScanStats from stdout capture."""
    stats = ScanStats()

    content = stdout_path.read_text()

    # Pattern: "Scan complete: N files, M chunks in X.Xs"
    match = re.search(r'Scan complete:\s+(\d+)\s+files?,\s+(\d+)\s+chunks?\s+in\s+([\d.]+)s', content)
    if match:
        stats.files_scanned = int(match.group(1))
        stats.chunks_created = int(match.group(2))
        stats.elapsed_seconds = float(match.group(3))

    # Pattern: "(K images processed)"
    match = re.search(r'\((\d+)\s+images?\s+processed\)', content)
    if match:
        stats.images_processed = int(match.group(1))

    # Pattern: "(L files failed)"
    match = re.search(r'\((\d+)\s+files?\s+failed\)', content)
    if match:
        stats.files_failed = int(match.group(1))

    # Files indexed = files scanned - files failed
    stats.files_indexed = stats.files_scanned - stats.files_failed
    stats.embeddings_created = stats.chunks_created  # Typically 1:1

    return stats


def parse_log_metrics(log_path: Path, provider_name: str) -> ImageMetrics:
    """Parse image processing metrics from DEBUG log."""
    metrics = ImageMetrics(latencies_ms=[])

    content = log_path.read_text()

    # Pattern: "[provider_name] source: SUCCESS - 123.0ms"
    success_pattern = rf'\[{re.escape(provider_name)}\]\s+([^:]+):\s+SUCCESS\s+-\s+([\d.]+)ms'
    for match in re.finditer(success_pattern, content):
        latency = float(match.group(2))
        metrics.latencies_ms.append(latency)
        metrics.success_count += 1

    # Pattern: "[provider_name] source: ERROR_CATEGORY error (attempt N/M)"
    retry_pattern = rf'\[{re.escape(provider_name)}\]\s+([^:]+):\s+(\w+)\s+error\s+\(attempt\s+(\d+)/(\d+)\)'
    for match in re.finditer(retry_pattern, content):
        category = match.group(2)
        attempt = int(match.group(3))
        metrics.errors_by_category[category] += 1
        if attempt > 1:
            metrics.retry_count += 1

    # Pattern: "[provider_name] Circuit breaker OPEN"
    cb_pattern = rf'\[{re.escape(provider_name)}\]\s+Circuit breaker OPEN'
    metrics.circuit_breaker_trips = len(re.findall(cb_pattern, content))

    metrics.failure_count = sum(metrics.errors_by_category.values())

    return metrics


def parse_profiling(profile_path: Path, top_n: int = 20) -> ProfilingStats:
    """Parse profiling report for top functions."""
    stats = ProfilingStats(top_functions=[])

    content = profile_path.read_text()

    # Skip header lines until we find the column headers
    lines = content.split('\n')
    data_start = 0
    for i, line in enumerate(lines):
        if 'ncalls' in line and 'tottime' in line:
            data_start = i + 1
            break

    # Parse function entries
    # Format: ncalls  tottime  percall  cumtime  percall filename:lineno(function)
    function_pattern = r'\s*(\d+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+(.+)'

    for line in lines[data_start:data_start + top_n]:
        match = re.match(function_pattern, line)
        if match:
            stats.top_functions.append({
                'ncalls': int(match.group(1)),
                'tottime': float(match.group(2)),
                'tottime_percall': float(match.group(3)),
                'cumtime': float(match.group(4)),
                'cumtime_percall': float(match.group(5)),
                'function': match.group(6).strip()
            })

    # Calculate total time from top functions
    if stats.top_functions:
        stats.total_time = sum(f['cumtime'] for f in stats.top_functions[:5])

    return stats


def calculate_latency_stats(latencies: List[float]) -> Dict[str, float]:
    """Calculate latency statistics."""
    if not latencies:
        return {'min': 0, 'median': 0, 'p95': 0, 'max': 0, 'mean': 0}

    sorted_lat = sorted(latencies)
    return {
        'min': sorted_lat[0],
        'median': statistics.median(sorted_lat),
        'p95': sorted_lat[int(len(sorted_lat) * 0.95)] if len(sorted_lat) > 1 else sorted_lat[0],
        'max': sorted_lat[-1],
        'mean': statistics.mean(sorted_lat)
    }


def determine_winner(sa_value: float, ag_value: float, lower_is_better: bool = True) -> str:
    """Determine winner based on metric comparison."""
    if sa_value == ag_value:
        return "Tie"

    if lower_is_better:
        return "Service Account" if sa_value < ag_value else "AI Gateway"
    else:
        return "Service Account" if sa_value > ag_value else "AI Gateway"


def generate_markdown_report(
    sa_stats: ScanStats,
    ag_stats: ScanStats,
    sa_metrics: ImageMetrics,
    ag_metrics: ImageMetrics,
    sa_profiling: ProfilingStats,
    ag_profiling: ProfilingStats,
    output_path: Path
) -> None:
    """Generate human-readable markdown report."""

    sa_latency = calculate_latency_stats(sa_metrics.latencies_ms)
    ag_latency = calculate_latency_stats(ag_metrics.latencies_ms)

    sa_success_rate = (sa_metrics.success_count / sa_stats.images_processed * 100) if sa_stats.images_processed > 0 else 0
    ag_success_rate = (ag_metrics.success_count / ag_stats.images_processed * 100) if ag_stats.images_processed > 0 else 0

    report = f"""# Gemini 3 Pro Provider Comparison Report

## Executive Summary

| Metric | Service Account | AI Gateway | Winner |
|--------|----------------|------------|--------|
| Total Time | {sa_stats.elapsed_seconds:.1f}s | {ag_stats.elapsed_seconds:.1f}s | {determine_winner(sa_stats.elapsed_seconds, ag_stats.elapsed_seconds)} |
| Files Processed | {sa_stats.files_indexed}/{sa_stats.files_scanned} | {ag_stats.files_indexed}/{ag_stats.files_scanned} | {determine_winner(sa_stats.files_indexed, ag_stats.files_indexed, False)} |
| Images Processed | {sa_stats.images_processed} | {ag_stats.images_processed} | {determine_winner(sa_stats.images_processed, ag_stats.images_processed, False)} |
| Success Rate | {sa_success_rate:.1f}% | {ag_success_rate:.1f}% | {determine_winner(sa_success_rate, ag_success_rate, False)} |
| Median Latency | {sa_latency['median']:.0f}ms | {ag_latency['median']:.0f}ms | {determine_winner(sa_latency['median'], ag_latency['median'])} |
| P95 Latency | {sa_latency['p95']:.0f}ms | {ag_latency['p95']:.0f}ms | {determine_winner(sa_latency['p95'], ag_latency['p95'])} |
| Failed Images | {sa_metrics.failure_count} | {ag_metrics.failure_count} | {determine_winner(sa_metrics.failure_count, ag_metrics.failure_count)} |
| Retry Attempts | {sa_metrics.retry_count} | {ag_metrics.retry_count} | {determine_winner(sa_metrics.retry_count, ag_metrics.retry_count)} |
| Circuit Breaker Trips | {sa_metrics.circuit_breaker_trips} | {ag_metrics.circuit_breaker_trips} | {determine_winner(sa_metrics.circuit_breaker_trips, ag_metrics.circuit_breaker_trips)} |

## Performance Analysis

### Service Account Provider (gemini-service-account)

**Strengths:**
- Direct Vertex AI access (no routing overhead)
- Median latency: {sa_latency['median']:.0f}ms
- Success rate: {sa_success_rate:.1f}%

**Weaknesses:**
- Failures: {sa_metrics.failure_count} images
- Retries: {sa_metrics.retry_count} attempts
- Circuit breaker trips: {sa_metrics.circuit_breaker_trips}

**Error Breakdown:**
"""

    for category, count in sorted(sa_metrics.errors_by_category.items()):
        report += f"- {category}: {count}\n"

    report += f"""
### AI Gateway Provider (aigateway)

**Strengths:**
- Enterprise routing and compliance
- Median latency: {ag_latency['median']:.0f}ms
- Success rate: {ag_success_rate:.1f}%

**Weaknesses:**
- Failures: {ag_metrics.failure_count} images
- Retries: {ag_metrics.retry_count} attempts
- Circuit breaker trips: {ag_metrics.circuit_breaker_trips}
- Additional routing overhead (~200-500ms typical)

**Error Breakdown:**
"""

    for category, count in sorted(ag_metrics.errors_by_category.items()):
        report += f"- {category}: {count}\n"

    report += f"""
## Detailed Metrics

### Latency Distribution

| Statistic | Service Account | AI Gateway | Difference |
|-----------|----------------|------------|------------|
| Min | {sa_latency['min']:.0f}ms | {ag_latency['min']:.0f}ms | {ag_latency['min'] - sa_latency['min']:.0f}ms |
| Median | {sa_latency['median']:.0f}ms | {ag_latency['median']:.0f}ms | {ag_latency['median'] - sa_latency['median']:.0f}ms |
| Mean | {sa_latency['mean']:.0f}ms | {ag_latency['mean']:.0f}ms | {ag_latency['mean'] - sa_latency['mean']:.0f}ms |
| P95 | {sa_latency['p95']:.0f}ms | {ag_latency['p95']:.0f}ms | {ag_latency['p95'] - sa_latency['p95']:.0f}ms |
| Max | {sa_latency['max']:.0f}ms | {ag_latency['max']:.0f}ms | {ag_latency['max'] - sa_latency['max']:.0f}ms |

### Scan Statistics

| Metric | Service Account | AI Gateway |
|--------|----------------|------------|
| Total Files | {sa_stats.files_scanned} | {ag_stats.files_scanned} |
| Indexed | {sa_stats.files_indexed} | {ag_stats.files_indexed} |
| Failed | {sa_stats.files_failed} | {ag_stats.files_failed} |
| Chunks Created | {sa_stats.chunks_created} | {ag_stats.chunks_created} |
| Embeddings | {sa_stats.embeddings_created} | {ag_stats.embeddings_created} |
| Total Time | {sa_stats.elapsed_seconds:.1f}s | {ag_stats.elapsed_seconds:.1f}s |

## Issues and Recommendations

"""

    # Generate actionable recommendations
    recommendations = []

    if sa_metrics.circuit_breaker_trips > 0 or ag_metrics.circuit_breaker_trips > 0:
        recommendations.append("⚠️ **Circuit breaker tripped** - Consider increasing circuit_threshold or investigating API stability")

    if sa_metrics.failure_count > ag_metrics.failure_count * 1.5:
        recommendations.append("⚠️ **Service Account has significantly more failures** - Check credentials and API quotas")
    elif ag_metrics.failure_count > sa_metrics.failure_count * 1.5:
        recommendations.append("⚠️ **AI Gateway has significantly more failures** - Check gateway configuration and routing")

    if ag_latency['median'] > sa_latency['median'] + 500:
        recommendations.append(f"📊 **AI Gateway overhead: {ag_latency['median'] - sa_latency['median']:.0f}ms** - Expected routing overhead, acceptable for enterprise compliance")

    if sa_success_rate < 95 or ag_success_rate < 95:
        recommendations.append(f"⚠️ **Low success rate detected** - Consider increasing max_retries or timeout values")

    if not recommendations:
        recommendations.append("✅ **Both providers performed well** - No critical issues detected")

    for rec in recommendations:
        report += f"{rec}\n\n"

    report += f"""
## Profiling Deep Dive

### Service Account - Top Functions by Cumulative Time

| Function | Calls | Total Time | Cumulative Time |
|----------|-------|------------|-----------------|
"""

    for func in sa_profiling.top_functions[:10]:
        report += f"| `{func['function'][:60]}...` | {func['ncalls']} | {func['tottime']:.3f}s | {func['cumtime']:.3f}s |\n"

    report += f"""
### AI Gateway - Top Functions by Cumulative Time

| Function | Calls | Total Time | Cumulative Time |
|----------|-------|------------|-----------------|
"""

    for func in ag_profiling.top_functions[:10]:
        report += f"| `{func['function'][:60]}...` | {func['ncalls']} | {func['tottime']:.3f}s | {func['cumtime']:.3f}s |\n"

    output_path.write_text(report)


def generate_json_report(
    sa_stats: ScanStats,
    ag_stats: ScanStats,
    sa_metrics: ImageMetrics,
    ag_metrics: ImageMetrics,
    sa_profiling: ProfilingStats,
    ag_profiling: ProfilingStats,
    output_path: Path,
    metadata: Dict[str, Any]
) -> None:
    """Generate machine-readable JSON report."""

    sa_latency = calculate_latency_stats(sa_metrics.latencies_ms)
    ag_latency = calculate_latency_stats(ag_metrics.latencies_ms)

    report = {
        'test_metadata': metadata,
        'summary': {
            'service_account': {
                'scan_stats': asdict(sa_stats),
                'success_rate': (sa_metrics.success_count / sa_stats.images_processed * 100) if sa_stats.images_processed > 0 else 0,
                'latency_stats': sa_latency
            },
            'aigateway': {
                'scan_stats': asdict(ag_stats),
                'success_rate': (ag_metrics.success_count / ag_stats.images_processed * 100) if ag_stats.images_processed > 0 else 0,
                'latency_stats': ag_latency
            }
        },
        'detailed_metrics': {
            'service_account': {
                'image_metrics': {
                    'success_count': sa_metrics.success_count,
                    'failure_count': sa_metrics.failure_count,
                    'retry_count': sa_metrics.retry_count,
                    'circuit_breaker_trips': sa_metrics.circuit_breaker_trips,
                    'errors_by_category': dict(sa_metrics.errors_by_category)
                }
            },
            'aigateway': {
                'image_metrics': {
                    'success_count': ag_metrics.success_count,
                    'failure_count': ag_metrics.failure_count,
                    'retry_count': ag_metrics.retry_count,
                    'circuit_breaker_trips': ag_metrics.circuit_breaker_trips,
                    'errors_by_category': dict(ag_metrics.errors_by_category)
                }
            }
        },
        'profiling': {
            'service_account': {
                'top_functions': sa_profiling.top_functions,
                'total_time': sa_profiling.total_time
            },
            'aigateway': {
                'top_functions': ag_profiling.top_functions,
                'total_time': ag_profiling.total_time
            }
        }
    }

    output_path.write_text(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description='Generate Gemini 3 Pro provider comparison report')
    parser.add_argument('--gemini-sa-log', required=True, help='Service account DEBUG log file')
    parser.add_argument('--aigateway-log', required=True, help='AI Gateway DEBUG log file')
    parser.add_argument('--gemini-sa-profile', required=True, help='Service account profiling report')
    parser.add_argument('--aigateway-profile', required=True, help='AI Gateway profiling report')
    parser.add_argument('--gemini-sa-stdout', required=True, help='Service account stdout capture')
    parser.add_argument('--aigateway-stdout', required=True, help='AI Gateway stdout capture')
    parser.add_argument('--output', required=True, help='Output markdown report path')
    parser.add_argument('--json-output', required=True, help='Output JSON report path')

    args = parser.parse_args()

    # Parse all inputs
    print("Parsing stdout statistics...")
    sa_stats = parse_stdout_stats(Path(args.gemini_sa_stdout))
    ag_stats = parse_stdout_stats(Path(args.aigateway_stdout))

    print("Parsing log metrics...")
    sa_metrics = parse_log_metrics(Path(args.gemini_sa_log), 'gemini-service-account')
    ag_metrics = parse_log_metrics(Path(args.aigateway_log), 'aigateway')

    print("Parsing profiling reports...")
    sa_profiling = parse_profiling(Path(args.gemini_sa_profile))
    ag_profiling = parse_profiling(Path(args.aigateway_profile))

    # Generate metadata
    metadata = {
        'model': 'gemini-3-pro-preview',
        'providers': ['gemini-service-account', 'aigateway'],
        'test_type': 'parallel_comparison'
    }

    # Generate reports
    print(f"Generating markdown report: {args.output}")
    generate_markdown_report(
        sa_stats, ag_stats,
        sa_metrics, ag_metrics,
        sa_profiling, ag_profiling,
        Path(args.output)
    )

    print(f"Generating JSON report: {args.json_output}")
    generate_json_report(
        sa_stats, ag_stats,
        sa_metrics, ag_metrics,
        sa_profiling, ag_profiling,
        Path(args.json_output),
        metadata
    )

    print("Report generation complete!")


if __name__ == '__main__':
    main()
