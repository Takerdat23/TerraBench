"""
impact_health_erf simulator tool.
"""

from .common import *  # noqa: F401,F403
from .official_backends import run_benmap_official, wants_official_backend

def impact_health_erf(
    exposure_series_path: str,
    population_exposed_path: str,
    erf_spec_path: str,
    *,
    run_id: str = "run-health-erf",
    instance_id: str = "instance-001",
    scenario_name: str = "baseline",
    scenario_description: str = "",
    assumptions: dict[str, Any] | None = None,
    seed: int = 41,
    model_version: str = "oracle:impact_health_erf@1.0.0",
    container_image: str | None = None,
    threads: int = 1,
    cpu_cores: int = 1,
    memory_gb: float = 1.0,
    timeout_seconds: int = 120,
    baseline_rate_per_100k: float = 200.0,
    baseline_cases_column: str | None = None,
    exposure_column: str = "exposure",
    output_dir: str = "./data/simulators",
    backend: str | None = None,
) -> ToolResponse:
    """
    Apply a health exposure-response workflow to estimate attributable burden.

    By default this delegates to a configured official BenMAP/health-impact
    command. Pass backend="proxy" to use the deterministic local ERF calculator.
    """

    tool_name = "impact_health_erf"
    start_ts = time.perf_counter()
    run_context = _build_run_context(
        run_id=run_id,
        instance_id=instance_id,
        scenario_name=scenario_name,
        scenario_description=scenario_description,
        start_iso=None,
        end_iso=None,
        timestep="daily",
        assumptions=assumptions,
        seed=seed,
        model_version=model_version,
        container_image=container_image,
        threads=threads,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        timeout_seconds=timeout_seconds,
        inputs=[
            (exposure_series_path, "exposure time series"),
            (population_exposed_path, "population exposed"),
            (erf_spec_path, "frozen ERF specification"),
        ],
    )
    warnings: list[str] = []
    if wants_official_backend(backend):
        return run_benmap_official(
            exposure_series_path=exposure_series_path,
            population_exposed_path=population_exposed_path,
            erf_spec_path=erf_spec_path,
            run_id=run_id,
            scenario_name=scenario_name,
            output_dir=output_dir,
            start_ts=start_ts,
            run_context=run_context,
            timeout_seconds=timeout_seconds,
            assumptions=assumptions,
            warnings=warnings,
        )
    try:
        exposure_rows = _load_rows(exposure_series_path)
        if not exposure_rows:
            return _error(tool_name, "Exposure time series input is empty.", start_ts=start_ts, run_context=run_context)
        population = _load_population(population_exposed_path)
        if population <= 0:
            warnings.append("Population input is missing or non-positive; attributable burden may be zero.")
        erf_spec = _read_json(erf_spec_path)
        if not isinstance(erf_spec, dict):
            return _error(tool_name, "ERF spec must be a JSON object.", start_ts=start_ts, run_context=run_context)

        beta = _as_float(erf_spec.get("beta"), 0.015)
        beta_ci = erf_spec.get("beta_ci") if isinstance(erf_spec.get("beta_ci"), list) else None
        ref_exposure = _as_float(erf_spec.get("reference_exposure"), 0.0)
        excess_only = bool(erf_spec.get("excess_only", True))
        lag_days = int(_as_float(erf_spec.get("lag_days"), 0))

        exposures = [_as_float(row.get(exposure_column) or row.get("exposure") or row.get("value"), 0.0) for row in exposure_rows]
        lagged_exposure: list[float] = []
        for idx in range(len(exposures)):
            start = max(0, idx - lag_days)
            window = exposures[start : idx + 1]
            lagged_exposure.append(statistics.fmean(window) if window else exposures[idx])

        if baseline_cases_column:
            baseline_series = [_as_float(row.get(baseline_cases_column), 0.0) for row in exposure_rows]
        else:
            baseline_total = baseline_rate_per_100k * max(0.0, population) / 100000.0
            baseline_each = baseline_total / max(1, len(exposure_rows))
            baseline_series = [baseline_each for _ in exposure_rows]

        output_rows: list[dict[str, Any]] = []
        attributable_cases = 0.0
        baseline_cases = 0.0
        af_values: list[float] = []
        attributable_low = 0.0
        attributable_high = 0.0
        beta_low = beta_high = None
        if beta_ci and len(beta_ci) >= 2:
            beta_low = _as_float(beta_ci[0], beta)
            beta_high = _as_float(beta_ci[1], beta)
            if beta_low > beta_high:
                beta_low, beta_high = beta_high, beta_low

        for idx, row in enumerate(exposure_rows):
            timestamp = _parse_datetime_from_row(row) or datetime(2000, 1, 1) + timedelta(days=idx)
            exposure_value = lagged_exposure[idx]
            baseline_case = max(0.0, baseline_series[idx])
            delta = exposure_value - ref_exposure
            if excess_only and delta <= 0:
                rr = 1.0
            else:
                rr = math.exp(beta * delta)
            af = 0.0 if rr <= 0 else _clamp((rr - 1.0) / rr, 0.0, 1.0)
            attributable = af * baseline_case
            attributable_cases += attributable
            baseline_cases += baseline_case
            af_values.append(af)
            row_payload = {
                "time": timestamp.isoformat(),
                "exposure": round(exposure_value, 5),
                "rr": round(rr, 6),
                "af": round(af, 6),
                "baseline_cases": round(baseline_case, 6),
                "attributable_cases": round(attributable, 6),
            }
            if beta_low is not None and beta_high is not None:
                rr_low = math.exp(beta_low * delta) if not (excess_only and delta <= 0) else 1.0
                rr_high = math.exp(beta_high * delta) if not (excess_only and delta <= 0) else 1.0
                af_low = 0.0 if rr_low <= 0 else _clamp((rr_low - 1.0) / rr_low, 0.0, 1.0)
                af_high = 0.0 if rr_high <= 0 else _clamp((rr_high - 1.0) / rr_high, 0.0, 1.0)
                attributable_low += af_low * baseline_case
                attributable_high += af_high * baseline_case
                row_payload["af_low"] = round(af_low, 6)
                row_payload["af_high"] = round(af_high, 6)
            output_rows.append(row_payload)

        attributable_fraction = attributable_cases / baseline_cases if baseline_cases > 0 else 0.0
        run_dir = _tool_run_dir(output_dir, tool_name, run_id, scenario_name)
        series_path = run_dir / "attributable_timeseries.csv"
        summary_path = run_dir / "burden_summary.json"
        _write_csv(series_path, output_rows)
        summary_payload: dict[str, Any] = {
            "attributable_fraction": attributable_fraction,
            "attributable_cases": attributable_cases,
            "baseline_cases": baseline_cases,
            "population_exposed": population,
            "beta": beta,
            "reference_exposure": ref_exposure,
            "lag_days": lag_days,
        }
        if beta_low is not None and beta_high is not None:
            summary_payload["attributable_cases_low"] = attributable_low
            summary_payload["attributable_cases_high"] = attributable_high
            summary_payload["beta_ci"] = [beta_low, beta_high]
        _write_json(summary_path, summary_payload)

        outputs = [
            _artifact_record(series_path, "Attributable burden time series"),
            _artifact_record(summary_path, "Burden summary from ERF"),
        ]
        metrics = {
            "attributable_fraction": _metric(attributable_fraction, "ratio"),
            "attributable_cases": _metric(attributable_cases, "cases"),
            "baseline_cases": _metric(baseline_cases, "cases"),
        }
        derived = {
            "af_mean": _metric(statistics.fmean(af_values) if af_values else 0.0, "ratio"),
        }
        if beta_low is not None and beta_high is not None:
            derived["attributable_cases_low"] = _metric(attributable_low, "cases")
            derived["attributable_cases_high"] = _metric(attributable_high, "cases")
        agentic_summary = {
            "headline": f"Estimated attributable burden: {metrics['attributable_cases']['value']:.2f} cases.",
            "primary_signal": {"metric": "attributable_cases", **metrics["attributable_cases"]},
            "risk_level": _risk_level(attributable_fraction, low=0.05, high=0.15),
            "comparison_ready": True,
            "recommended_next_tools": ["impact_health_utci", "impact_climada_run"],
        }
        return _success(
            tool_name=tool_name,
            start_ts=start_ts,
            metrics=metrics,
            derived_metrics=derived,
            output_artifacts=outputs,
            run_context=run_context,
            warnings=warnings,
            agentic_summary=agentic_summary,
        )
    except Exception as exc:
        return _error(tool_name, str(exc), start_ts=start_ts, run_context=run_context, warnings=warnings)


__all__ = ["impact_health_erf"]
