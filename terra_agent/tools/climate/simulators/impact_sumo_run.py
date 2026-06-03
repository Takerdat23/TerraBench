"""
impact_sumo_run simulator tool.
"""

from .common import *  # noqa: F401,F403
from .official_backends import run_sumo_official, wants_official_backend

def impact_sumo_run(
    network_path: str,
    routes_path: str,
    disruptions_path: str,
    *,
    run_id: str = "run-sumo",
    instance_id: str = "instance-001",
    scenario_name: str = "baseline",
    scenario_description: str = "",
    assumptions: dict[str, Any] | None = None,
    seed: int = 47,
    model_version: str = "oracle:impact_sumo@1.0.0",
    container_image: str | None = "container:sumo-proxy@sha256:deterministic",
    threads: int = 1,
    cpu_cores: int = 2,
    memory_gb: float = 2.0,
    timeout_seconds: int = 900,
    output_dir: str = "./data/simulators",
    backend: str | None = None,
) -> ToolResponse:
    """
    Simulate mobility disruption metrics on SUMO network and route inputs.

    By default this uses the official SUMO executable backend. Pass
    backend="proxy" to use the deterministic local SUMO-style proxy.
    """

    tool_name = "impact_sumo_run"
    start_ts = time.perf_counter()
    run_context = _build_run_context(
        run_id=run_id,
        instance_id=instance_id,
        scenario_name=scenario_name,
        scenario_description=scenario_description,
        start_iso=None,
        end_iso=None,
        timestep="event",
        assumptions=assumptions,
        seed=seed,
        model_version=model_version,
        container_image=container_image,
        threads=threads,
        cpu_cores=cpu_cores,
        memory_gb=memory_gb,
        timeout_seconds=timeout_seconds,
        inputs=[
            (network_path, "SUMO network"),
            (routes_path, "SUMO routes"),
            (disruptions_path, "disruptions specification"),
        ],
    )
    warnings: list[str] = []
    if wants_official_backend(backend):
        return run_sumo_official(
            network_path=network_path,
            routes_path=routes_path,
            disruptions_path=disruptions_path,
            run_id=run_id,
            scenario_name=scenario_name,
            output_dir=output_dir,
            start_ts=start_ts,
            run_context=run_context,
            timeout_seconds=timeout_seconds,
            warnings=warnings,
        )
    try:
        network = _parse_sumo_network(network_path)
        if not network:
            return _error(tool_name, "No usable edges found in network.", start_ts=start_ts, run_context=run_context)
        trips = _parse_sumo_routes(routes_path)
        if not trips:
            return _error(tool_name, "No vehicles/trips found in routes input.", start_ts=start_ts, run_context=run_context)
        disruptions = _read_json(disruptions_path)
        if not isinstance(disruptions, dict):
            return _error(tool_name, "Disruptions input must be a JSON object.", start_ts=start_ts, run_context=run_context)
        closed_edges = set(str(edge) for edge in disruptions.get("edge_closures", []) if edge is not None)
        speed_factors: dict[str, float] = defaultdict(lambda: 1.0)
        for row in disruptions.get("speed_reductions", []):
            if not isinstance(row, dict):
                continue
            edge_id = str(row.get("edge_id") or "")
            if not edge_id:
                continue
            factor = _clamp(_as_float(row.get("speed_factor"), 1.0), 0.05, 1.0)
            speed_factors[edge_id] = min(speed_factors[edge_id], factor)
        for row in disruptions.get("capacity_drops", []):
            if not isinstance(row, dict):
                continue
            edge_id = str(row.get("edge_id") or "")
            if not edge_id:
                continue
            factor = _clamp(_as_float(row.get("capacity_factor"), 1.0), 0.05, 1.0)
            speed_factors[edge_id] = min(speed_factors[edge_id], factor)

        rng = random.Random(seed)
        trip_rows: list[dict[str, Any]] = []
        edge_usage = defaultdict(int)
        edge_delay = defaultdict(float)
        completed_times: list[float] = []
        total_delay = 0.0
        completed = 0
        cancelled = 0
        for trip in trips:
            vehicle_id = str(trip.get("vehicle_id"))
            edges = [edge for edge in trip.get("edges", []) if edge in network]
            if not edges:
                cancelled += 1
                trip_rows.append(
                    {
                        "vehicle_id": vehicle_id,
                        "status": "cancelled",
                        "base_travel_time_s": 0.0,
                        "travel_time_s": 0.0,
                        "delay_s": 0.0,
                        "reason": "no_valid_route_edges",
                    }
                )
                continue
            if any(edge in closed_edges for edge in edges):
                cancelled += 1
                trip_rows.append(
                    {
                        "vehicle_id": vehicle_id,
                        "status": "cancelled",
                        "base_travel_time_s": round(sum(network[e]["length_m"] / network[e]["speed_m_s"] for e in edges), 5),
                        "travel_time_s": None,
                        "delay_s": None,
                        "reason": "route_contains_closed_edge",
                    }
                )
                continue

            base_time = 0.0
            disrupted_time = 0.0
            for edge in edges:
                length = network[edge]["length_m"]
                speed = network[edge]["speed_m_s"]
                base_segment = length / max(0.1, speed)
                factor = _clamp(speed_factors[edge], 0.05, 1.0)
                disrupted_segment = length / max(0.1, speed * factor)
                base_time += base_segment
                disrupted_time += disrupted_segment
                edge_usage[edge] += 1
                edge_delay[edge] += max(0.0, disrupted_segment - base_segment)
            noise = rng.uniform(0.985, 1.015)
            disrupted_time *= noise
            delay = max(0.0, disrupted_time - base_time)
            total_delay += delay
            completed += 1
            completed_times.append(disrupted_time)
            trip_rows.append(
                {
                    "vehicle_id": vehicle_id,
                    "status": "completed",
                    "base_travel_time_s": round(base_time, 5),
                    "travel_time_s": round(disrupted_time, 5),
                    "delay_s": round(delay, 5),
                }
            )

        avg_travel = statistics.fmean(completed_times) if completed_times else 0.0
        p95_travel = _percentile(completed_times, 0.95) if completed_times else 0.0
        run_dir = _tool_run_dir(output_dir, tool_name, run_id, scenario_name)
        tripinfo_path = run_dir / "tripinfo.csv"
        edge_stats_path = run_dir / "edge_stats.csv"
        summary_path = run_dir / "sumo_summary.json"
        _write_csv(tripinfo_path, trip_rows)
        edge_rows = []
        for edge_id, usage in sorted(edge_usage.items()):
            edge_rows.append(
                {
                    "edge_id": edge_id,
                    "vehicle_count": usage,
                    "mean_delay_s": round(edge_delay[edge_id] / max(1, usage), 6),
                    "is_closed": int(edge_id in closed_edges),
                    "speed_factor": round(speed_factors[edge_id], 5),
                }
            )
        _write_csv(edge_stats_path, edge_rows)
        _write_json(
            summary_path,
            {
                "avg_travel_time_s": avg_travel,
                "p95_travel_time_s": p95_travel,
                "total_delay_s": total_delay,
                "throughput_vehicles": completed,
                "cancelled_vehicles": cancelled,
            },
        )
        outputs = [
            _artifact_record(tripinfo_path, "Trip-level travel time outcomes"),
            _artifact_record(edge_stats_path, "Edge-level usage and delay statistics"),
            _artifact_record(summary_path, "SUMO proxy aggregate metrics"),
        ]
        metrics = {
            "avg_travel_time_s": _metric(avg_travel, "s"),
            "p95_travel_time_s": _metric(p95_travel, "s"),
            "total_delay_s": _metric(total_delay, "s"),
            "throughput_vehicles": _metric(float(completed), "vehicles"),
        }
        derived = {
            "cancelled_vehicles": _metric(float(cancelled), "vehicles"),
            "completion_ratio": _metric(completed / max(1, len(trips)), "ratio"),
        }
        risk_anchor = 1.0 - derived["completion_ratio"]["value"]
        agentic_summary = {
            "headline": f"Network throughput is {completed}/{len(trips)} vehicles with total delay {total_delay:.1f}s.",
            "primary_signal": {"metric": "completion_ratio", **derived["completion_ratio"]},
            "risk_level": _risk_level(risk_anchor, low=0.1, high=0.25),
            "comparison_ready": True,
            "recommended_next_tools": ["hazard_overlay_osm", "impact_climada_run"],
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


__all__ = ["impact_sumo_run"]
