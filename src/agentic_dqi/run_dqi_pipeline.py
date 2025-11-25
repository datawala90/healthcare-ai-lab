from src.agentic_dqi.agents import (
    load_dqi_config,
    ingress_agent,
    profiler_agent,
    anomaly_agent,
    reporter_agent,
)


def main():
    cfg = load_dqi_config()

    print("[Supervisor] Step 1: Load baseline & current batches...")
    dfs = ingress_agent(cfg)

    print("[Supervisor] Step 2: Profile drift...")
    metrics = profiler_agent(cfg, dfs)

    print("[Supervisor] Step 3: Ask LLM to explain anomalies...")
    analysis = anomaly_agent(cfg, metrics)

    print("[Supervisor] Step 4: Write reports...")
    reporter_agent(cfg, metrics, analysis)

    print("[Supervisor] All done. Check outputs/dqi/reports and outputs/dqi/json.")


if __name__ == "__main__":
    main()
