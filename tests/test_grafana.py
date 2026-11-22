import json
import os
import yaml
import pytest


def test_grafana_datasource_provisioning():
    """Verify Grafana PostgreSQL datasource provisioning config."""
    path = os.path.join(
        "infrastructure", "grafana", "provisioning", "datasources", "postgres.yml"
    )
    assert os.path.exists(path), f"Datasource config not found at {path}"

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["apiVersion"] == 1
    assert "datasources" in data
    assert len(data["datasources"]) > 0

    ds = data["datasources"][0]
    assert ds["name"] == "PostgreSQL-CrossVerify"
    assert ds["type"] == "postgres"
    assert ds["url"] == "postgres:5432"
    assert ds["database"] == "crossverify"
    assert ds["user"] == "postgres"
    assert ds["jsonData"]["sslmode"] == "disable"


def test_grafana_dashboard_provider_provisioning():
    """Verify Grafana dashboard provider provisioning config."""
    path = os.path.join(
        "infrastructure", "grafana", "provisioning", "dashboards", "dashboards.yml"
    )
    assert os.path.exists(path), f"Provider config not found at {path}"

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    assert data["apiVersion"] == 1
    assert "providers" in data
    assert len(data["providers"]) > 0

    provider = data["providers"][0]
    assert provider["name"] == "CrossVerify Dashboards"
    assert provider["options"]["path"] == "/var/lib/grafana/dashboards"


def test_grafana_dashboard_json_schema():
    """Verify structure, panels, metrics, and thresholds of the telemetry dashboard JSON."""
    path = os.path.join(
        "infrastructure", "grafana", "dashboards", "crossverify_telemetry.json"
    )
    assert os.path.exists(path), f"Dashboard JSON not found at {path}"

    with open(path, "r", encoding="utf-8") as f:
        dashboard = json.load(f)

    # Core attributes
    assert dashboard["title"] == "CrossVerify: Dual-Path LLM Verification & Telemetry"
    assert dashboard["uid"] == "crossverify-telemetry-dash"
    assert "panels" in dashboard
    assert len(dashboard["panels"]) >= 7

    # Validate essential metric panels
    panel_titles = [p.get("title") for p in dashboard["panels"]]
    assert "Total Evaluations" in panel_titles
    assert "Rolling Drift Rate (15m)" in panel_titles
    assert "Candidate p95 Latency (15m)" in panel_titles
    assert "Candidate Error Rate (15m)" in panel_titles
    assert "JSON Schema Pass Rate" in panel_titles
    assert "Champion vs. Candidate Latency Distribution" in panel_titles
    assert "Semantic Drift Rate & Cosine Similarity Over Time" in panel_titles
    assert "Recent Evaluations & Discrepancy Audit Log" in panel_titles

    # Verify SQL query presence in key panels
    for panel in dashboard["panels"]:
        if panel.get("type") in ("stat", "timeseries", "table"):
            targets = panel.get("targets", [])
            assert len(targets) > 0, f"Panel '{panel.get('title')}' missing target query"
            assert "rawSql" in targets[0]
            assert "evaluations" in targets[0]["rawSql"].lower()


def test_docker_compose_grafana_service():
    """Verify docker-compose includes grafana service and configuration."""
    with open("docker-compose.yml", "r", encoding="utf-8") as f:
        dc = yaml.safe_load(f)

    assert "services" in dc
    assert "grafana" in dc["services"]

    grafana = dc["services"]["grafana"]
    assert "grafana/grafana" in grafana["image"]
    assert grafana["container_name"] == "crossverify-grafana"
    assert "crossverify_net" in grafana["networks"]
    assert "postgres" in grafana["depends_on"]
    assert "volumes" in dc
    assert "grafana_data" in dc["volumes"]

