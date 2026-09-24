from apps.studio.services.monitoring_service import list_monitoring_reports, monitoring_summary

def test_monitoring_summary_reads_drift_report(tmp_path):
    root = tmp_path / "monitoring"
    root.mkdir()
    (root / "drift.json").write_text('{"status":"drift","drift_score":0.79,"psi":16.6,"warnings":["feature_distribution_shift"]}')
    summary = monitoring_summary(tmp_path)
    assert summary["status"] == "drift"
    assert summary["report_count"] == 1
    assert list_monitoring_reports(tmp_path)[0]["drift_score"] == 0.79
