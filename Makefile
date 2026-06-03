.PHONY: smoke test evaluate report bootstrap

smoke:
	python scripts/run_smoke_test.py

test:
	pytest tests/

evaluate:
	python scripts/run_evaluation.py --pred examples/minimal_item/prediction_example.json --gt examples/minimal_item/number_ground_truth.json --out outputs/example_eval

report:
	python scripts/generate_report.py --metrics evaluation_results/metric_scores_sample --out outputs/reports

bootstrap:
	python scripts/run_bootstrap.py --metrics evaluation_results/metric_scores_sample --out outputs/bootstrap
