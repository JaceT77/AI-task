import subprocess
import sys
import structlog

log = structlog.get_logger(__name__)

pipeline_scripts = [
    "./src/t01_clean_data.py",
    "./src/t02_feature_engineering.py",
    "./src/t03_train_model.py",
    "./src/t05_generate_submissions.py",
    "./src/t07_summary_report.py",
    "./src/t08_generate_pdf_report.py",
]


def execute_pipeline() -> None:
    log.info(event="starting_freight_ml_pipeline")

    for script in pipeline_scripts:
        log.info(event="executing_step", script=script)
        result = subprocess.run([sys.executable, script])

        if result.returncode != 0:
            log.error(event="pipeline_failed", script=script)
            sys.exit(1)

    log.info(event="executing_final_validation_scorer")
    subprocess.run(
        [
            sys.executable,
            "./src/score.py",
            "--predictions",
            "./src/data/validation-predictions.csv",
            "--december-predictions",
            "./src/data/december-chart-inputs.csv",
        ]
    )

    log.info(event="pipeline_complete")


if __name__ == "__main__":
    execute_pipeline()
