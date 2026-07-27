"""
Experiment logger abstraction for Comet ML and W&B.

Usage:
    logger = ExperimentLogger(backend="comet", project="d2l-sft", config={...})
    logger.log({"train/loss": 0.5, "train/lr": 1e-5}, step=100)
    logger.finish()
"""

import logging

log = logging.getLogger(__name__)


class ExperimentLogger:
    """Unified logger supporting comet, wandb, or none."""

    def __init__(self, backend: str, project: str, config: dict | None = None):
        self.backend = backend
        self._experiment = None

        if backend == "comet":
            import comet_ml
            self._experiment = comet_ml.Experiment(
                project_name=project,
                auto_metric_logging=False,
                auto_param_logging=False,
            )
            if config:
                self._experiment.log_parameters(config)
            log.info(f"Comet ML experiment: {self._experiment.get_key()}")

        elif backend == "wandb":
            import wandb
            wandb.init(project=project, config=config)
            self._experiment = wandb
            log.info(f"W&B run: {wandb.run.name}")

        elif backend == "none":
            log.info("Logging disabled")

        else:
            raise ValueError(f"Unknown logger backend: {backend}")

    def log(self, metrics: dict, step: int):
        if self.backend == "comet":
            for key, value in metrics.items():
                self._experiment.log_metric(key, value, step=step)
        elif self.backend == "wandb":
            self._experiment.log(metrics, step=step)

    def log_text(self, text: str, step: int, metadata: dict | None = None):
        """Log text for debugging (e.g. model generations)."""
        if self.backend == "comet":
            self._experiment.log_text(text, step=step, metadata=metadata)
        elif self.backend == "wandb":
            import wandb
            self._experiment.log({"text": wandb.Html(f"<pre>{text}</pre>")}, step=step)

    def finish(self):
        if self.backend == "comet":
            self._experiment.end()
        elif self.backend == "wandb":
            self._experiment.finish()
