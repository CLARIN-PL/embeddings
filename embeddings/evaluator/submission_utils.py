from typing import Generator, Iterable

from wandb.apis.public import Run


def filter_hps_summary_runs(runs: Iterable[Run]) -> Iterable[Run]:
    for run in runs:
        # type ignore due to untyped logged_artifacts() function
        for artifact in run.logged_artifacts():  # type: ignore[no-untyped-call]
            if artifact.name.split(":")[0] == "hps_result":
                yield run


def filter_retrains(runs: Iterable[Run]) -> Generator[Run, None, None]:
    for run in runs:
        if (
            len(run.summary.keys()) > 1
            and run.config["predict_subset"] == "LightingDataModuleSubset.TEST"
        ):
            yield run
