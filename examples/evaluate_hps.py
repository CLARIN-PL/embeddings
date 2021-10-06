# Temporary script for Development Purposes
from tempfile import TemporaryDirectory

import flair
import torch

from embeddings.pipeline.hps_pipeline import (
    OptimizedFlairClassificationPipeline,
    OptimizedFlairSequenceLabelingPipeline,
)
from embeddings.pipeline.hugging_face_classification import HuggingFaceClassificationPipeline
from embeddings.pipeline.hugging_face_sequence_labeling import HuggingFaceSequenceLabelingPipeline


def main() -> None:
    flair.device = torch.device("cpu")  # TODO: remove after testing
    hps_pipeline = OptimizedFlairSequenceLabelingPipeline(
        dataset_name="clarin-pl/kpwr-ner",
        input_column_name="tokens",
        target_column_name="ner",
    ).persisting(best_params_path="best_params.yaml", log_path="hps_log.pickle")
    sl_df, metadata = hps_pipeline.run()
    output_path = TemporaryDirectory()
    metadata["output_path"] = output_path.name
    pipeline = HuggingFaceSequenceLabelingPipeline(**metadata)
    sl_results = pipeline.run()
    output_path.cleanup()

    tc_hps_pipeline = OptimizedFlairClassificationPipeline(
        dataset_name="clarin-pl/polemo2-official",
        input_column_name="text",
        target_column_name="target",
    ).persisting(best_params_path="best_prams_tc.yaml", log_path="tc_hps_log.pickle")
    tc_df, tc_metadata = tc_hps_pipeline.run()
    output_path = TemporaryDirectory()
    tc_metadata["output_path"] = output_path.name
    tc_pipeline = HuggingFaceClassificationPipeline(**tc_metadata)
    tc_results = tc_pipeline.run()
    output_path.cleanup()

    print(
        sl_df,
        "\n",
        sl_results,
        "\n\n",
        tc_df,
        "\n",
        tc_results,
    )


main()
