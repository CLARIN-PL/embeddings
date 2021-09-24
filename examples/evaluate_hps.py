from tempfile import TemporaryDirectory

from embeddings.pipeline.hps_pipeline import OptimizedFlairSequenceLabelingPipeline
from embeddings.pipeline.hugging_face_sequence_labeling import HuggingFaceSequenceLabelingPipeline


def main() -> None:
    hps_pipeline: OptimizedFlairSequenceLabelingPipeline = OptimizedFlairSequenceLabelingPipeline(
        dataset_name="clarin-pl/kpwr-ner",
        embedding_name="clarin-pl/roberta-polish-kgr10",
        input_column_name="tokens",
        target_column_name="ner",
    )
    df, metadata = hps_pipeline.run()
    output_path = TemporaryDirectory()
    metadata["output_path"] = output_path.name
    pipeline = HuggingFaceSequenceLabelingPipeline(**metadata)
    print(pipeline.run())
    output_path.cleanup()


main()
