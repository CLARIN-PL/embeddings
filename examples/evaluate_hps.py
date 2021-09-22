from embeddings.pipeline.hps_pipeline import OptimizedFlairSequenceLabelingPipeline
from embeddings.pipeline.hugging_face_sequence_labeling import HuggingFaceSequenceLabelingPipeline
from tempfile import TemporaryDirectory


def main() -> None:
    hps_pipeline = OptimizedFlairSequenceLabelingPipeline(
        dataset_name="clarin-pl/kpwr-ner",
        embedding_name="clarin-pl/roberta-polish-kgr10",
        input_column_name="tokens",
        target_column_name="ner",
    )
    trials_df, metadata = hps_pipeline.run()
    fine_tune_embeddings = metadata.pop("fine_tune_embeddings")
    output_path = TemporaryDirectory()
    pipeline = HuggingFaceSequenceLabelingPipeline(**metadata, output_path=output_path.name)
    print(pipeline.run())
    output_path.cleanup()


main()
