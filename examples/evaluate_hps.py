from embeddings.pipeline.hps_pipeline import OptimizedFlairSequenceLabelingPipeline


def main() -> None:
    pipeline = OptimizedFlairSequenceLabelingPipeline(
        dataset_name="clarin-pl/kpwr-ner",
        embedding_name="clarin-pl/roberta-polish-kgr10",
        input_column_name="tokens",
        target_column_name="ner",
    )
    print(pipeline.run())


main()
