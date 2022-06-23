from dataclasses import dataclass
from pathlib import Path

from transformers.models.auto import AutoFeatureExtractor, AutoProcessor, AutoTokenizer
from transformers.onnx.convert import export, validate_model_outputs
from transformers.onnx.features import FeaturesManager
from transformers.onnx.utils import get_preprocessor

from embeddings.model.lightning_module.huggingface_module import HuggingFaceLightningModule
from embeddings.model.lightning_module.text_classification import TextClassificationModule
from embeddings.utils.loggers import get_logger

_logger = get_logger(__name__)


@dataclass
class OnnxExportConfiguration:
    output: Path
    model_name: str
    model: HuggingFaceLightningModule


class OnnxExporter:
    @staticmethod
    def get_features(config: OnnxExportConfiguration) -> str:
        feature = None
        if isinstance(config.model, TextClassificationModule):
            feature = "sequence-classification"
        if not feature:
            raise f"Unable to get features for {config}"

        _logger.info(f"Using {feature}")
        return feature

    def export(self, config: OnnxExportConfiguration) -> None:
        model_id = config.model_name
        preprocessor = AutoTokenizer.from_pretrained(model_id)

        features = OnnxExporter.get_features(config)
        # Allocate the model
        model = FeaturesManager.get_model_from_feature(features, model_id, framework="pt")
        model_kind, model_onnx_config = FeaturesManager.check_supported_model_or_raise(
            model, feature=features
        )
        onnx_config = model_onnx_config(model.config)

        # Ensure the requested opset is sufficient
        opset = onnx_config.default_onnx_opset

        output = config.output if config.output.is_file() else config.output.joinpath("model.onnx")
        onnx_inputs, onnx_outputs = export(
            preprocessor,
            model,
            onnx_config,
            opset,
            output,
        )
        atol = onnx_config.atol_for_validation

        validate_model_outputs(onnx_config, preprocessor, model, output, onnx_outputs, atol)
        _logger.info(f"All good, model saved at: {output.as_posix()}")


if __name__ == "__main__":
    main()
