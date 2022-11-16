import gc

import torch


def cleanup_torch_model_artifacts() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
