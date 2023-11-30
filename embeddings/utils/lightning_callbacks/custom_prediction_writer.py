import os
import pickle

from pytorch_lightning.callbacks import BasePredictionWriter


class CustomPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of its respective rank
        predpath = os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pkl")
        with open(predpath, "wb") as f:
            pickle.dump(predictions, f)

        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        idxpath = os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pkl")
        with open(idxpath, "wb") as f:
            pickle.dump(batch_indices, f)
