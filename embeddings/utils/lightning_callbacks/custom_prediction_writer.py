import os

import torch
from pytorch_lightning.callbacks import BasePredictionWriter


class CustomPredictionWriter(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of its respective rank
        torch.save(
            obj=torch.cat([x[0] for x in predictions]),
            f=os.path.join(self.output_dir, f"logits_{trainer.global_rank}.pt")
        )
        torch.save(
            obj=torch.cat([x[1] for x in predictions]),
            f=os.path.join(self.output_dir, f"predictions_{trainer.global_rank}.pt")
        )
        # optionally, you can also save `batch_indices` to get the information about the data index
        # from your prediction data
        batch_indices = [y for x in batch_indices for y in x]
        torch.save(
            batch_indices, os.path.join(self.output_dir, f"batch_indices_{trainer.global_rank}.pt")
        )
