import logging
from pathlib import Path

import click
from data_loader import DataCollatorForLazyLanguageModeling, LazyLineByLineTextDataset
from torch.utils.tensorboard import SummaryWriter
from transformers import RobertaForMaskedLM, XLMTokenizer
from transformers import Trainer, TrainingArguments

logging.basicConfig(
    filename=f"language_model_domain_pretrain.log",
    filemode="a",
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    level=logging.DEBUG,
)

# Writer will output to ./runs/ directory by default
writer = SummaryWriter()


@click.command()
@click.option("--train_dataset_path", help="Path to the file with train texts, line-by-line texts.")
@click.option("--model_name", help="Model name to be saved")
def main(train_dataset_path: str, model_name: str):
    logging.info("Loading Roberta")
    tokenizer = XLMTokenizer.from_pretrained("allegro/herbert-klej-cased-tokenizer-v1")
    model = RobertaForMaskedLM.from_pretrained("allegro/herbert-klej-cased-v1")

    logging.info("Loading dataset collator")
    data_collator = DataCollatorForLazyLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=0.15
    )

    logging.info("Loading lazily dataset")
    dataset = LazyLineByLineTextDataset(file_path=train_dataset_path)

    logging.info("Setup training params")
    training_args = TrainingArguments(
        output_dir="models",
        overwrite_output_dir=True,
        num_train_epochs=10,
        per_device_train_batch_size=8,
        warmup_steps=100,
        save_steps=10_000,
        # save_total_limit=2,  # we will save all checkpoints for experimental purposes
    )

    logging.info("Setup training params")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
        tb_writer=writer,
    )

    logging.info("Start training")
    trainer.train()

    logging.info("Save model")
    trainer.save_model(f"models/f{model_name}-{Path(train_dataset_path).stem}")


if __name__ == "__main__":
    main()
