import torch
from datasets import load_dataset
from transformers import TrainingArguments, EarlyStoppingCallback

from model.protein_encoder.builder import build_protein_encoder
from model.task_model.pretrain_model import ProteinTextCLIPForPretrain
from model.text_encoder.builder import build_text_encoder
from trainer.pretrain_trainer import CLIPPretrainTrainer


class PretrainTask(object):
    def __init__(self, run_config):
        self.run_config = run_config
        self.protein_encoder = build_protein_encoder(run_config)
        self.text_encoder = build_text_encoder(run_config)
        self.task_model = self.build_task_model()
        self.dataset = self.build_dataset()
        self.train_args = self.build_train_args()
        self.trainer = self.build_trainer()

    def build_task_model(self):
        raise NotImplementedError()

    def build_dataset(self):
        raise NotImplementedError()

    def build_train_args(self):
        raise NotImplementedError()

    def build_trainer(self):
        raise NotImplementedError()

    def run(self):
        self.trainer.train()


class ProteinTextCLIPPretrainTask(PretrainTask):
    def __init__(self, run_config):
        super().__init__(run_config)

    def build_task_model(self):
        return ProteinTextCLIPForPretrain(self.run_config, self.protein_encoder, self.text_encoder)

    def build_dataset(self):
        def preprocess_function(examples):
            protein_tokenized_examples = self.protein_encoder.tokenizer(examples["seq"], truncation=True,
                                                                        padding="max_length",
                                                                        max_length=self.run_config.max_length)
            text_tokenized_examples = self.text_encoder.tokenizer(examples["text"], truncation=True,
                                                                  padding="max_length",
                                                                  max_length=512)
            return {
                'protein_input_ids': protein_tokenized_examples['input_ids'],
                'protein_attention_mask': protein_tokenized_examples['attention_mask'],
                'text_input_ids': text_tokenized_examples['input_ids'],
                'text_attention_mask': text_tokenized_examples['attention_mask'],
            }

        dataset = load_dataset("json", data_files={
            'train': f'{self.run_config.data_path}/{self.run_config.dataset}/train.json',
            'valid': f'{self.run_config.data_path}/{self.run_config.dataset}/valid.json',
        })
        dataset = dataset.map(preprocess_function, batched=True, num_proc=8)
        dataset.set_format(type='torch', columns=['protein_input_ids', 'protein_attention_mask',
                                                  'text_input_ids', 'text_attention_mask'])
        return dataset

    def build_train_args(self):
        return TrainingArguments(
            output_dir=self.run_config.output_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="step",
            per_device_train_batch_size=self.run_config.batch_size,
            per_device_eval_batch_size=self.run_config.batch_size,
            num_train_epochs=self.run_config.num_epochs,
            weight_decay=self.run_config.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model=self.run_config.metric_for_best_model,
            greater_is_better=False if self.run_config.metric_for_best_model in ["mae", "rmse", "loss"] else True,
            fp16=self.run_config.fp16,
            push_to_hub=False,
            report_to=["wandb"],
        )

    def build_trainer(self):
        def collate_fn(batch):
            return {
                'protein_input_ids': torch.stack([example['protein_input_ids'] for example in batch]),
                'protein_attention_mask': torch.stack([example['protein_attention_mask'] for example in batch]),
                'text_input_ids': torch.stack([example['text_input_ids'] for example in batch]),
                'text_attention_mask': torch.stack([example['text_attention_mask'] for example in batch]),
            }

        return CLIPPretrainTrainer(
            run_config=self.run_config,
            model=self.task_model,
            args=self.train_args,
            data_collator=collate_fn,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["valid"],
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )
