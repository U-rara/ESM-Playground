import torch
import numpy as np
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef
from transformers import TrainingArguments, EarlyStoppingCallback
from trainer import DownstreamTrainer

from metric import f1_max, area_under_prc, spearmanr
from model.protein_encoder.builder import build_protein_encoder
from model.task_model.downstream_model import EsmForSequenceClassification
from utils import EvaluateCallback


class DownstreamTask(object):
    def __init__(self, config):
        self.config = config
        self.num_labels = config.num_labels
        self.protein_encoder = build_protein_encoder(config)
        self.task_model = self.build_task_model()
        self.dataset = self.build_dataset()
        self.train_args = self.build_train_args()
        self.trainer = self.build_trainer()

    def build_task_model(self):
        raise NotImplementedError()

    def build_dataset(self):
        def preprocess_function(examples):
            tokenized_examples = self.protein_encoder.tokenizer(examples["seq"], truncation=True, padding=True,
                                                                max_length=self.config.max_length)
            tokenized_examples['label'] = torch.tensor(examples['label'])
            return tokenized_examples

        dataset = load_dataset("json", data_files={
            'train': f'{self.config.data_path}/{self.config.dataset}/train.json',
            'valid': f'{self.config.data_path}/{self.config.dataset}/valid.json',
            'test': f'{self.config.data_path}/{self.config.dataset}/test.json',
        })
        dataset = dataset.map(preprocess_function, batched=True)
        dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        return dataset

    def build_train_args(self):
        return TrainingArguments(
            output_dir=self.config.output_path,
            evaluation_strategy="epoch",
            save_strategy="epoch",
            logging_strategy="epoch",
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            load_best_model_at_end=True,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=False if self.config.metric_for_best_model in ['mae', 'rmse'] else True,
            fp16=self.config.fp16,
            push_to_hub=False,
            report_to=["wandb"],
        )

    def build_trainer(self):
        trainer = DownstreamTrainer(
            config=self.config,
            model=self.task_model,
            args=self.train_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["valid"],
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )
        trainer.add_callback(EvaluateCallback(trainer, self.dataset["test"]))
        return trainer

    def compute_metrics(self, eval_pred):
        raise NotImplementedError()

    def run(self):
        self.trainer.train()
        self.trainer.evaluate(self.dataset["test"], metric_key_prefix="test")


class SingleLabelSequenceClassificationTask(DownstreamTask):
    def __init__(self, config):
        super().__init__(config)

    def build_task_model(self):
        model_config = self.protein_encoder.model_config
        model_config.num_labels = self.num_labels
        model = self.protein_encoder.model
        task_model = EsmForSequenceClassification(model_config, model)
        return task_model

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        return {
            "accuracy": (predictions == labels).mean(),
            "matthews correlation coefficient": matthews_corrcoef(labels, predictions)
        }


class MultiLabelSequenceClassificationTask(SingleLabelSequenceClassificationTask):
    def __init__(self, config):
        super().__init__(config)

    def build_trainer(self):
        def collate_fn(examples):
            labels = torch.stack([example['label'] for example in examples])
            input_ids = torch.stack([example['input_ids'] for example in examples])
            attention_mask = torch.stack([example['attention_mask'] for example in examples])
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels
            }

        trainer = DownstreamTrainer(
            config=self.config,
            model=self.task_model,
            args=self.train_args,
            train_dataset=self.dataset["train"],
            eval_dataset=self.dataset["valid"],
            compute_metrics=self.compute_metrics,
            data_collator=collate_fn,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
        )
        trainer.add_callback(EvaluateCallback(trainer, self.dataset["test"]))
        return trainer

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        return {
            "f1_max": f1_max(torch.tensor(predictions), torch.tensor(labels)),
            "auprc_micro": area_under_prc(torch.tensor(predictions).flatten(), torch.tensor(labels).long().flatten())
        }


class SequenceRegressionTask(SingleLabelSequenceClassificationTask):
    def __init__(self, config):
        super().__init__(config)

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        predictions = torch.tensor(predictions).squeeze()
        labels = torch.tensor(labels).squeeze()
        return {
            "mae": (predictions - labels).abs().mean(),
            "rmse": ((predictions - labels) ** 2).mean().sqrt(),
            "spearman": spearmanr(predictions, labels)
        }
