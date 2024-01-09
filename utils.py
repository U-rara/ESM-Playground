from transformers import TrainerCallback


class EvaluateCallback(TrainerCallback):

    def __init__(self, trainer, test_dataset):
        self.trainer = trainer
        self.test_dataset = test_dataset

    def on_epoch_end(self, args, state, control, **kwargs):
        metrics = self.trainer.evaluate(eval_dataset=self.test_dataset, metric_key_prefix="testing")
        print(metrics)
