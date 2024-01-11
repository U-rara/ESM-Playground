from transformers import Trainer


class CLIPPretrainTrainer(Trainer):
    def __init__(self, run_config, **kwargs):
        super().__init__(**kwargs)
        self.run_config = run_config

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        decay_parameters = self.get_decay_parameter_names(self.model)
        if self.run_config.protein_model_fixed and self.run_config.text_model_fixed:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            ratio_parameters = []

            if not self.run_config.protein_model_fixed:
                ratio_parameters += [n for n, p in self.model.protein_encoder.named_parameters()]
            if not self.run_config.text_model_fixed:
                ratio_parameters += [n for n, p in self.model.text_encoder.named_parameters()]

            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if
                        (n in decay_parameters and n in ratio_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.run_config.lr_ratio * self.run_config.lr
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if
                        (n not in decay_parameters and n in ratio_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.run_config.lr_ratio * self.run_config.lr
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if
                        (n in decay_parameters and n not in ratio_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters() if
                        (n not in decay_parameters and n not in ratio_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                }
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)
        return self.optimizer