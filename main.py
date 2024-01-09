import hydra
import transformers
import wandb
from omegaconf import OmegaConf

transformers.logging.set_verbosity_info()
tasks = __import__('task.downstream_task', fromlist='*')


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(config):
    print(OmegaConf.to_yaml(config))
    wandb.config = OmegaConf.to_container(
        config, resolve=True, throw_on_missing=True
    )
    task = getattr(tasks, config.task)(config)
    task.run()


if __name__ == '__main__':
    main()
