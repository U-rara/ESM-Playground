import hydra
import transformers
import wandb
from omegaconf import OmegaConf

transformers.logging.set_verbosity_info()
downstream_tasks = __import__('task.downstream_task', fromlist='*')
pretrain_tasks = __import__('task.pretrain_task', fromlist='*')


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(run_config):
    print(OmegaConf.to_yaml(run_config))
    if run_config.task.endswith("PretrainTask"):
        task = getattr(pretrain_tasks, run_config.task)(run_config)
    else:
        task = getattr(downstream_tasks, run_config.task)(run_config)
    task.run()



if __name__ == '__main__':
    main()
