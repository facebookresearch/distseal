# Copyright (c) Meta Platforms, Inc. and affiliates.

"""

Example usage:
    PYTHONPATH=deps python distill.py config=configs/training/rar/inmodel_latent_decoder.yaml imagenet.data_dir=/path/to/imagenet/train/split run_dir=/path/to/output
"""

from omegaconf import OmegaConf


def get_config():    
    cli_conf = OmegaConf.from_cli()
    if hasattr(cli_conf, "config"):
        yaml_conf = OmegaConf.load(cli_conf.config)
        
        # remove .config from cli_conf
        OmegaConf.set_struct(cli_conf, False)
        del cli_conf.config
        OmegaConf.set_struct(cli_conf, True) 
        
        conf = OmegaConf.merge(yaml_conf, cli_conf)
    
    return conf

def train_autoencoder(cfg):
    from deps.efficientvit.aecore.trainer import Trainer, TrainerConfig
    
    train_cfg = OmegaConf.structured(TrainerConfig)
    train_cfg = OmegaConf.merge(train_cfg, cfg)
    train_cfg = OmegaConf.to_object(train_cfg)
    trainer = Trainer(train_cfg)
    trainer.train()


def train_diffusion(cfg):
    from deps.efficientvit.diffusioncore.trainer import Trainer, TrainerConfig
    
    train_cfg = OmegaConf.structured(TrainerConfig)
    train_cfg = OmegaConf.merge(train_cfg, cfg)
    train_cfg = OmegaConf.to_object(train_cfg)
    trainer = Trainer(train_cfg)
    trainer.train()


def main():
    cfg = get_config()
    print(f"Configuration: {OmegaConf.to_yaml(cfg)}")
    
    # Distill autoregressive models
    if cfg.model.startswith("maskgit-vqgan") or cfg.model.startswith("dc-ae"):
        train_autoencoder(cfg)
    
    elif cfg.model.startswith("uvit"):
        train_diffusion(cfg)

    else:
        raise NotImplementedError(f"Unsupported model: {cfg.model}")


if __name__ == "__main__":
    main()
