import argparse
import sys
from pathlib import Path
import os

import torch
from pytorch_lightning import Trainer

import datasets
import systems
import models
from utils.misc import load_config
from utils.misc import config_to_primitive, get_rank


def load_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to config file')
    parser.add_argument('--gpu', default='0', help='GPU(s) to be used')
    parser.add_argument('--resume', default=None, help='path to the weights to be resumed')

    args, extras = parser.parse_known_args()

    # os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    config = load_config(args.config, cli_args=extras)
    config.cmd_args = vars(args)

    return config, args, extras


if __name__ == '__main__':
    config, args, extras = load_cfg()
    n_gpus = len(args.gpu.split(','))
    rank = get_rank()

    dm = datasets.make(config.dataset.name, config.dataset)
    dm.setup('validate')
    dataloader = dm.val_dataloader()
    dataset = dataloader.dataset

    state_dict = torch.load(args.resume)
    state_dict = {k[6:]: v for k, v in state_dict['state_dict'].items()}

    model = models.make(config.model.name, config.model)
    model.training = False
    model.load_state_dict(state_dict)
    model = model.to(device=rank)
    model.update_step(0, 0)
    model.background_color = torch.ones((3,), dtype=torch.float32, device=rank)

    system = systems.make(config.system.name, config, load_from_checkpoint=args.resume)
    system.dataset = dataset

    # callbacks = []
    # loggers = []
    # strategy = 'ddp_find_unused_parameters_false'
    # trainer = Trainer(
    #     devices=n_gpus,
    #     accelerator='gpu',
    #     callbacks=callbacks,
    #     logger=loggers,
    #     strategy=strategy,
    #     **config.trainer
    # )
    # trainer.strategy._lightning_module = _maybe_unwrap_optimized(system)
    # trainer._run(system, args.resume)

    batch = next(iter(dataloader))
    system.preprocess_data(batch, 'validate')
    out = model(batch['rays'].cuda())

    # fix ray_indices (based on ray chunking)
    ray_indices = out['ray_indices']
    ray_offsets = [
        torch.ones(
            (out['num_samples'][i].item(),),
            dtype=ray_indices.dtype, device=ray_indices.device
        ) * num_samples
        for i, num_samples in
            enumerate(range(0, batch['rays'].shape[0], config.model.ray_chunk))
    ]
    ray_offsets = torch.cat(ray_offsets, dim=0)
    ray_indices = ray_indices - ray_offsets

    print({
        k: v.shape if torch.is_tensor(v) else v
        for k, v in out.items()
    })

