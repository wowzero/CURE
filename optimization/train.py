# the code base is adapted from open-reasoner-zero (orz) and openrlhf

import os
import re
import sys
import ast
import ray
import json
import copy
import torch
import asyncio
import argparse
import numpy as np
from typing import List
from loguru import logger
from jinja2 import Template
from dataclasses import dataclass
from collections import defaultdict
from functools import cached_property
from typing_extensions import override
from itertools import islice, zip_longest
from omegaconf.listconfig import ListConfig
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Awaitable, Callable, List, Optional, Tuple


from train_utils.dataset import PromptDataset
from train_utils.rl.trainer import RayPPOTrainer
from train_utils.base_exp import BasePPOExp, BasePPOExpConfig
from train_utils.utils import _validate_args
from ray.runtime_env import RuntimeEnv



import optimization_config



# our optimization scripts are in ./train_utils/rl/...

@dataclass
class PPOExpConfig(BasePPOExpConfig):
    use_compute_reward_fn: bool = True

    # Conditional settings with production values first
    total_num_nodes: int = optimization_config.total_num_nodes

    # resource related settings
    ref_num_nodes: int = total_num_nodes
    ref_num_gpus_per_node: int = 1
    actor_num_nodes: int = total_num_nodes
    actor_num_gpus_per_node: int = 1
    critic_num_nodes: int = total_num_nodes
    critic_num_gpus_per_node: int = 1
    colocate_all: bool = True
    colocate_critic_reward: bool = True
    colocate_actor_ref: bool = True
    vllm_num_engines: int = total_num_nodes
    vllm_tensor_parallel_size: int = 1
    adam_offload: bool = False
    zero_stage: int = 3

    # path related settings
    pretrain: Optional[str] = optimization_config.pretrained_model

    rl_data: Optional[str] = "temp_data/rl_data.json"
    separate_training: bool = optimization_config.separate_training
    rl_code_data: Optional[str] = "temp_data/rl_code_data.json"
    rl_case_data: Optional[str] = "temp_data/rl_case_data.json"

    os.makedirs(os.path.dirname("./ckpt"), exist_ok=True)
    ckpt_path: str = f"ckpt"
    save_path: str = f"ckpt"
    tensorboard_log_dir: str = f"ckpt"

    # ppo related settings
    actor_learning_rate: float = optimization_config.actor_learning_rate
    num_warmup_steps: int = optimization_config.num_warmup_steps
    prompt_max_len: int = optimization_config.prompt_max_len
    enable_prefix_caching: bool = True
    advantage_normalize: bool = False

    policy_update_steps: int = optimization_config.policy_update_steps
    micro_train_batch_size: int = 1
    micro_forward_batch_size: int = 1
    freezing_actor_steps: int = -1
    init_kl_coef: float = 0
    # 更换KL loss + k3
    kl_loss_coef: float = optimization_config.kl_loss_coef
    use_kl_loss: bool = optimization_config.use_kl_loss
    use_kl_estimator_k3: bool = optimization_config.use_kl_estimator_k3

    # generate related settings
    generate_max_len: int = optimization_config.generate_max_len
    packing_max_len: int = optimization_config.packing_max_len
    
    max_epochs: int = optimization_config.max_epochs # number of iteration of training

    gamma: float = 1.0
    lambd: float = 1.0

    optimized_model_name: str = optimization_config.optimized_model_name



class PPOExp(BasePPOExp):
    @cached_property
    def trainer(self):
        return RayPPOTrainer(
            cfg=self.cfg,
            strategy=self.strategy,
            tokenizer=self.tokenizer,
            colocate_pg=self.get_colocate_pg,
        )


async def _run_with_data(exp, rl_data):
    _validate_args(exp.cfg)
    ray.init(
        runtime_env=RuntimeEnv(
            env_vars={
                "NCCL_DEBUG": "WARN",
                "NCCL_PXN_DISABLE": "1",
                "NCCL_ALGO": "^Ring",
                "NCCL_NET_OVERHEAD": "1000000",
                "CUDA_LAUNCH_BLOCKING": "1",
            }
        )
    )

    await exp.trainer.build_models(exp.PolicyRayActor)
    await exp.trainer.train(rl_data)


def main(rl_data, pretrain=None):
    cfg = PPOExpConfig()
    if pretrain is not None:
        cfg.pretrain = pretrain

    exp = PPOExp().set_cfg(cfg)

    logger.info(exp.get_cfg_as_str(exp.cfg))
    if not os.path.exists(exp.cfg.save_path):
        os.makedirs(exp.cfg.save_path, exist_ok=True)
    if not os.path.exists(exp.cfg.tensorboard_log_dir):
        os.makedirs(exp.cfg.tensorboard_log_dir, exist_ok=True)
    if not os.path.exists(exp.cfg.ckpt_path):
        os.makedirs(exp.cfg.ckpt_path, exist_ok=True)
    asyncio.run(_run_with_data(exp, rl_data))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", type=str)
    parser.add_argument("--rl_data", type=str, default="temp_data/rl_data.json")
    parser.add_argument("--rl_code_data", type=str, default="temp_data/rl_code_data.json")
    parser.add_argument("--rl_case_data", type=str, default="temp_data/rl_case_data.json")
    args = parser.parse_args()

    if optimization_config.separate_training:
        with open(args.rl_code_data, 'r') as f:
            code_data = json.load(f)
        with open(args.rl_case_data, 'r') as f:
            case_data = json.load(f)
        rl_data = (code_data, case_data)
    else:
        with open(args.rl_data, 'r') as f:
            rl_data = json.load(f)

    main(rl_data, pretrain=args.pretrain)
