import asyncio
import json
import math
import os
import random
from functools import partial
from heapq import heapify, heappop, heappush
from typing import Any, Awaitable, Callable, List, Optional, Tuple, Union

import ray
import torch
from loguru import logger
from omegaconf.dictconfig import DictConfig
from ray.util.placement_group import PlacementGroup, placement_group
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from train_utils.rl.actors import PPORayActorGroup
from train_utils.rl.replay_buffer import Experience, NaiveReplayBuffer
from train_utils.utils import ORZDeepspeedStrategy as DeepspeedStrategy
from train_utils.utils import (
    Timer,
    compute_approx_kl,
    compute_reward,
    get_advantages_and_returns,
    masked_mean,
    normalize_advantages,
)


class RayPPOTrainer:
    def __init__(
        self,
        cfg: DictConfig,
        strategy: DeepspeedStrategy,
        tokenizer,
        colocate_pg: Optional[PlacementGroup] = None,
    ):
        self.cfg = cfg
        self.strategy = strategy
        self.tokenizer = tokenizer
        self.colocate_pg = colocate_pg

        self.writer = SummaryWriter(log_dir=self.cfg.tensorboard_log_dir)
        self.replay_buffer = NaiveReplayBuffer(
            sample_batch_size=self.cfg.micro_train_batch_size,
            limit=0,
            cpu_offload=True,
            packing_samples=True,
        )

    def __del__(self):
        self.writer.close()

    async def eval(self):
        raise NotImplementedError("Eval function should be implemented in user's exp")

    async def train(self, rl_data):

        if self.cfg.separate_training:
            code_data, case_data = rl_data
            train_data_list = [code_data, case_data]
        else:
            train_data_list = [rl_data]

        for dataset in train_data_list:
            logger.info("Training with provided dataset")

            await self.make_experience(dataset)

            num_policy_dp_nodes = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node
            num_critic_dp_nodes = self.cfg.critic_num_nodes * self.cfg.critic_num_gpus_per_node
            policy_buffers = self.replay_buffer.split_to_n_batches(num_policy_dp_nodes)

            async with Timer("Actor model training"):
                await self.policy_model.backload_to_gpu()
                await self.ppo_local_train_policy(policy_buffers, 1)
                await self.policy_model.offload_to_cpu()
            
            self.replay_buffer.clear()

        await self.policy_model.async_save_model(self.tokenizer, 1)
        logger.info("Successfully save model weights, training finish.")

        

    @torch.no_grad()
    async def make_experience(self, rl_data, **generate_kwargs):
        experiences = []

        import json

        prompts_set = []
        response_set = []
        rewards_set = []
        
        for i in range(len(rl_data)):
            prompts_set.append(rl_data[i]["prompt"])
            response_set.append(rl_data[i]["response"])
            rewards_set.append(rl_data[i]["reward"])

        if self.cfg.use_compute_reward_fn:
            async with Timer("Calculate custom rewards"):
                dp_tasks = []
                reward_fn = partial(self.custom_reward_fn, reward_model_fn=self._warp_custom_reward_model_fn())
                all_prompts, outputs, custom_rewards = await reward_fn(prompts_set, response_set, rewards_set)
                assert len(all_prompts) == len(
                    outputs
                ), "generate objects number after custom reward function must be equal to all inputs number"
        else:
            all_prompts, outputs, custom_rewards = all_prompts, outputs, None
        
        # 1.3 packing samples
        async with Timer("Packing samples"):
            (
                ret_sequences,
                ret_attention_masks,
                ret_num_actions,
                ret_packed_seq_lens,
                ret_custom_rewards,
            ) = self._convert_prompts_outputs_to_batch_tensors_packing(
                all_prompts, outputs, custom_rewards, self.cfg.packing_max_len
            )
            action_masks = None

        # 1.4 inference and calculate values, log probs, rewards, kl divergence
        async with Timer("Inference and calculate values, log probs, rewards, kl divergence"):
            experiences = await self.inference_and_calculates(
                ret_sequences,
                ret_attention_masks,
                action_masks,
                ret_num_actions,
                ret_packed_seq_lens,
                ret_custom_rewards,
            )
            logger.info(f"experiences size: {len(experiences)}")

        # 3. calculate advantages and returns / along with tensorboard logging

        async with Timer("Calculate advantages and returns"):
            adv_tasks = []
            for experience in experiences:
                adv_tasks.append(self._calc_advantages_and_returns(experience))

            for tsk in asyncio.as_completed(adv_tasks):
                experience, metrics = await tsk
                self.replay_buffer.append(experience)


    @torch.no_grad()
    async def inference_and_calculates(
        self,
        sequences_all: List[torch.Tensor],
        attention_mask_all: List[torch.Tensor],
        action_mask_all: Optional[List[torch.Tensor]],
        num_actions_all: Optional[List[int]],
        packed_seq_lens_all: Optional[List[int]],
        custom_rewards_all: Optional[List[torch.Tensor]],
    ):
        num_policy_dp_groups = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node
        num_critic_dp_groups = self.cfg.critic_num_nodes * self.cfg.critic_num_gpus_per_node
        num_ref_dp_groups = self.cfg.ref_num_nodes * self.cfg.ref_num_gpus_per_node
        num_reward_dp_groups = self.cfg.reward_num_nodes * self.cfg.reward_num_gpus_per_node

        async def micro_infer_model(num_dps, model_type, sequences, num_actions, attention_mask, packed_seq_lens):
            dp_iterator = self._split_dp_batch(
                (sequences, num_actions, attention_mask, packed_seq_lens),
                num_dps,
            )
            dp_tasks = []
            for dp_rank, (
                micro_sequences,
                micro_num_actions,
                micro_attention_mask,
                micro_packed_seq_lens,
            ) in enumerate(dp_iterator):
                model = self._get_dp_group_models(dp_rank, model_type)

                async def forward_fn(
                    local_model, fwd_sequences, fwd_num_actions, fwd_attention_mask, fwd_packed_seq_lens
                ):
                    return await local_model.forward.remote(
                        sequences=fwd_sequences,
                        num_actions=fwd_num_actions,
                        attention_mask=fwd_attention_mask,
                        packed_seq_lens=fwd_packed_seq_lens,
                    )

                dp_tasks.append(
                    self._split_and_run_micro_batch(
                        partial(forward_fn, model),
                        (micro_sequences, micro_num_actions, micro_attention_mask, micro_packed_seq_lens),
                        self.cfg.micro_forward_batch_size,
                    )
                )
            results = await asyncio.gather(*dp_tasks)
            results = sum(results, [])
            return results

        if action_mask_all is not None:
            num_actions_all = action_mask_all.size(1)

        # calculate ref log probs
        if self.cfg.colocate_all:
            await self.policy_model.backload_to_gpu()

        
        base_action_log_probs_ref = micro_infer_model(
            num_policy_dp_groups, "policy_model", sequences_all, num_actions_all, attention_mask_all, packed_seq_lens_all
        )
        results = await asyncio.gather(base_action_log_probs_ref)
        base_log_probs = results[0]
        

        if self.cfg.colocate_all:
            await self.policy_model.offload_to_cpu()
        else:
            empty_cache_tasks = [
                self.policy_model.async_run_method("empty_cache")
            ]
            await asyncio.gather(*empty_cache_tasks)

        # 6. calculate kl divergence

        experiences = []
        base_log_probs = base_log_probs[: len(sequences_all)]
        for i in range(len(base_log_probs)):
            response_length = torch.Tensor(num_actions_all[i]).unsqueeze(0)
            total_length = torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0)
            kl = base_log_probs[i] - base_log_probs[i]
            info = {
                "reward": None,
                "custom_rewards": custom_rewards_all[i] if custom_rewards_all is not None else None,
                "response_length": response_length,
                "total_length": total_length,
                "num_actions": num_actions_all[i],
            }
            experiences.append(
                Experience(
                    sequences_all[i],
                    base_log_probs[i],
                    None,
                    None,
                    None,
                    attention_mask_all[i],
                    None,
                    response_length,
                    torch.Tensor(packed_seq_lens_all[i]).unsqueeze(0),
                    info,
                    kl,
                )
            )
        return experiences


    async def build_models(self, PolicyRayActor):
        cfg = self.cfg
        pg = None

        if cfg.colocate_all:
            assert (
                cfg.actor_num_nodes == cfg.critic_num_nodes
                and cfg.actor_num_gpus_per_node == cfg.critic_num_gpus_per_node
                and cfg.actor_num_nodes == cfg.ref_num_nodes
                and cfg.actor_num_gpus_per_node == cfg.ref_num_gpus_per_node
                and cfg.actor_num_gpus_per_node == 1
                and cfg.actor_num_nodes == cfg.vllm_num_engines
            ), "num_nodes and num_gpus_per_node must be the same when colocate all models and each actor has only one gpu."
            pg = self.colocate_pg

            policy_model = PPORayActorGroup(
                cfg.actor_num_nodes,
                cfg.actor_num_gpus_per_node,
                PolicyRayActor,
                pg=pg,
                num_gpus_per_actor=0.2,
            )
            
            critic_model = None
            reward_models = None

        else:
            if cfg.colocate_actor_ref:
                assert (
                    cfg.actor_num_nodes == cfg.ref_num_nodes
                    and cfg.actor_num_gpus_per_node == cfg.ref_num_gpus_per_node
                ), "num_nodes and num_gpus_per_node must be the same when colocate actor and ref model."

                bundles = [
                    {"GPU": cfg.actor_num_gpus_per_node, "CPU": cfg.actor_num_gpus_per_node}
                    for _ in range(cfg.actor_num_nodes)
                ]
                pg = placement_group(bundles, strategy="PACK")
                ray.get(pg.ready())

            policy_model = PPORayActorGroup(
                cfg.actor_num_nodes,
                cfg.actor_num_gpus_per_node,
                PolicyRayActor,
                pg=pg,
                num_gpus_per_actor=0.75 if pg else 1,
            )

            critic_model = None
            reward_models = None

        if not cfg.colocate_all:
            refs = []
            refs.extend(policy_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            await asyncio.gather(*refs)
            await policy_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
        else:
            #await asyncio.gather(*ref_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            await asyncio.gather(*policy_model.async_init_model_from_pretrained(self.strategy, cfg.pretrain))
            await policy_model.async_run_method("_set_pad_token_id", self.tokenizer.pad_token_id)
            await policy_model.offload_to_cpu()

        self.policy_model = policy_model
        self.critic_model = None
        self.reward_model = None

        logger.info("init policy model done")

    async def ppo_local_train_policy(self, replay_buffers: List[NaiveReplayBuffer], global_steps: int):
        if global_steps > self.cfg.freezing_actor_steps:
            async with Timer("Policy model training"):
                await self.policy_model.async_ppo_train(global_steps, replay_buffers)
            await self.policy_model.async_run_method("empty_cache")

        if global_steps > self.cfg.freezing_actor_steps:
            return 

    async def custom_reward_fn(
        self,
        prompts: List[str],
        outputs: List[Any],
        scores: List[Any],
        reward_model_fn: Callable[[List[str], List[str]], Awaitable[torch.Tensor]],
    ) -> Tuple[List[str], List[str], List[torch.Tensor]]:
        responses = outputs
        output_tokens = self._tokenize(responses, self.cfg.generate_max_len, padding=False)["input_ids"]

        # make a pre-token score tensor for each output, for example: [0, 0, 0, 0, r]
        score_tensors = []
        for score, output_token in zip(scores, output_tokens):
            score_tensor = torch.zeros(len(output_token))
            if len(output_token) > 0:
                score_tensor[-1] = score
            score_tensors.append(score_tensor)

        # rm empty response
        res_prompts = []
        res_responses = []
        res_score_tensors = []
        for prompt, response, score_tensor in zip(prompts, responses, score_tensors):
            if len(response) > 0:
                res_prompts.append(prompt)
                res_responses.append(response)
                res_score_tensors.append(score_tensor)

        return res_prompts, res_responses, res_score_tensors

    @torch.no_grad()
    async def _calc_advantages_and_returns(self, experience: Experience):
        num_actions = experience.info["num_actions"]
        reward = await compute_reward.remote(
            experience.info["reward"],
            self.cfg.init_kl_coef,
            experience.kl,
            custom_rewards=experience.info["custom_rewards"],
            action_mask=experience.action_mask,
            num_actions=num_actions,
            reward_clip_range=self.cfg.reward_clip_range,
            use_kl_loss=self.cfg.use_kl_loss,
        )
        experience.advantages, experience.returns = await get_advantages_and_returns.remote(
            experience.values,
            reward,
            experience.action_mask,
            num_actions,
            self.cfg.gamma,
            self.cfg.lambd,
            packing=True,
        )
        return_sums = reward.sum(dim=-1)
        return_sums /= len(num_actions)
        experience.info["return"] = return_sums
        experience.kl = None

        del experience.info["num_actions"]
        del experience.info["custom_rewards"]
        del experience.info["reward"]
        experience.to_device("cpu")

        # for replay buffer split batch
        num_packed_samples = len(num_actions)
        return_sums /= num_packed_samples
        experience.info["response_length"] = torch.Tensor(experience.info["response_length"]).mean().unsqueeze(0)
        experience.info["total_length"] = torch.Tensor(experience.info["total_length"]).mean().unsqueeze(0)

        metrics = {
        }

        return experience, metrics

    def _convert_prompts_outputs_to_batch_tensors(self, prompts: List[str], outputs: List[str]):
        # This function is used when not packing samples
        # concat all outputs to following format:
        #
        # | [PAD] [PAD] token token token | token token [EOS] [PAD] |
        # | token token token token token | token token [EOS] [PAD] |
        # | [PAD] [PAD] [PAD] token token | token token token [EOS] |
        # |<---------- prompt ----------->|<-------- answer ------->|
        max_input_len, max_output_len = 0, 0
        prompt_token_lens, response_token_lens = [], []
        inputs_token_ids, outputs_token_ids = [], []
        for prompt, output in zip(prompts, outputs):
            input_token_ids = self._tokenize(prompt, self.cfg.prompt_max_len, padding=False)["input_ids"]
            response_token_ids = self._tokenize(output, self.cfg.generate_max_len, padding=False)["input_ids"]

            inputs_token_ids.append(input_token_ids)
            outputs_token_ids.append(response_token_ids)

            prompt_token_len = len(input_token_ids)
            response_token_len = len(response_token_ids)
            prompt_token_lens.append(prompt_token_len)
            response_token_lens.append(response_token_len)

            max_input_len = max(max_input_len, prompt_token_len)
            max_output_len = max(max_output_len, response_token_len)

        pad_token_id, eos_token_id = self.tokenizer.pad_token_id, self.tokenizer.eos_token_id
        sequences = []
        for i, prompt in enumerate(prompts):
            # left padding input
            input_len = prompt_token_lens[i]
            input_ids = [pad_token_id] * (max_input_len - input_len) + list(inputs_token_ids[i])

            # right padding output
            output_len = response_token_lens[i]
            output_ids = list(outputs_token_ids[i]) + [pad_token_id] * (max_output_len - output_len)

            # replace last token with eos_token_id if it is not eos_token_id, keep the total length of output_ids
            # output_ids[output_len - 1] = eos_token_id

            # concat input and output
            sequences.append(input_ids + output_ids)

        sequences = torch.tensor(sequences)

        sequences, attention_mask, action_mask = self._process_sequences(
            sequences, max_input_len, eos_token_id, pad_token_id
        )
        return sequences, attention_mask, action_mask

    def _convert_prompts_outputs_to_batch_tensors_packing(
        self, prompts: List[str], outputs: List[str], custom_rewards: Optional[List[torch.Tensor]], packing_max_len: int
    ):
        ret_sequences = []
        ret_attention_masks = []
        ret_num_actions = []
        ret_packed_seq_lens = []
        if custom_rewards is not None:
            ret_custom_rewards = []
        else:
            ret_custom_rewards = None

        assert (
            len(prompts) == len(outputs) and len(prompts) > 0
        ), "prompts and outputs must have the same length and length must be greater than 0"

        def _new_instance():
            out_sequence = torch.full((packing_max_len,), torch.tensor(self.tokenizer.pad_token_id), dtype=torch.long)
            out_attention_mask = torch.zeros((packing_max_len,), dtype=torch.int)
            out_num_actions = []
            out_packed_seq_lens = []
            rewards = [] if custom_rewards else None
            seq_offset = 0
            seq_index = 0
            return (
                out_sequence,
                out_attention_mask,
                out_num_actions,
                out_packed_seq_lens,
                rewards,
                seq_offset,
                seq_index,
            )

        def _accumulate(
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            seq_offset,
            seq_index,
            sequence,
            attention_mask,
            num_action,
            total_len,
            custom_rewards,
            i,
        ):
            out_sequence[seq_offset : seq_offset + total_len] = torch.tensor(sequence)
            out_attention_mask[seq_offset : seq_offset + total_len] = seq_index + 1
            out_num_actions.append(num_action)
            out_packed_seq_lens.append(total_len)
            if custom_rewards:
                rewards.append(custom_rewards[i])
            return seq_offset + total_len, seq_index + 1

        sequences = []
        attention_masks = []
        num_actions = []
        total_lens = []

        input_token_ids = self._tokenize(prompts, self.cfg.prompt_max_len, padding=False)["input_ids"]
        response_token_ids = self._tokenize(outputs, self.cfg.generate_max_len, padding=False)["input_ids"]

        for input_ids, response_ids in zip(input_token_ids, response_token_ids):
            sequences.append(input_ids + response_ids)
            attention_masks.append(torch.ones((len(input_ids) + len(response_ids),), dtype=torch.float32))
            num_actions.append(len(response_ids))
            total_lens.append(len(input_ids) + len(response_ids))

        # make packed sequences
        (
            out_sequence,
            out_attention_mask,
            out_num_actions,
            out_packed_seq_lens,
            rewards,
            seq_offset,
            seq_index,
        ) = _new_instance()
        for i, (sequence, attention_mask, num_action, total_len) in enumerate(
            zip(sequences, attention_masks, num_actions, total_lens)
        ):
            if seq_offset + total_len < packing_max_len:
                seq_offset, seq_index = _accumulate(
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                    sequence,
                    attention_mask,
                    num_action,
                    total_len,
                    custom_rewards,
                    i,
                )
            elif seq_offset + total_len == packing_max_len:
                seq_offset, seq_index = _accumulate(
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                    sequence,
                    attention_mask,
                    num_action,
                    total_len,
                    custom_rewards,
                    i,
                )
                valid_size = out_attention_mask.nonzero().size(0)
                ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                ret_num_actions.append(out_num_actions)
                ret_packed_seq_lens.append(out_packed_seq_lens)
                if custom_rewards:
                    ret_custom_rewards.append(rewards)
                (
                    out_sequence,
                    out_attention_mask,
                    out_num_actions,
                    out_packed_seq_lens,
                    rewards,
                    seq_offset,
                    seq_index,
                ) = _new_instance()
            elif seq_offset + total_len > packing_max_len:
                if seq_offset > 0:
                    valid_size = out_attention_mask.nonzero().size(0)
                    ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
                    ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
                    ret_num_actions.append(out_num_actions)
                    ret_packed_seq_lens.append(out_packed_seq_lens)
                    if custom_rewards:
                        ret_custom_rewards.append(rewards)
                    (
                        out_sequence,
                        out_attention_mask,
                        out_num_actions,
                        out_packed_seq_lens,
                        rewards,
                        seq_offset,
                        seq_index,
                    ) = _new_instance()
                    seq_offset, seq_index = _accumulate(
                        out_sequence,
                        out_attention_mask,
                        out_num_actions,
                        out_packed_seq_lens,
                        rewards,
                        seq_offset,
                        seq_index,
                        sequence,
                        attention_mask,
                        num_action,
                        total_len,
                        custom_rewards,
                        i,
                    )

        if seq_offset > 0:
            valid_size = out_attention_mask.nonzero().size(0)
            ret_sequences.append(out_sequence[:valid_size].unsqueeze(0))
            ret_attention_masks.append(out_attention_mask[:valid_size].unsqueeze(0))
            ret_num_actions.append(out_num_actions)
            ret_packed_seq_lens.append(out_packed_seq_lens)
            if custom_rewards:
                ret_custom_rewards.append(rewards)

        return ret_sequences, ret_attention_masks, ret_num_actions, ret_packed_seq_lens, ret_custom_rewards

    def _get_dp_group_models(self, dp_rank: int, model_type: str = ""):
        model = getattr(self, model_type)
        if model_type == "reward_model":
            model = model[0]
        return model._actor_handlers[dp_rank]

    def _split_dp_batch(self, batch, num_dp, drop_last=False):
        # Convert batch tuple to list of lists, handling None values
        batch_lists = []
        batch_size = None
        for item in batch:
            if item is not None:
                if batch_size is None:
                    batch_size = len(item)
                batch_lists.append(item)
            else:
                batch_lists.append(None)

        if drop_last:
            dp_size = batch_size // num_dp
        else:
            dp_size = (batch_size + num_dp - 1) // num_dp
        valid_size = dp_size * num_dp

        if not drop_last:
            padding_index = None
            for i in range(len(batch_lists)):
                if batch_lists[i] is not None and (
                    isinstance(batch_lists[i], torch.Tensor) or isinstance(batch_lists[i], list)
                ):
                    padding_size = valid_size - len(batch_lists[i])
                    if padding_size > 0:
                        if padding_index is None:
                            if padding_size > len(batch_lists[i]):
                                padding_index = random.choices(range(len(batch_lists[i])), k=padding_size)
                            else:
                                padding_index = random.sample(range(len(batch_lists[i])), padding_size)
                        if isinstance(batch_lists[i], torch.Tensor):
                            batch_lists[i] = torch.cat([batch_lists[i], batch_lists[i][padding_index]], dim=0)
                        elif isinstance(batch_lists[i], list):
                            batch_lists[i] = batch_lists[i] + [batch_lists[i][j] for j in padding_index]

        for i in range(num_dp):
            # Extract micro batch for each input list
            micro_batch = []
            for batch_list in batch_lists:
                if batch_list is None:
                    micro_batch.append(None)
                elif isinstance(batch_list, torch.Tensor) or isinstance(batch_list, list):
                    micro_batch.append(batch_list[i * dp_size : (i + 1) * dp_size])
                else:
                    micro_batch.append(batch_list)
            yield tuple(micro_batch)

    def _split_dp_batch_dynamic_balance(self, batch, num_dp, balanced_values):
        batch = list(batch)
        assert len(batch) == len(balanced_values), "batch and balanced_values must have the same length"
        results = self._split_weighted_objects(zip(balanced_values, batch), num_dp)
        # re organize to the original format
        for i in range(num_dp):
            ret = [[] for _ in range(len(results[i][0]))]
            for sample in results[i]:
                for j, v in enumerate(sample):
                    ret[j].append(v)
            yield ret

    def _split_weighted_objects(self, items, n):
        result = [[] for _ in range(n)]

        heap = [(0, i) for i in range(n)]
        heapify(heap)

        sorted_items = sorted(items, key=lambda x: x[0], reverse=True)

        for weight, obj in sorted_items:
            current_sum, index = heappop(heap)
            result[index].append(obj)
            heappush(heap, (current_sum + weight, index))

        return result

    async def _split_and_run_micro_batch(self, async_fn, batch_args, micro_size):
        # Ensure batch_args is a sequence of lists with equal length
        batch_size = len(batch_args[0])
        results = []
        # Process in micro batches
        for i in range(0, batch_size, micro_size):
            # Take slice i:i+micro_size from each argument
            micro_batch_args = []
            for arg in batch_args:
                if arg is not None:
                    if not isinstance(arg, torch.Tensor) and not isinstance(arg, list):
                        micro_batch_args.append(arg)
                    elif micro_size > 1 or isinstance(arg, torch.Tensor):
                        micro_batch_args.append(arg[i : i + micro_size])
                    else:
                        micro_batch_args.append(arg[i])
                else:
                    micro_batch_args.append(None)
            results.append(await async_fn(*micro_batch_args))
        return results


    def _process_sequences(self, sequences: torch.Tensor, input_len, eos_token_id, pad_token_id):
        attention_mask = (sequences.ne(eos_token_id) & sequences.ne(pad_token_id)).to(dtype=torch.long)
        seq_length = attention_mask.size(1)

        eos_indices = seq_length - attention_mask.long().fliplr().argmax(dim=1, keepdim=True).clamp(min=1)
        sequences.scatter_(dim=1, index=eos_indices, value=eos_token_id)

        # For Llama3 and Qwen2 models, there are some eos_tokens in the middle of the prompt.
        first_token_indices = attention_mask.long().argmax(dim=1, keepdim=True)
        mask = torch.arange(seq_length).unsqueeze(0).expand(sequences.size(0), -1).to(device=sequences.device)
        attention_mask = (mask >= first_token_indices) & (mask <= eos_indices).to(dtype=torch.long)

        # in RL, state_i (current token) + action_i (next token) -> state_i+1 (next token)
        state_seq = sequences[:, input_len - 1 : -1]
        action_mask = state_seq.ne(eos_token_id) & state_seq.ne(pad_token_id)
        action_mask[:, 0] = 1

        return sequences, attention_mask, action_mask

    def _tokenize(self, texts, max_length=99999999, padding=True, device=None):
        if not padding:
            # when padding is False, return tokenized texts as list
            return self.tokenizer(
                texts,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        batch = self.tokenizer(
            texts,
            return_tensors="pt",
            add_special_tokens=False,
            max_length=max_length,
            padding=True,
            truncation=True,
        )
        return {k: v.to(device) for k, v in batch.items()}

    def _detokenize(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)

    def _warp_custom_reward_model_fn(self):
        if self.reward_model:
            # TODO: support multiple reward models]
            num_policy_dp_groups = self.cfg.actor_num_nodes * self.cfg.actor_num_gpus_per_node

            async def warpped_reward_model_fn(prompts: List[str], outputs: List[str]):
                (
                    sequences,
                    attention_mask,
                    _,
                    packed_seq_lens,
                    _,
                ) = self._convert_prompts_outputs_to_batch_tensors_packing(
                    prompts, outputs, None, self.cfg.packing_max_len
                )
                split_iterator = self._split_dp_batch(
                    (sequences, attention_mask, packed_seq_lens), num_policy_dp_groups
                )
                dp_tasks = []

                async def _rm_run(rm, seq, mask, lens):
                    return await rm.forward.remote(seq, mask, packed_seq_lens=lens)

                for dp_rank, args in enumerate(split_iterator):
                    rm = self._get_dp_group_models(dp_rank, "reward_model")
                    dp_tasks.append(
                        self._split_and_run_micro_batch(
                            partial(_rm_run, rm),
                            args,
                            self.cfg.micro_forward_batch_size,
                        )
                    )
                outputs = await asyncio.gather(*dp_tasks)
                outputs = sum(outputs, [])  # gather dp
                outputs = outputs[: len(sequences)]  # drop padding
                outputs = torch.hstack(outputs)

                assert outputs.size(0) == len(prompts), "reward outputs number must be equal to prompts number"
                return outputs

            return warpped_reward_model_fn
        else:
            return None


