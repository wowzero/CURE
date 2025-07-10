import os
import sys
import subprocess
from termcolor import cprint

from optimization import optimization_config
from optimization import sample as sample_module
from optimization import execute as execute_module
from optimization import reward as reward_module
from optimization import train as train_module





# if you are the first time to train the model, set this to be True.
# if you have stopped the process and want to keep training, simply set this to be False and run this script.
start_from_scratch = True



eval_interval = optimization_config.eval_interval
save_interval = optimization_config.save_interval
total_steps = optimization_config.total_steps
pretrain_model = optimization_config.pretrained_model
model = os.path.abspath("") + "/optimization/ckpt/" +  optimization_config.optimized_model_name
if start_from_scratch == False:
    pretrain_model = model
eval_dataset = optimization_config.eval_dataset
train_dataset = optimization_config.train_dataset
gpu_groups = optimization_config.gpu_groups
eval_k_code = optimization_config.eval_k_code
eval_k_case = optimization_config.eval_k_case
eval_scale_tuple_list = optimization_config.eval_scale_tuple_list
eval_num_chunks = optimization_config.eval_num_chunks
eval_no_example = optimization_config.eval_no_example
eval_max_test = optimization_config.eval_max_test


def begin_with(file_name):
    with open(file_name, "w") as f:
        f.write("")

if start_from_scratch:
    os.makedirs("evaluation/results", exist_ok=True)
    os.makedirs("optimization/results", exist_ok=True)
    #begin_with("evaluation/results/results-eval-" + pretrain_model.replace("/", ".") + "-" + eval_dataset + ".txt")
    #begin_with("optimization/results/results-rl-" + pretrain_model.replace("/", ".") + "-" + train_dataset + ".txt")
    begin_with("evaluation/results/results-eval-" + model.replace("/", ".") + "-" + eval_dataset + ".txt")
    begin_with("optimization/results/results-rl-" + model.replace("/", ".") + "-" + train_dataset + ".txt")

# evaluation
def evaluation(model, eval_dataset, gpu_groups):
    cprint(f"This is the {i}-th step for evaluation.", color = "green")
    subprocess.run(
        f'python eval.py '
        f'--pretrained_model {model} '
        f'--dataset {eval_dataset} '
        '--use_api False '
        '--exe_verbose False '
        '--is_final_eval False '
        '--single_eval False '
        f'--k_code {eval_k_code} '
        f'--k_case {eval_k_case} '
        f'--scale_tuple_list "{repr(eval_scale_tuple_list)}" '
        f'--num_chunks {eval_num_chunks} '
        f'--no_example {eval_no_example} '
        f'--max_test {eval_max_test} '
        f'--gpu_groups "{repr(gpu_groups)}" ',
        shell=True,
        cwd='evaluation',
        check=True,
    )


# samlpe
def sample(model):
    cprint(f"This is the {i}-th step for sampling.", color="green")
    return sample_module.main(["--pretrained_model", model])

# execute
def execute(data, iteration):
    cprint(f"This is the {i}-th step for execution.", color="green")
    return execute_module.main(data, iteration=iteration)

# assign reward
def assign_reward(data):
    return reward_module.main(data)

# train
def train(rl_data, model):
    cprint(f"This is the {i}-th step for training.", color="green")
    train_module.main(rl_data)
    subprocess.run("rm -f optimization/ckpt/event*", shell=True, check=True)

# save
def save(model_from, model_to):
    os.makedirs(model_to, exist_ok=True)
    subprocess.run(f"rm -rf {model_to}/*", shell=True, check=True)
    subprocess.run(f"cp -r {model_from}/* {model_to}/", shell=True, check=True)

# the first step if train from scratch
i = 0
#evaluation(pretrain_model, eval_dataset, gpu_groups)
data = sample(pretrain_model)
data = execute(data, i)
rl_data = assign_reward(data)
train(rl_data, pretrain_model)
i += 1

# start the iterative optimization
while i <= total_steps:

    if i % eval_interval == 0:
        evaluation(model, eval_dataset, gpu_groups)
    if i % save_interval == 0:
        save(model, f"optimization/ckpt/iter{i}")

    if i == total_steps:
        break

    data = sample(model)
    data = execute(data, i)
    rl_data = assign_reward(data)
    train(rl_data, model)

    i += 1






