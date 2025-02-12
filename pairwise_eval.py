from evaluator import LLMPairwiseEvaluator
from llm import GPTChat, OllamaClass, NvidiaChat, ClaudeChat, OpenRouterChat
import argparse
import os
import time
import yaml
import json
import logging

import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser(description='Compare two destinations')
parser.add_argument('--llm', type=str, help='LLM', choices=["gpt3", "gpt4", "gpt4o", "gpt4o-mini", \
                                                            "llama3", "mistral", "nvidia", "claude35", "deepseekv3", "llama33"], default="gpt4o-mini")
parser.add_argument('--exp_dir', type=str, help='Experiments directory', default="experiments_strum")
# parser.add_argument('--exp_id',type=str, help='Dataset', default=None)
parser.add_argument('--ref_exp_dir', type=str, help='Reference Experiments directory', default=None)
# parser.add_argument('--ref_exp_id', type=str, help='Reference Experiment ID', default=None)

args = parser.parse_args()

API_KEY = os.getenv("OPENAI_API_KEY")
NVIDIA_API_KEY = os.getenv("NIM_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
llm_gpt3 = GPTChat(model_name='gpt-3.5-turbo-0125', api_key=API_KEY)
llm_gpt4 = GPTChat(model_name='gpt-4-turbo', api_key=API_KEY)
llm_gpt4o = GPTChat(model_name='gpt-4o', api_key=API_KEY) 
llm_gpt4o_mini = GPTChat(model_name='gpt-4o-mini', api_key=API_KEY)
llm_llama3 = OllamaClass(model_name='llama3')
llm_mistral = OllamaClass(model_name='mistral')
llm_nvidia = NvidiaChat(model_name='nvidia/llama-3.1-nemotron-70b-instruct', api_key=NVIDIA_API_KEY)
llm_claude35 = ClaudeChat(model_name='claude-3-5-sonnet-20241022', api_key=CLAUDE_API_KEY)
llm_deepseekv3 = OpenRouterChat(model_name='deepseek/deepseek-chat', api_key=OPENROUTER_API_KEY)
llm_llama33 = OpenRouterChat(model_name='meta-llama/llama-3.3-70b-instruct', api_key=OPENROUTER_API_KEY)

llm_name = args.llm
if llm_name == "gpt3":
    llm = llm_gpt3
elif llm_name == "gpt4":
    llm = llm_gpt4
elif llm_name == "gpt4o":
    llm = llm_gpt4o
elif llm_name == "gpt4o-mini":
    llm = llm_gpt4o_mini
elif llm_name == "llama3":
    llm = llm_llama3
elif llm_name == "mistral":
    llm = llm_mistral
elif llm_name == "nvidia":
    llm = llm_nvidia
elif llm_name == "claude35":
    llm = llm_claude35
elif llm_name == "deepseekv3":
    llm = llm_deepseekv3
elif llm_name == "llama33":
    llm = llm_llama33


exp_dir = "/".join(args.exp_dir.split("/")[:-1])
conf_dir = os.path.join(exp_dir, "config.yaml")

config = None
with open(conf_dir) as f:
    config = yaml.safe_load(f)

domain = config["domain"] if "domain" in config else "travel destination"

exp_id = args.exp_dir.split("/")[-1]
exp_id_dir = os.path.join(exp_dir, exp_id)

ref_exp_dir = "/".join(args.ref_exp_dir.split("/")[:-1])
ref_exp_id = args.ref_exp_dir.split("/")[-1]
ref_exp_id_dir = os.path.join(ref_exp_dir, ref_exp_id)

ref_dir_name = ref_exp_dir.split("/")[-1]

logging.basicConfig(filename=os.path.join(exp_id_dir, "eval.log"), level=eval(f"logging.{config['log_level']}")) 

logger = logging.getLogger("comparative-eval")

# Add StreamHandler to logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

exp_results = None
with open(os.path.join(exp_id_dir, "results.json")) as f:
    exp_results = json.load(f)

ref_results = None
with open(os.path.join(ref_exp_id_dir, "results.json")) as f:
    ref_results = json.load(f)

prompt_dir = "prompts"
pairwise_asp_prompt_path = config["prompts"]["pairwise_asp_eval"]
pp_name = pairwise_asp_prompt_path.split(".")[0]
pairwise_asp_prompt = None
with open(os.path.join(prompt_dir, pairwise_asp_prompt_path)) as f:
    pairwise_asp_prompt = f.read()

pairwise_asp_eval = LLMPairwiseEvaluator(llm, pairwise_asp_prompt)

eval_results = []

results_path = os.path.join(exp_id_dir, f"{ref_dir_name}_{pp_name}_{llm_name}_results.json")
if os.path.exists(results_path):
    with open(results_path) as f:
        eval_results = json.load(f)

logger.info(f"Using LLM: {args.llm}")
logger.info(f"Domain: {domain}")
for i, comparison in enumerate(exp_results):
    if i < len(eval_results):
        logger.info(f"Skipping {i} as it is already processed.")
        continue
    query = comparison["query"]
    dest1 = comparison["dest1"]
    dest2 = comparison["dest2"]
    sents1 = comparison["sentences1"]
    sents2 = comparison["sentences2"]
    compare_output = comparison["output"]

    intermediate = comparison["intermediate"]

    reference = ref_results[i]
    ref_query = reference["query"]
    assert query == ref_query, f"Queries do not match: {query} != {ref_query}"

    reference_output = reference["output"]

    eval_curr = {
        "query": query,
        "dest1": dest1,
        "dest2": dest2,
        "a_output": compare_output,
        "b_output": reference_output,
        "aspect_eval": {},
    }
    if type(compare_output) != dict or type(reference_output) != dict:
        logger.warning(f"Error in output for comparison {i}")
        eval_results.append(eval_curr)
        continue
    aspects_comp = set(compare_output[dest1].keys())
    # aspects_comp = set([asp.lower() for asp in aspects_comp])

    aspects_ref = set(reference_output[dest1].keys())
    # aspects_ref = set([asp.lower() for asp in aspects_ref])

    aspects = aspects_comp & aspects_ref

    d1_out = list(compare_output[dest1].values())
    d2_out = list(compare_output[dest2].values())

    logger.info(f"Evaluating comparison between {dest1} and {dest2} for query: {query}")


    for asp in aspects:
        if asp not in compare_output[dest1] or asp not in compare_output[dest2] or asp not in reference_output[dest1] or asp not in reference_output[dest2]:
            logger.warning(f"Aspect {asp} not found in comparison or reference output")
            continue
        logger.info(f"Evaluating aspect: {asp}")
        comp_asp_result = {
            dest1: compare_output[dest1][asp],
            dest2: compare_output[dest2][asp]
        }
        ref_asp_result = {
            dest1: reference_output[dest1][asp],
            dest2: reference_output[dest2][asp]
        }

        eval_result = pairwise_asp_eval.evaluate(comp_asp_result, ref_asp_result, query, domain)
        eval_curr["aspect_eval"][asp] = eval_result

    eval_results.append(eval_curr)

    with open(results_path, "w") as f:
        json.dump(eval_results, f, indent=4)
    
    time.sleep(1)
        
logger.info(f"Evaluation complete, results saved to {ref_dir_name}_{pp_name}_results.json")
