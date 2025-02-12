from llm import GPTChat, OllamaClass, NvidiaChat, ClaudeChat
import json
from general_comparer import GeneralComparer
import os
import argparse
import yaml
import logging
import datetime
import tqdm

from dotenv import load_dotenv
load_dotenv()

parser = argparse.ArgumentParser(description='Compare two destinations')
parser.add_argument('--llm', type=str, help='LLM', choices=["gpt3", "gpt4", "gpt4o", "gpt4o-mini", \
                                                            "llama3", "mistral", "nvidia", "claude-opus", \
                                                                "claude-sonnet", "claude-haiku"], default=None)
parser.add_argument('--exp_dir', type=str, help='Experiments directory', default="experiments_strum")
parser.add_argument('--exp_id', type=str, help='Experiment ID', default=None)
parser.add_argument('--limit', type=int, help='No. of queries limit', default=float('inf'))

args = parser.parse_args()

API_KEY = os.getenv("OPENAI_API_KEY")
NVIDIA_API_KEY = os.getenv("NIM_API_KEY")
CLAUDE_API_KEY = os.getenv("CLAUDE_API_KEY")
llm_gpt3 = GPTChat(model_name='gpt-3.5-turbo-0125', api_key=API_KEY)
llm_gpt4 = GPTChat(model_name='gpt-4-turbo', api_key=API_KEY)
llm_gpt4o = GPTChat(model_name='gpt-4o', api_key=API_KEY) 
llm_gpt4o_mini = GPTChat(model_name='gpt-4o-mini', api_key=API_KEY)
llm_llama3 = OllamaClass(model_name='llama3')
llm_mistral = OllamaClass(model_name='mistral')
llm_nvidia = NvidiaChat(model_name='nvidia/llama-3.1-nemotron-70b-instruct', api_key=NVIDIA_API_KEY)
llm_claude_opus = ClaudeChat(model_name='claude-3-opus-20240229', api_key=CLAUDE_API_KEY)
llm_claude_sonnet = ClaudeChat(model_name='claude-3-5-sonnet-20241022', api_key=CLAUDE_API_KEY)
llm_claude_haiku = ClaudeChat(model_name='claude-3-haiku-20240307', api_key=CLAUDE_API_KEY)

exp_dir = args.exp_dir
conf_dir = os.path.join(exp_dir, "config.yaml")

exp_id = args.exp_id if args.exp_id else datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

config = None
with open(conf_dir) as f:
    config = yaml.safe_load(f)  

llm_comparer = config["llm_comparer"] if "llm_comparer" in config else None 

# set logging path to experiments directory
exp_id_dir = os.path.join(exp_dir, exp_id)
os.makedirs(exp_id_dir, exist_ok=True)
logging.basicConfig(filename=os.path.join(exp_id_dir, f"{llm_comparer}.log"), level=eval(f"logging.{config['log_level']}")) 

logger = logging.getLogger("comparative")

# Add StreamHandler to logger
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

llm = config["llm"] if "llm" in config else args.llm
logger.info(f"Using LLM: {llm}")
if llm == "gpt3":
    llm = llm_gpt3
elif llm == "gpt4":
    llm = llm_gpt4
elif llm == "gpt4o":
    llm = llm_gpt4o
elif llm == "gpt4o-mini":
    llm = llm_gpt4o_mini
elif llm == "llama3":
    llm = llm_llama3
elif llm == "mistral":
    llm = llm_mistral
elif llm == "nvidia":
    llm = llm_nvidia
elif llm == "claude-opus":
    llm = llm_claude_opus
elif llm == "claude-sonnet":
    llm = llm_claude_sonnet
elif llm == "claude-haiku":
    llm = llm_claude_haiku

debate_mode = config["debate_mode"] if "debate_mode" in config else "single"

prompt_dir = config["prompt_dir"]
prompt_paths = config["prompts"]
prompts = {}
for key in prompt_paths:
    pth = os.path.join(prompt_dir, prompt_paths[key])
    if os.path.exists(pth):
        with open(pth) as f:
            prompts[key] = f.read()

dataset_name = config["dataset"]
logger.info(f"Using dataset: {dataset_name}")

mode = config["mode"] if "mode" in config else "value_merge"
logger.info(f"Using mode: {mode}")

debate = config["debate"] if "debate" in config else False
debate_iter = config["debate_iter"] if "debate_iter" in config else 1
logger.info(f"Debate: {debate}, Debate iterations: {debate_iter}")

model_class = eval(llm_comparer) if llm_comparer else STRUM


compare_model = model_class(logger, llm, prompts=prompts, mode=mode, debate=debate, debate_iter=debate_iter, debate_mode=debate_mode)

with open(f"datasets/{dataset_name}.json") as f:
    dataset = json.load(f)

if type(dataset) == dict:
    dataset = [dataset]

exp_results = []

results_path = os.path.join(exp_id_dir, "results.json")
if os.path.exists(results_path):
    with open(results_path) as f:
        exp_results = json.load(f)

limit = min(args.limit, len(dataset))

for i, data in enumerate(dataset[:limit]):
    if i < len(exp_results):
        logger.info(f"Skipping {i} as it is already processed.")
        continue
    query = data["query"]
    dest1 = data["dest_1"]
    sents1 = data["sentences_1"]
    dest2 = data["dest_2"]
    sents2 = data["sentences_2"]
    intermediate_results = data["intermediate_results"] if "intermediate_results" in data else {}
    output_just = None

    logger.info(f"Comparing {dest1} and {dest2} using query: {query} - intermediate results: {len(intermediate_results) > 0}")

    retry = True
    retry_count = 0
    orig_intermediate_results = intermediate_results
    while retry and retry_count < 3:
        try:
            intermediate, output = compare_model.compare(query, dest1, sents1, dest2, sents2, intermediate_results=intermediate_results)
            retry = False
        except Exception as e:
            logger.error(f"Error while comparing {dest1} and {dest2}: {e}")
            intermediate = {}
            output = f"Error: {e}"
            intermediate_results = orig_intermediate_results
            retry_count += 1

        if "justification" in output:
            output_just = output["justification"]
            del output["justification"]

        entry = {
            "query": query,
            "dest1": dest1,
            "dest2": dest2,
            "sentences1": sents1,
            "sentences2": sents2,
            "intermediate": intermediate,
            "output": output
        }

        if output_just:
            entry["output_justification"] = output_just

        if (dest1 not in output or dest2 not in output) and not retry:
            retry = True
            retry_count += 1
            logger.error(f"Retry {retry_count} for {dest1} and {dest2}")

    exp_results.append(entry)

    with open(results_path, "w") as f:
        json.dump(exp_results, f, indent=4)

logger.info(f"Experiment results saved in {exp_id_dir}")



