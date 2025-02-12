# Q-STRUM

## Overview

Q-STRUM is a tool for comparing two entities using a set of prompts and a language model.

### Datasets

The datasets are located in the `datasets` folder. There are three datasets:

- `datasets/hotels.json`: The Hotels dataset
- `datasets/restaurants.json`: The Restaurants dataset
- `datasets/traveldestinations.json`: The TravelDest dataset

## Getting Started

### Requirements

To run the experiments, you need to install the required packages listed in `requirements.txt`. You can do this using pip:

```bash
pip install -r requirements.txt
```

### Environment Variables

Before running the experiments, you need to set the following environment variables:

- `OPENAI_API_KEY`: Your OpenAI API key
- `NIM_API_KEY`: Your NVIDIA API key
- `CLAUDE_API_KEY`: Your Claude API key
- `OPENROUTER_API_KEY`: Your OpenRouter API key
You can set these variables in your `.env` file or directly in your shell. There is an example file in the root directory called `example.env`.

## Experiment Setup

### Configuration

The configuration is done in the `config.yaml` file. A sample configuration file, `config.yaml` is provided in the root directory. Place a copy of this file in the directory of the experiment you want to run. The configuration parameters are as follows:

- `llm`: The language model to be used for comparisons (e.g., `gpt4o`).
- `dataset`: The path within the datasets directory to the dataset you will be using for the experiments (e.g., `hotels` will use the `hotels.json` file in the `datasets` directory).
- `domain`: The domain of the dataset (e.g., `hotel`).
- `prompt_dir`: The directory where the prompt templates are stored (e.g., `prompts`).
- `llm_comparer`: The class used for comparing the outputs (e.g., `GeneralComparer`).
- `log_level`: The logging level for the experiments (e.g., `DEBUG`).
- `debate`: A boolean indicating whether to enable debate mode (e.g., `True`).
- `debate_iter`: The number of iterations for the debate mode (e.g., `1`).
- `debate_mode`: The mode of debate (e.g., `single` to debate one aspect at a time, `all` to debate all aspects at once).
- `mode`: Specifies which intermediate stages to run (e.g., `filter` to apply the filter stage after attribute merge, `attr_merge` to only run the attribute merge stage, `value_merge` to run the value merge stage).
- `prompts`: Specifies the specific prompt templates to be used for each stage.
    - `extract`: The prompt template for the extract stage.
    - `attribute_merge`: The prompt template for the attribute merge stage.
    - `filter`: The prompt template for the filter stage.
    - `value_merge`: The prompt template for the value merge stage.
    - `contrast`: The prompt template for the contrast stage.
    - `debate`: The prompt template for the debate stage.
    - `debate_json`: The prompt template for the debate summary stage.
    - `usefulness`: The prompt template for the usefulness stage.
    - `pairwise_asp_eval`: The prompt template for the pairwise evaluation stage.

### Running the Experiments

To run the experiments, you can use the following command:

```bash
python main.py --llm <llm_name> --exp_dir <experiment_directory> [--exp_id <experiment_id>]
```

The output will be saved in `{exp_dir}/{exp_id}` with the name `results.json`.

### Extracting Intermediate Results

To extract the intermediate results from the experiment, use the `notebooks/intermediate_results_extractor.ipynb` notebook. Instructions are provided in the notebook. Note, when performing pairwise evaluation, you should use the intermediate results as the dataset for the other experiments, to ensure consistent intermediate outputs.

## Evaluation

After running the experiments, you can evaluate the results by running the following command:

```bash
python pairwise_eval.py --llm <llm_name> --exp_dir <experiment_directory> --ref_exp_dir <reference_experiment_directory>
```

The output will be saved in `{exp_dir}\` with the name `{parent of ref_exp_dir}_{eval prompt name}_{llm}_results.json`.

In the evaluation process, you will be comparing two outputs: `exp_id` and `ref_exp_id`. Here, `exp_id` is referred to as "A" and `ref_exp_id` is referred to as "B". 

To ensure a comprehensive evaluation, it is important to run the comparisons in both directions. This means you will compare "A" against "B" and then "B" against "A". 

For this purpose, you can utilize the Jupyter notebook located at `notebooks/pairwise_eval_bidirectional.ipynb` to obtain the bidirectional results. This notebook is designed to facilitate the evaluation of the two experiment outputs effectively.

### Bidirectional Bias Mitigation

In order to mitigate the bias that may arise from the direction of the comparison, we can run the comparison in both directions. This will give us two sets of results. In the paper, we only consider a win if both directions agree on the win, otherwise the result is a tie. To do this, we can use the Jupyter notebook located at `notebooks/pairwise_eval_bidirectional.ipynb`. Instructions are provided in the notebook.








