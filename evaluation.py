
from typing import List, Callable, Dict

import os
import json
import re
import string
import time
import numpy as np

import hydra
from path import Path
from hydra.core.hydra_config import HydraConfig
from datasets import load_dataset, Dataset

from openai import OpenAI
from evaluate import load

import inflect



# This prompt is modified from
#"https://github.com/SalesforceAIResearch/SFR-RAG/blob/main/README_ContextualBench.md"

SYSTEM_PROMPT="""
You are an expert in retrieval QA. Please respond with the exact answer only. Dont be verbose or provide extra information."""
USER_PROMPT="""
Based on the given context answer the question, Context: {context} Question: {question} Answer:"""

ENGINE = inflect.engine()

def get_model(host:str, api_key: str, port: int) -> OpenAI:
    
    #if llm_type == "openai":
    client = OpenAI(
    base_url=f"http://{host}:{port}/v1",
    api_key=api_key,)
        
    return client


def get_metric(metrics: List[str] | str) -> Dict[str, Callable]:
    if isinstance(metrics, str):
        return {metrics: load(metrics)}    
    return {metric: load(metric) for metric in metrics}


def generate_inference(client: OpenAI, model: str, dataset: Dataset, gen_config: Dict) -> Dict[str, str]:
    inferences = dict()
    performances = dict()
    for idx, example in enumerate(dataset):
        prompt = get_prompt(example)
        inference, performance = get_inference(client, model, prompt, gen_config)
        inferences[idx] = inference
        performances[idx] = performance
    return inferences, performances

def get_prompt(example: Dataset) -> List[dict[str, str]]:
    return [{
        "role": "system",
        "content": SYSTEM_PROMPT
    },
    {
        "role": "user",
        "content": USER_PROMPT.format(context=" ".join(example["retrieved_contexts"]), 
                                      question=example["user_input"])
    }]
    
    

def get_inference(client : OpenAI , model: str, messages: List[dict[str, str]], gen_config: Dict) -> str:

    start_time = time.time()
    completion = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=gen_config.max_tokens,
    temperature=gen_config.temperature,
    top_p=gen_config.top_p,)
    end_time = time.time()
    
    gen_token_num = completion.usage.completion_tokens
    prompt_token_num = completion.usage.prompt_tokens
    performance = {"latency" : end_time - start_time,
                    "throughput": gen_token_num /(end_time - start_time),
                    "completion_tokens": gen_token_num,
                    "prompt_tokens": prompt_token_num}
    
    return completion.choices[0].message.content, performance
    

def get_references(dataset: Dataset) -> Dict[str, str]:
    return {idx: data["response"] for idx, data in enumerate(dataset)}

def normalizer(s,p=ENGINE) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def handle_punc(text):
        exclude = set(string.punctuation + "".join([u"‘", u"’", u"´", u"`"]))
        return ''.join(ch if ch not in exclude else ' ' for ch in text)

    def lower(text):
        return text.lower()

    def replace_underscore(text):
        return text.replace('_', ' ')
    
    def convert_numbers_to_words(text):
        words = text.split()
        result = []
        for word in words:
            if word.isdigit() and int(word) < 100:
                word_in_words = p.number_to_words(int(word))
                result.append(word_in_words)
            else:
                result.append(word)
        return ' '.join(result)

    return white_space_fix(remove_articles(handle_punc(convert_numbers_to_words(lower(replace_underscore(s)))))).strip()


def mean_value(dict_: Dict[str, List[float]], item_) -> float:
    return np.mean([ l[item_] for _, l in dict_.items()])

def evaluate_inference(inferences: Dict[str, str], references:Dict[str, str], performances:Dict[str, str] , metrics: Dict[str, Callable]) -> Dict[str, Dict[str, float]]:
    results = dict()
    inferences = [normalizer(inference) for inference in inferences.values()]
    references = [normalizer(reference) for reference in references.values()]

    print("===")
    for i in range(5):
        print("inference: ", inferences[i])
        print("references:", references[i])
        print("===")

    for metric_name, metric in metrics.items():
        
        result = metric.compute(predictions=inferences, references=references)
        results[metric_name] = result
    
    results['latency'] = mean_value(performances, 'latency')
    results['throughput'] = mean_value(performances, 'throughput')
    results['completion_tokens'] = mean_value(performances, 'completion_tokens')
    results['prompt_tokens'] = mean_value(performances, 'prompt_tokens') 
    return results


def save_results(results: Dict[str, Dict[str, float]], results_dir: str, model_name : str) -> None:
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    with open(f"{results_dir}/{model_name}.json", "w") as f:
        json.dump(results, f, indent=4)
    pass


@hydra.main(version_base=None, config_path="./config", config_name="config.yaml")
def main(config):
    running_config = HydraConfig.get()
    config_name = Path(running_config.job.config_name).stem

    cache_dir = config.paths.cache_dir
    results_dir = config.paths.results_dir

    
    dataset = load_dataset(config.dataset.hf_id, config.dataset.hf_sub_id, split=config.dataset.split, cache_dir=cache_dir)
    

    model = get_model(config.host, config.api_key, config.port)
    metrics = get_metric(config.metrics)
    references = get_references(dataset)
    
    model_id = config.model.llm_id
    inferences, performances = generate_inference(model, model_id, dataset, gen_config=config.gen_config)

    results = evaluate_inference(inferences, references, performances, metrics)
    save_results(results, results_dir, config.model.llm_pretty_name)

if __name__ == "__main__":
    main()