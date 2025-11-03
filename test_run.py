import os

# Change this if you saved the main to a different filename
SCRIPT = "test_mian.py"

def run(test_params):
    log_file, log_name = get_log_name(test_params)

    # base args (shared)
    base_args = [
        f"--eval_model_code {test_params['eval_model_code']}",
        f"--eval_dataset {test_params['eval_dataset']}",
        f"--split {test_params['split']}",
        f"--query_results_dir {test_params['query_results_dir']}",
        f"--model_name {test_params['model_name']}",
        f"--top_k {test_params['top_k']}",
        f"--use_truth {str(test_params['use_truth'])}",
        f"--gpu_id {test_params['gpu_id']}",
        f"--attack_method {test_params['attack_method']}",
        f"--adv_per_query {test_params['adv_per_query']}",
        f"--score_function {test_params['score_function']}",
        f"--repeat_times {test_params['repeat_times']}",
        f"--M {test_params['M']}",
        f"--seed {test_params['seed']}",
        f"--name {log_name}",
    ]

    # ppl args
    if test_params.get("use_ppl_defense", True):
        ppl_args = [
            "--use_ppl_defense True",
            f"--ppl_backend {test_params['ppl_backend']}",
            f"--ppl_filter_mode {test_params['ppl_filter_mode']}",
            f"--ppl_threshold {test_params['ppl_threshold']}",
            f"--ppl_keep_ratio {test_params['ppl_keep_ratio']}",
            f"--ppl_max_length {test_params['ppl_max_length']}",
        ]
        if test_params["ppl_backend"] == "openai":
            # cl100k_base tokenizer is implied by OpenAI chat models
            ppl_args += [f"--openai_model {test_params['openai_model']}"]
        else:  # hf backend
            ppl_args += [
                f"--ppl_model_name {test_params['ppl_model_name']}",
                f"--ppl_batch_size {test_params['ppl_batch_size']}",
            ]
    else:
        ppl_args = ["--use_ppl_defense False"]

    args_str = " ".join(base_args + ppl_args)

    cmd = f"nohup python3 -u {SCRIPT} {args_str} > {log_file} &"
    print(f"Launching:\n{cmd}\nLogs: {log_file}")
    os.system(cmd)


def get_log_name(test_params):
    # Directory per backend for clarity
    backend_tag = test_params.get("ppl_backend", "none")
    out_dir = f"perplexity/new/logs/{backend_tag}/{test_params['query_results_dir']}_logs"
    os.makedirs(out_dir, exist_ok=True)

    if test_params['use_truth']:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}"
    else:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"

    if test_params['attack_method'] is not None:
        log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"

    if test_params.get('note'):
        log_name = test_params['note']

    return os.path.join(out_dir, f"{log_name}.txt"), log_name


# ===================== EXAMPLES =====================

# OpenAI backend (cl100k_base) — matches the paper’s setup
test_params = {
    # beir_info
    "eval_model_code": "contriever",
    "eval_dataset": "nq",
    "split": "test",
    "query_results_dir": "main",

    # LLM setting
    "model_name": "llama7b",
    "use_truth": False,
    "top_k": 5,
    "gpu_id": 1,

    # attack
    "attack_method": "LM_targeted",
    "adv_per_query": 5,
    "score_function": "dot",
    "repeat_times": 10,
    "M": 10,
    "seed": 12,

    # PPL defense (OpenAI cl100k_base)
    "use_ppl_defense": True,
    "ppl_backend": "openai",            # <- OpenAI backend (uses cl100k_base)
    "openai_model": "gpt-4o",      # any cl100k_base chat model
    "ppl_filter_mode": "high",          # 'high' = drop high-PPL; 'low' = drop low-PPL
    "ppl_threshold": 80.0,
    "ppl_keep_ratio": 1.0,
    "ppl_max_length": 512,
    # HF-only args (ignored when ppl_backend=openai)
    "ppl_model_name": "gpt2",
    "ppl_batch_size": 8,

    "note": None,
}

# If you want a fast local baseline instead, set:
# test_params["ppl_backend"] = "hf"
# test_params["ppl_model_name"] = "gpt2-medium"  # for a bit stronger LM
# test_params["ppl_batch_size"] = 8

if __name__ == "__main__":
    run(test_params)
