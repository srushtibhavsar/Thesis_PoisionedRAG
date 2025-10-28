import os

def run(test_params):

    log_file, log_name = get_log_name(test_params)

    cmd = f"nohup python3 -u perplexity_main.py \
        --eval_model_code {test_params['eval_model_code']}\
        --eval_dataset {test_params['eval_dataset']}\
        --split {test_params['split']}\
        --query_results_dir {test_params['query_results_dir']}\
        --model_name {test_params['model_name']}\
        --top_k {test_params['top_k']}\
        --use_truth {test_params['use_truth']}\
        --gpu_id {test_params['gpu_id']}\
        --attack_method {test_params['attack_method']}\
        --adv_per_query {test_params['adv_per_query']}\
        --score_function {test_params['score_function']}\
        --repeat_times {test_params['repeat_times']}\
        --M {test_params['M']}\
        --seed {test_params['seed']}\
        --name {log_name}\
        --enable_ppl_defense {test_params['enable_ppl_defense']}\
        --ppl_threshold {test_params['ppl_threshold']}\
        --ppl_percentile {test_params['ppl_percentile']}\
        --save_ppl_stats {test_params['save_ppl_stats']}\
        > {log_file} &"
        
    os.system(cmd)


def get_log_name(test_params):
    # Generate a log file name
    os.makedirs(f"/mnt/SAS_A/srushti_thesis/Final_Code/PoisonedRAG/perplexity/logs/{test_params['query_results_dir']}_logs", exist_ok=True)

    if test_params['use_truth']:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Truth--M{test_params['M']}x{test_params['repeat_times']}"
    else:
        log_name = f"{test_params['eval_dataset']}-{test_params['eval_model_code']}-{test_params['model_name']}-Top{test_params['top_k']}--M{test_params['M']}x{test_params['repeat_times']}"
    
    if test_params['attack_method'] != None:
        log_name += f"-adv-{test_params['attack_method']}-{test_params['score_function']}-{test_params['adv_per_query']}-{test_params['top_k']}"

    if test_params['note'] != None:
        log_name = test_params['note']
    
    return f"/mnt/SAS_A/srushti_thesis/Final_Code/PoisonedRAG/perplexity/logs/{test_params['query_results_dir']}_logs/{log_name}.txt", log_name



test_params = {
    # beir_info
    'eval_model_code': "contriever",
    'eval_dataset': "nq",
    'split': "test",
    'query_results_dir': 'main',

    # LLM setting
    'model_name': 'llama7b', 
    'use_truth': False,
    'top_k': 5,
    'gpu_id': 1,

    # attack
    'attack_method': 'LM_targeted',
    'adv_per_query': 5,
    'score_function': 'dot',
    'repeat_times': 10,
    'M': 10,
    'seed': 12,
    
    # Perplexity defense
    'enable_ppl_defense': 'True',      # turn defense on
    'ppl_threshold': None,             # derive from percentile if None
    'ppl_percentile': 95.0,            # use 95th percentile of clean texts
    'save_ppl_stats': 'True',          # save stats for ROC/AUC

    'note': None
}

# for dataset in ['nq', 'hotpotqa', 'msmarco']:
#     test_params['eval_dataset'] = dataset
run(test_params)