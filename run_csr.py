#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from mcts_rl.trainers.tsrl_trainer import TSRLTrainer

def main():
    os.environ["TRANSFORMERS_CACHE"] = "./cache"
    model_name = "mistral-7b"

    print("加载模型与分词器…")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    params = {
        "task_name": "csr",           # Commonsense Reasoning
        "dataset_name": "ARC-Challenge",
        "data_dir": "./data",
        "output_dir": "./output/csr",
        "max_train_steps": 8000,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "mcts_simulations": 15,
        "rollout_steps": 6,
        "self_eval_weight": 0.5,
        "kl_beta": 0.01,
        "pref_threshold": 0.0,
        "save_checkpoint_steps": 2000,
    }

    print("初始化 CSR Trainer…")
    trainer = TSRLTrainer(
        model=model,
        tokenizer=tokenizer,
        **params
    )

    print("开始 MCTS-DPO 常识推理训练…")
    trainer.train()

if __name__ == "__main__":
    main()
