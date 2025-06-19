#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from mcts_rl.trainers.tsrl_trainer import TSRLTrainer

def main():
    # —— 环境配置 —— #
    os.environ["TRANSFORMERS_CACHE"] = "./cache"
    # 模型名（确保 conda-recipe 已安装对应模型）
    model_name = "mistral-7b"

    # —— 加载模型与分词器 —— #
    print("加载模型与分词器…")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    # —— 训练参数 —— #
    params = {
        "task_name": "mathqa",
        "dataset_name": "gsm8k",
        "data_dir": "./data",
        "output_dir": "./output/mathqa",
        "max_train_steps": 10000,
        "batch_size": 4,
        "learning_rate": 1e-5,
        "mcts_simulations": 20,        # 每个节点 MCTS 模拟次数
        "rollout_steps": 8,           # 每次 rollout 长度
        "self_eval_weight": 0.5,      # 自评得分权重 λ
        "kl_beta": 0.01,              # DPO 中 KL 惩罚系数 β
        "pref_threshold": 0.0,        # DPO 中偏好阈值 τ
        "save_checkpoint_steps": 2000,
    }

    # —— 初始化 Trainer —— #
    print("初始化 Trainer…")
    trainer = TSRLTrainer(
        model=model,
        tokenizer=tokenizer,
        **params
    )

    # —— 开始训练 —— #
    print("开始 MCTS-DPO 算术推理训练…")
    trainer.train()

if __name__ == "__main__":
    main()
