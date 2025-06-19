import torch
from transformers import AutoTokenizer

def compute_step_reward(state, action, model, tokenizer, self_eval_weight):
    """
    计算单步奖励：结果正确性 + 自评信心
    """
    # 1) 终止奖励简单置零，或可根据需要实现对答案正确性的判断
    r_outcome = 0.0

    # 2) 自评置信度取当前 token 的 log-prob
    with torch.no_grad():
        enc = tokenizer(state + action, return_tensors="pt", truncation=True, max_length=512)
        logits = model(**enc).logits
        logp = torch.log_softmax(logits[:, -1, :], dim=-1)
        token_id = tokenizer.encode(action, add_special_tokens=False)[0]
        r_self = logp[0, token_id].item()

    return r_outcome + self_eval_weight * r_self

def extract_preferences(prefs, threshold=0.0, tokenizer=None):
    """
    将 prefs 列表拆分为正/负样本 batch，返回字典格式
    """
    pos_inputs, neg_inputs = [], []
    for state, pos, neg in prefs:
        pos_enc = tokenizer(state + pos, return_tensors="pt", truncation=True, max_length=512)
        neg_enc = tokenizer(state + neg, return_tensors="pt", truncation=True, max_length=512)
        pos_inputs.append(pos_enc)
        neg_inputs.append(neg_enc)

    def collate(batch):
        import torch
        input_ids = torch.nn.utils.rnn.pad_sequence(
            [b.input_ids.squeeze(0) for b in batch], batch_first=True, padding_value=tokenizer.pad_token_id
        )
        masks = torch.nn.utils.rnn.pad_sequence(
            [b.attention_mask.squeeze(0) for b in batch], batch_first=True, padding_value=0
        )
        return {"input_ids": input_ids, "attention_mask": masks}

    return collate(pos_inputs), collate(neg_inputs)
