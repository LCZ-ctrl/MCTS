import math
import torch
from mcts_rl.algorithms.mcts.utils import compute_step_reward

class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.N = 0
        self.W = 0.0
        self.Q = 0.0
        self.P = 0.0

class MCTSSearch:
    def __init__(self, model, tokenizer, num_simulations, rollout_steps, self_eval_weight):
        self.model = model
        self.tokenizer = tokenizer
        self.num_simulations = num_simulations
        self.rollout_steps = rollout_steps
        self.self_eval_weight = self_eval_weight

    def run_batch(self, batch):
        prefs = []
        for example in batch:
            root = Node(state=example)
            self._expand(root)
            for _ in range(self.num_simulations):
                leaf, path = self._select(root)
                reward = self._simulate(leaf)
                self._backup(path, reward)
            prefs.append(self._extract_preferences(root))
        return prefs

    def _select(self, node):
        path = []
        while node.children:
            total_N = sum(c.N for c in node.children.values())
            best, best_val = None, -float("inf")
            for child in node.children.values():
                u = child.Q + child.P * math.sqrt(total_N) / (1 + child.N)
                if u > best_val:
                    best_val, best = u, child
            node = best
            path.append(node)
        return node, path

    def _expand(self, node):
        inputs = self.tokenizer(
            node.state, return_tensors="pt", truncation=True, max_length=512
        )
        logits = self.model(**inputs).logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1).squeeze(0)
        topk = torch.topk(probs, k=10)
        for idx, p in zip(topk.indices.tolist(), topk.values.tolist()):
            action = self.tokenizer.decode([idx])
            child_state = node.state + action
            child = Node(state=child_state, parent=node, action=action)
            child.P = p
            node.children[action] = child

    def _simulate(self, node):
        total_reward = 0.0
        state = node.state
        for _ in range(self.rollout_steps):
            inputs = self.tokenizer(state, return_tensors="pt", truncation=True, max_length=512)
            logits = self.model(**inputs).logits[:, -1, :]
            action_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1).item()
            action = self.tokenizer.decode([action_id])
            next_state = state + action
            r = compute_step_reward(state, action, self.model, self.tokenizer, self.self_eval_weight)
            total_reward += r
            state = next_state
        return total_reward

    def _backup(self, path, reward):
        for node in reversed(path):
            node.N += 1
            node.W += reward
            node.Q = node.W / node.N

    def _extract_preferences(self, root):
        prefs = []
        def dfs(node):
            if not node.children:
                return
            best = max(node.children.values(), key=lambda c: c.Q)
            worst = min(node.children.values(), key=lambda c: c.Q)
            prefs.append((node.state, best.action, worst.action))
            for child in node.children.values():
                dfs(child)
        dfs(root)
        return prefs
