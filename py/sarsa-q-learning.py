import numpy as np
import pdb 

# 环境参数
states = range(10)  # 假设有10个状态
actions = ['left', 'right']  # 可能的动作
gamma = 0.9  # 折扣因子
alpha = 0.1  # 学习率
epsilon = 0.1  # 探索率

# Q表初始化
Q = {s: {a: 0 for a in actions} for s in states}

# 环境交互函数示例
def step(state, action):
    if action == 'right':
        next_state = state + 1 if state < len(states) - 1 else state
        reward = 1 if next_state == len(states) - 1 else -1
    else:
        next_state = state - 1 if state > 0 else state
        reward = -1
    return next_state, reward

# epsilon-greedy 策略
def choose_action(state, Q):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        return max(Q[state], key=Q[state].get)


def sarsa(Q, episodes):
    for _ in range(episodes):
        state = np.random.choice(states)  # 随机初始状态
        action = choose_action(state, Q)  # 根据当前Q表选择动作
        
        while True:
            next_state, reward = step(state, action)
            next_action = choose_action(next_state, Q)  # 在新状态选择动作
            # Q值更新
            Q[state][action] += alpha * (reward + gamma * Q[next_state][next_action] - Q[state][action])
            state, action = next_state, next_action
            
            if state == len(states) - 1:  # 达到终点状态
                break


def q_learning(Q, episodes):
    for _ in range(episodes):
        state = np.random.choice(states)  # 随机初始状态
        
        while True:
            action = choose_action(state, Q)  # 根据当前Q表选择动作
            next_state, reward = step(state, action)
            # Q值更新，使用max来选择最佳未来动作的值
            pdb.set_trace()
            best_next_action = max(Q[next_state], key=Q[next_state].get)
            Q[state][action] += alpha * (reward + gamma * Q[next_state][best_next_action] - Q[state][action])
            
            state = next_state
            
            if state == len(states) - 1:  # 达到终点状态
                break

def main():
    # 设置随机种子以便结果可复现
    np.random.seed(42)

    # 算法参数
    episodes = 1000  # 总的训练回合数

    # Q表初始化
    Q_sarsa = {s: {a: 0 for a in actions} for s in states}
    Q_q_learning = {s: {a: 0 for a in actions} for s in states}

    # 训练 Sarsa
    sarsa(Q_sarsa, episodes)
    print("Sarsa learned Q-values:")
    for state in sorted(Q_sarsa.keys()):
        print(f"State {state}: {Q_sarsa[state]}")

    # 训练 Q-learning
    q_learning(Q_q_learning, episodes)
    print("\nQ-learning learned Q-values:")
    for state in sorted(Q_q_learning.keys()):
        print(f"State {state}: {Q_q_learning[state]}")

if __name__ == '__main__':
    # main()

    Q = {
    'state1': {'action1': -1.0, 'action2': 1.5},
    'state2': {'action1': 0.5, 'action2': 2.0}
    }

    print(max(Q['state1'], key=Q['state1'].get))