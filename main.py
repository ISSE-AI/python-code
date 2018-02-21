"""
Make it more robust.
Stop episode once the finger stop at the final position for 50 steps.
Feature & reward engineering.
"""

#from part5.env import ArmEnv
#from part5.rl import DDPG

from env import ArmEnv
from rl import DDPG      # 强化学习算法采用的是DDPG，DDPG相关函数模型是由TensorFlow提供


MAX_EPISODES = 500       # 测试回合数
MAX_EP_STEPS = 200       # 每回合强制结束回合的步数上限


#ON_TRAIN = True   # 训练
ON_TRAIN = False   # 测试



# set env
env = ArmEnv()
s_dim = env.state_dim      # 强化学习算法有几个输入state
a_dim = env.action_dim     # 机械臂有两个关节的动作
a_bound = env.action_bound # 规定一下action的边界，可以不规定



# set RL method (continuous) : DDPG - 深度强化学习算法
rl = DDPG(a_dim, s_dim, a_bound)



steps = []


# 训练
def train():
    # start training
    for i in range(MAX_EPISODES):            # 训练多少个回合
        s = env.reset()                      # 每回合前进行初始化       
        ep_r = 0.
        for j in range(MAX_EP_STEPS):        # 每个回合最大的步数
            env.render()                     # 可视化

            a = rl.choose_action(s)          # 深度学习：输入state→神经网络→输出action

            s_, r, done = env.step(a)        # 强化学习：根据这个action（输出的动作）给出环境反馈即下一个state和reward

            rl.store_transition(s, a, r, s_) # DDPG算法需要有一个存放记忆库的步骤，存满后，开始学习

            ep_r += r
            if rl.memory_full:               # 判断记忆库是否存满
                # start to learn once has fulfilled the memory
                rl.learn()                   # 判断记忆库已存满，开始学习

            s = s_                           # 迭代更新进入下一轮循环
            if done or j == MAX_EP_STEPS-1:
                print('Ep: %i | %s | ep_r: %.1f | step: %i' % (i, '---' if not done else 'done', ep_r, j))
                break
    rl.save() # 存模型


# 测试
def eval():
    rl.restore() # 取模型
    env.render()
    env.viewer.set_vsync(True)
    while True:
        s = env.reset()
        for _ in range(200):
            env.render()
            a = rl.choose_action(s)
            s, r, done = env.step(a)
            if done:
                break


if ON_TRAIN:
    train()
else:
    eval()



