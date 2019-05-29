# 问题描述与项目流程
## 1.问题描述
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190529100945788.png#pic_center)  
在该项目中，你将使用强化学习算法（本文使用的**Q-Learning**），实现一个自动走迷宫的机器人。
1. 如上图所示，机器人初始位置在地图左上角。在我们的迷宫中，有**墙壁（黑色方块）**、**炸弹（黄色圆块）**及**终点（绿色方块）**。机器人要尽可能避开陷阱，并且拿到黄金后，以最少的步子到达终点。
2. 机器人可执行的动作包括：向左走 **L** 、向右走 **R** 、向上走 **U** 、向下走 **D** 。
3. 执行不同动作后，根据不同的情况会活动不同的奖励，具体而言，有以下几种情况： 
	- 走到墙壁： -10
	- 走到陷阱：- 30
	- 走到终点：+50

## 2.强化学习：算法理解
### 2.1强化学习总览
强化学习作为机器学习算法的一种，其模式也是让智能体在“训练”中学到“经验”，以实现给定的任务。但不同于监督学习与非监督学习，在强化学习的框架中，我们更侧重通过智能体与环境的交互来学习。通常在监督学习和非监督学习任务中，智能体往往需要通过给定的训练集，辅之以既定的训练目标（如最小化损失函数），通过给定的学习算法来实现这一目标。然而在强化学习中，智能体则是通过其与环境交互得到的奖励进行学习。这个环境可以是虚拟的（如虚拟的迷宫），也可以是真实的（自动驾驶汽车在真实道路上收集数据）。

在强化学习中有五个核心组成部分，它们分别是：**环境（Environment）、智能体（Agent）、状态（State）、动作（Action）和奖励（Reward）**。在某一时间节点 $t$：

* 智能体在从环境中感知其所处的状态 $s_t$
* 智能体根据某些准则选择动作 $a_t$
* 环境根据智能体选择的动作，向智能体反馈奖励 $r_{t+1}$

通过合理的学习算法，智能体将在这样的问题设置下，成功学到一个在状态 $s_t$ 选择动作 $a_t$ 的策略 $\pi (s_t) = a_t$。

### 2.2代码实现
#### 2.2.1 生成环境
首先，我们需要创造环境，一个让机器人能学习的环境。本文中也就是我们的迷宫。
	详细代码就不展示了，有兴趣的欢迎去我的[Github](https://github.com/StephenLouis/Reinforcement_Learning)查看完整代码。

```python
from maze_env import Maze

env = Maze()	#迷宫可以自定义，这里我们生成的是 5*5 的迷宫
env.mainloop()
```
![在这里插入图片描述](https://img-blog.csdnimg.cn/20190529100945788.png#pic_center)
#### 2.2.2 定义动作
&emsp;&emsp;接下来，定义机器人是如何选择行动的。这里需要引入增强学习中**epsilon greedy**的概念。因为在初始阶段, 随机的探索环境, 往往比固定的行为模式要好, 所以这也是累积经验的阶段, 我们希望探索者不会那么贪婪(greedy)。说说我的理解，上图迷宫中，当机器人第一次找到黄金后，如果不控制他的贪婪程度，那么很可能他每次都会直奔去，加入地图中还有第二个黄金，则很有可能被忽略（即缺少对地图的完全搜索）。
&emsp;&emsp;所以**epsilon**就是用来控制贪婪程度的值。**epsilon**可以随着探索时间不断提升(越来越贪婪), 不过在这个例子中, 我们就固定成 **epsilon** = 0.9, 90% 的时间是选择最优策略, 10% 的时间来探索。
```python
   def choose_action(self, observation):
        self.check_state_exist(observation)	# 检查这个state（状态）下的所有action值
        # action selection
        if np.random.uniform() < self.epsilon:	# 非贪婪模式
            # choose best action
            state_action = self.q_table.loc[observation, :]
            # some actions may have the same value, randomly choose on in these actions
            action = np.random.choice(state_action[state_action == np.max(state_action)].index)
        else:
            # choose random action
            action = np.random.choice(self.actions)	# 贪婪模式
        return action
```
#### 2.2.3 环境反馈 
当机器人做出行为后，环境也需要给行为一个反馈，也就是我们说的**惩罚**和**奖励**。环境需要反馈出下个  **state（S_ ）** 和上个 **state（S）**已经做出 **action（A）** 所获得的 **reward（R）** 当机器人撞到则惩罚 ，踩到陷阱则惩罚，进入终点则奖励。
```python
    def step(self, action):
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state

        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            s_ = 'terminal'
        elif s_ in [self.canvas.coords(self.barrier1),
                    self.canvas.coords(self.barrier2),
                    self.canvas.coords(self.barrier3),
                    self.canvas.coords(self.barrier4),
                    self.canvas.coords(self.barrier5),
                    self.canvas.coords(self.barrier6)]:
            reward = -1
            done = True
            s_ = 'terminal'
        else:
            reward = 0
            done = False

        return s_, reward, done
```

#### 2.2.4 环境更新
接下来是更新环境，代码较简单。

```python
    def learn(self, s, a, r, s_):
        self.check_state_exist(s_)
        q_predict = self.q_table.loc[s, a]
        if s_ != 'terminal':
            q_target = r + self.gamma * self.q_table.loc[s_, :].max()  # next state is not terminal
        else:
            q_target = r  # next state is terminal
        self.q_table.loc[s, a] += self.lr * (q_target - q_predict)  # update
```

#### 2.2.5 强化学习主循环
**最重要**的地方就在这里. 你定义的 RL 方法都在这里体现.。（最后大约**200个episode**能找到两条最短路径）
![Q—Learing 算法](https://img-blog.csdnimg.cn/20190529153412964.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L0dpbGdhbWU=,size_16,color_FFFFFF,t_70#pic_center)

```python
def update():
    for episode in range(100):
        # initial observation
        observation = env.reset()

        while True:
            # fresh env
            env.render()

            # RL choose action based on observation
            action = RL.choose_action(str(observation))

            # RL take action and get next observation and reward
            observation_, reward, done = env.step(action)

            # RL learn from this transition
            RL.learn(str(observation), action, reward, str(observation_))

            # swap observation
            observation = observation_

            # break while loop when end of this episode
            if done:
                break

    # end of game
    print('game over')
    env.destroy()
```
