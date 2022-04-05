# MDP

## 理论基础

### Markov Process(MP)

#### Markov Property（马尔可夫性）

即未来的演变(将来)不依赖于它以往的演变(过去):

对于状态历史**$h_{t}=\left\{s_{1}, s_{2}, s_{3}, \ldots, s_{t}\right\}$**，如果状态转移有马尔可夫性，即满足：
$$
p\left(s_{t+1} \mid s_{t}\right) =p\left(s_{t+1} \mid h_{t}\right) \tag{1}
$$

$$
p\left(s_{t+1} \mid s_{t}, a_{t}\right) =p\left(s_{t+1} \mid h_{t}, a_{t}\right) \tag{2}
$$

可理解为$s_{t+1}$只与$s_t$有关，而与之前的所有历史状态无关



#### Markov Process/Markov Chain（MP）

![2.5](https://gitee.com/ftfwjft/images/raw/master/image/cloud/2.5.png)

图中有四个状态，四个状态之间可以互相转移，比如从 $s_1$ 开始，

* $s_1$ 有 0.1 的概率继续存活在 $s_1$ 状态，

* 有 0.2 的概率转移到 $s_2$， 

* 有 0.7 的概率转移到 $s_4$ 。

  

可以用用`状态转移矩阵(State Transition Matrix)` $P$ 来描述状态转移 $p\left(s_{t+1}=s^{\prime} \mid s_{t}=s\right)$，如下式所示。
$$
P=\left[\begin{array}{cccc}
P\left(s_{1} \mid s_{1}\right) & P\left(s_{2} \mid s_{1}\right) & \ldots & P\left(s_{N} \mid s_{1}\right) \\
P\left(s_{1} \mid s_{2}\right) & P\left(s_{2} \mid s_{2}\right) & \ldots & P\left(s_{N} \mid s_{2}\right) \\
\vdots & \vdots & \ddots & \vdots \\
P\left(s_{1} \mid s_{N}\right) & P\left(s_{2} \mid s_{N}\right) & \ldots & P\left(s_{N} \mid s_{N}\right)
\end{array}\right]
$$
状态转移矩阵类似于一个 conditional probability，当我们知道当前我们在 $s_t$ 这个状态过后，到达下面所有状态的一个概念。所以它每一行其实描述了是从一个节点到达所有其它节点的概率

给定了马尔科夫链和状态转移矩阵后，我们可以对链进行采样，得到一串`轨迹（trajectory）`

如从$s_{1}$采样，可得轨迹：

- $s_{1}, s_{4}, s_{3}, s_2, s_1$
- $s_1, s_2, s_1$

通过对这个状态的采样，我们可以获得很多这样的轨迹



### Markov Reward Process(MRP)

**`马尔可夫奖励过程(Markov Reward Process, MRP)` 是马尔可夫链再加上了一个奖励函数。**在 MRP 中，转移矩阵和状态都是跟马尔可夫链一样的，多了一个`奖励函数(reward function)`。**奖励函数 $R$ 是一个期望**，就是说当你到达某一个状态的时候，可以获得多大的奖励，然后这里另外定义了一个 discount factor $\gamma$ 。如果状态数是有限的，$R$ 可以是一个向量。

简单点说就是从一个状态转移到另一个状态时，可能获得一些奖励



#### Return and Value function

这里我们进一步定义一些概念。

*  `Horizon` 是指一个回合的长度（每个回合最大的时间步数），它是由有限个步数决定的。

*  `Return(回报)` 说的是把奖励进行折扣后所获得的收益。Return 可以定义为奖励的逐步叠加，如下式所示：

$$
G_{t}=R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\gamma^{3} R_{t+4}+\ldots+\gamma^{T-t-1} R_{T}
$$

这里有一个叠加系数，越往后得到的奖励，折扣得越多。这说明我们其实更希望得到现有的奖励，未来的奖励就要把它打折扣。

* 当我们有了 return 过后，就可以定义一个状态的价值了，就是 `state value function`。对于 MRP，state value function 被定义成是 return 的期望，如下式所示：

$$
\begin{aligned}
V_{t}(s) &=\mathbb{E}\left[G_{t} \mid s_{t}=s\right] \\
&=\mathbb{E}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\ldots+\gamma^{T-t-1} R_{T} \mid s_{t}=s\right]
\end{aligned}
$$

表示当前状态的价值$V(s)$的期望表示为**未来可能获得的折扣价值之和**



需要**discount factor** $\gamma$的理由：

- 有些马尔可夫过程是带环的，它并没有终结，我们想避免这个无穷的奖励。
- 我们并没有建立一个完美的模拟环境的模型，也就是说，我们对未来的评估**不一定是准确**的，我们不一定完全信任我们的模型，因为这种不确定性，所以我们对未来的预估增加一个折扣。我们想把这个不确定性表示出来，希望尽可能快地得到奖励，而不是在未来某一个点得到奖励。
- 如果这个奖励是有实际价值的，我们可能是更希望**立刻**就得到奖励，而不是后面再得到奖励（现在的钱比以后的钱更有价值）。
- 在人的行为里面来说的话，大家也是想得到即时奖励。
- 有些时候可以把这个系数设为 0，$\gamma=0$：我们就只关注了它当前的奖励。我们也可以把它设为 1，$\gamma=1$：对未来并没有折扣，未来获得的奖励跟当前获得的奖励是一样的。

Discount factor 可以作为强化学习 agent 的一个超参数来进行调整，然后就会得到不同行为的 agent。



#### Bellman Equation

$$
V(s)=\underbrace{R(s)}_{\text {Immediate reward }}+\underbrace{\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)}_{\text {Discounted sum of future reward }}
$$

- $s'$可看作从状态$s$出发未来的所有状态

- 转移概率$P(s'|s)$  是指从当前状态转移到未来状态的概率。

- $V(s')$ 代表的是未来某一个状态的价值。我们从当前这个位置开始，有一定的概率去到未来的所有状态，所以我们要把这个概率也写上去，这个转移矩阵也写上去，然后我们就得到了未来状态，然后再乘以一个 $\gamma$，这样就可以把未来的奖励打折扣。

- 第二部分可以看成是未来奖励的折扣总和(Discounted sum of future reward)。

  

##### Bellman Equation Derivation

推导如下：
$$
\begin{aligned}
V(s)&=\mathbb{E}\left[G_{t} \mid s_{t}=s\right]\\
&=\mathbb{E}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\ldots \mid s_{t}=s\right]  \\
&=\mathbb{E}\left[R_{t+1}|s_t=s\right] +\gamma \mathbb{E}\left[R_{t+2}+\gamma R_{t+3}+\gamma^{2} R_{t+4}+\ldots \mid s_{t}=s\right]\\
&=R(s)+\gamma \mathbb{E}[G_{t+1}|s_t=s] \\
&=R(s)+\gamma \mathbb{E}[V(s_{t+1})|s_t=s]\\
&=R(s)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s\right) V\left(s^{\prime}\right)
\end{aligned}
$$
当状态转移矩阵已知时，可将**Bellman Equation**写为矩阵形式：
$$
\left[\begin{array}{c}
V\left(s_{1}\right) \\
V\left(s_{2}\right) \\
\vdots \\
V\left(s_{N}\right)
\end{array}\right]=\left[\begin{array}{c}
R\left(s_{1}\right) \\
R\left(s_{2}\right) \\
\vdots \\
R\left(s_{N}\right)
\end{array}\right]+\gamma\left[\begin{array}{cccc}
P\left(s_{1} \mid s_{1}\right) & P\left(s_{2} \mid s_{1}\right) & \ldots & P\left(s_{N} \mid s_{1}\right) \\
P\left(s_{1} \mid s_{2}\right) & P\left(s_{2} \mid s_{2}\right) & \ldots & P\left(s_{N} \mid s_{2}\right) \\
\vdots & \vdots & \ddots & \vdots \\
P\left(s_{1} \mid s_{N}\right) & P\left(s_{2} \mid s_{N}\right) & \ldots & P\left(s_{N} \mid s_{N}\right)
\end{array}\right]\left[\begin{array}{c}
V\left(s_{1}\right) \\
V\left(s_{2}\right) \\
\vdots \\
V\left(s_{N}\right)
\end{array}\right]
$$
可以直接求解：
$$
\begin{aligned}
V &= R+ \gamma PV \\
IV &= R+ \gamma PV \\
(I-\gamma P)V &=R \\
V&=(I-\gamma P)^{-1}R
\end{aligned}
$$
我们可以通过矩阵求逆把这个 V 的这个价值直接求出来。但是一个问题是这个矩阵求逆的过程的复杂度是 $O(N^3)$。所以当状态非常多的时候，比如说从十个状态到一千个状态，到一百万个状态。那么当我们有一百万个状态的时候，这个转移矩阵就会是个一百万乘以一百万的矩阵，这样一个大矩阵的话求逆是非常困难的，**所以这种通过解析解去求解的方法只适用于很小量的 MRP。**



#### Iterative Algorithm for Computing Value of a MRP

接下来我们来求解这个价值函数。**我们可以通过迭代的方法来解这种状态非常多的 MRP(large MRPs)，**比如说：

* 动态规划的方法，

* 蒙特卡罗的办法(通过采样的办法去计算它)，

* 时序差分学习(Temporal-Difference Learning)的办法。 `Temporal-Difference Learning` 叫 `TD Leanring`，它是动态规划和蒙特卡罗的一个结合。

  

##### 蒙特卡罗(Monte Carlo)方法

![2.16](https://gitee.com/ftfwjft/images/raw/master/image/cloud/2.16.png)

对于一个MRP，我们可以从一个状态开始，不断进行采样，获得很多串轨迹，然后可以计算每串轨迹的$G_t$，经过多次采样后，使用累计的$G_t$除以采样次数，就可以得到该状态的$V(s)$



##### 动态规划

![2.17](https://gitee.com/ftfwjft/images/raw/master/image/cloud/2.17.png)

对于一个MRP，一直去迭代它的 Bellman equation，让它最后收敛，即最后更新的状态跟你上一个状态变化并不大的时候，更新就可以停止，我们就可以得到它的一个状态价值。

动态规划的方法基于后继状态值的估计来更新状态值的估计（算法二中的第 3 行用 $V'$ 来更新 $V$ ）。也就是说，它们根据其他估算值来更新估算值。我们称这种基本思想为 bootstrapping。



### Markov Decision Process(MDP)

**相对于 MRP，`马尔可夫决策过程(Markov Decision Process)`多了一个 `decision`，其它的定义跟 MRP 都是类似的**:

* 这里多了一个决策，多了一个动作。
* 状态转移也多了一个条件，变成了 $P\left(s_{t+1}=s^{\prime} \mid s_{t}=s, a_{t}=a\right)$。你采取某一种动作，然后你未来的状态会不同。未来的状态不仅是依赖于你当前的状态，也依赖于在当前状态 agent 采取的这个动作。
* 对于这个价值函数，它也是多了一个条件，多了一个你当前的这个动作，变成了 $R\left(s_{t}=s, a_{t}=a\right)=\mathbb{E}\left[r_{t} \mid s_{t}=s, a_{t}=a\right]$。你当前的状态以及你采取的动作会决定你在当前可能得到的奖励多少。

#### Policy

- Policy定义在某一状态应采取什么样的动作
- 知道当前状态过后，我们可以把当前状态带入 policy function，然后就会得到一个概率，即： 

$$
\pi(a \mid s)=P\left(a_{t}=a \mid s_{t}=s\right)
$$

概率就代表了在所有可能的动作里面怎样采取行动，比如可能有 0.7 的概率往左走，有 0.3 的概率往右走，这是一个概率的表示。

* 另外这个策略也可能是**确定**的，它有可能是直接输出一个值。或者就直接告诉你当前应该采取什么样的动作，而不是一个动作的概率。

* 假设这个概率函数应该是稳定的(stationary)，不同时间点，你采取的动作其实都是对这个 policy function 进行采样。

我们可以将 MRP 转换成 MDP。已知一个 MDP 和一个 policy $\pi$ 的时候，我们可以把 MDP 转换成 MRP。

在 MDP 里面，转移函数 $P(s'|s,a)$  是基于它当前状态以及它当前的 action。因为我们现在已知它 policy function，就是说在每一个状态，我们知道它可能采取的动作的概率，那么就可以直接把这个 action 进行加和，直接把这个 a 去掉，那我们就可以得到对于 MRP 的一个转移，这里就没有 action。
$$
 P^{\pi}\left(s^{\prime} \mid s\right)=\sum_{a \in A} \pi(a \mid s) P\left(s^{\prime} \mid s, a\right)
$$

对于这个奖励函数，我们也可以把 action 拿掉，这样就会得到一个类似于 MRP 的奖励函数。

$$
R^{\pi}(s)=\sum_{a \in A} \pi(a \mid s) R(s, a)
$$

#### Comparison of MP/MRP and MDP

![](https://gitee.com/ftfwjft/images/raw/master/image/cloud/2.21.png)



**这里我们看一看，MDP 里面的状态转移跟 MRP 以及 MP 的一个差异。**

* 马尔可夫过程的转移是直接就决定。比如当前状态是 s，那么就直接通过这个转移概率决定了下一个状态是什么。
* 但对于 MDP，它的中间多了一层这个动作 a ，就是说在你当前这个状态的时候，首先要决定的是采取某一种动作，那么你会到了某一个黑色的节点。到了这个黑色的节点，因为你有一定的不确定性，当你当前状态决定过后以及你当前采取的动作过后，你到未来的状态其实也是一个概率分布。**所以在这个当前状态跟未来状态转移过程中这里多了一层决策性，这是 MDP 跟之前的马尔可夫过程很不同的一个地方。**在马尔可夫决策过程中，动作是由 agent 决定，所以多了一个 component，agent 会采取动作来决定未来的状态转移。

#### Value function for MDP

顺着 MDP 的定义，我们可以把 `状态-价值函数(state-value function)`，就是在 MDP 里面的价值函数也进行一个定义，它的定义是跟 MRP 是类似的，如式 (3)  所示：
$$
v^{\pi}(s)=\mathbb{E}_{\pi}\left[G_{t} \mid s_{t}=s\right] 
$$
但是这里期望与决策相关，就是这个期望是基于你采取的这个 policy ，就当你的 policy 决定过后，**我们通过对这个 policy 进行采样来得到一个期望，那么就可以计算出它的这个价值函数。**

这里我们另外引入了一个 `Q 函数(Q-function)`。Q 函数也被称为 `action-value function`。**Q 函数定义的是在某一个状态采取某一个动作，它有可能得到的这个 return 的一个期望**，如式 (4) 所示：
$$
q^{\pi}(s, a)=\mathbb{E}_{\pi}\left[G_{t} \mid s_{t}=s, A_{t}=a\right]
$$
这里期望其实也是 over policy function。所以你需要对这个 policy function 进行一个加和，然后得到它的这个价值。
**对 Q 函数中的动作函数进行加和，就可以得到价值函数**，如式 (5) 所示：
$$
v^{\pi}(s)=\sum_{a \in A} \pi(a \mid s) q^{\pi}(s, a)
$$

#### Q-function Bellman Equation

此处我们给出 Q 函数的 Bellman equation：

$$
\begin{aligned}
q(s,a)&=\mathbb{E}\left[G_{t} \mid s_{t}=s,a_{t}=a\right]\\
&=\mathbb{E}\left[R_{t+1}+\gamma R_{t+2}+\gamma^{2} R_{t+3}+\ldots \mid s_{t}=s,a_{t}=a\right]  \\
&=\mathbb{E}\left[R_{t+1}|s_{t}=s,a_{t}=a\right] +\gamma \mathbb{E}\left[R_{t+2}+\gamma R_{t+3}+\gamma^{2} R_{t+4}+\ldots \mid s_{t}=s,a_{t}=a\right]\\
&=R(s,a)+\gamma \mathbb{E}[G_{t+1}|s_{t}=s,a_{t}=a] \\
&=R(s,a)+\gamma \mathbb{E}[V(s_{t+1})|s_{t}=s,a_{t}=a]\\
&=R(s,a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s,a\right) V\left(s^{\prime}\right)
\end{aligned}
$$

### Bellman Expectation Equation

通过对状态-价值函数进行一个分解，我们就可以得到一个类似于之前 MRP 的 Bellman Equation，这里叫 `Bellman Expectation Equation`，如：
$$
v^{\pi}(s)=E_{\pi}\left[R_{t+1}+\gamma v^{\pi}\left(s_{t+1}\right) \mid s_{t}=s\right]
$$
对于 Q 函数，我们也可以做类似的分解，也可以得到 Q 函数的 Bellman Expectation Equation，如：
$$
q^{\pi}(s, a)=E_{\pi}\left[R_{t+1}+\gamma q^{\pi}\left(s_{t+1}, A_{t+1}\right) \mid s_{t}=s, A_{t}=a\right]
$$
可证得：
$$
q^{\pi}(s, a)=R_{s}^{a}+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) v^{\pi}\left(s^{\prime}\right)
$$
代入$v^{\pi}(s)$定义：
$$
v^{\pi}(s)=\sum_{a \in A} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) v^{\pi}\left(s^{\prime}\right)\right)
$$
代表了当前状态的价值跟未来状态价值之间的一个关联，同理：
$$
q^{\pi}(s, a)=R(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) \sum_{a^{\prime} \in A} \pi\left(a^{\prime} \mid s^{\prime}\right) q^{\pi}\left(s^{\prime}, a^{\prime}\right) \tag{11}
$$
代表了当前时刻的 Q 函数跟未来时刻的 Q 函数之间的一个关联，上两式是 Bellman expectation equation 的另一种形式。

![2.25](https://cdn.jsdelivr.net/gh/ftfwjft/clouding/data/2.25.png)
$$
v^{\pi}(s)=\sum_{a \in A} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) v^{\pi}\left(s^{\prime}\right)\right) \tag{12}
$$
![2.26](https://cdn.jsdelivr.net/gh/ftfwjft/clouding/data/2.26.png)

### **Policy Evaluation(Prediction)**

当我们知道一个 MDP 以及要采取的策略 $\pi$ ，计算价值函数 $v^{\pi}(s)$ 的过程就是 `policy evaluation`。就像我们在评估这个策略，我们会得到多大的奖励。

![2.28](https://cdn.jsdelivr.net/gh/ftfwjft/clouding/data/2.28.png)

![2.29](https://cdn.jsdelivr.net/gh/ftfwjft/clouding/data/2.29.png)

假设$P(Left)=0.5 \ and \ P(Right)=0.5$,如果折扣因子是 0.5，我们可以通过下面这个等式进行迭代:
$$
v_{t}^{\pi}(s)=\sum_{a} P(\pi(s)=a)\left(r(s, a)+\gamma \sum_{s^{\prime} \in S} P\left(s^{\prime} \mid s, a\right) v_{t-1}^{\pi}\left(s^{\prime}\right)\right)
$$
MDP 的 `prediction` 和 `control` 是 MDP 里面的核心问题。

* 预测问题：
  * 输入：MDP $<S,A,P,R,\gamma>$ 和 policy $\pi$  或者 MRP $<S,P^{\pi},R^{\pi},\gamma>$。
  * 输出：value function $v^{\pi}$。
  * Prediction 是说给定一个 MDP 以及一个 policy $\pi$ ，去计算它的 value function，就对于每个状态，它的价值函数是多少。

* 控制问题：
  * 输入：MDP  $<S,A,P,R,\gamma>$。
  * 输出：最佳价值函数(optimal value function) $v^*$ 和最佳策略(optimal policy) $\pi^*$。
  * Control 就是说我们去寻找一个最佳的策略，然后同时输出它的最佳价值函数以及最佳策略。
* 在 MDP 里面，prediction 和 control 都可以通过动态规划去解决。
* 要强调的是，这两者的区别就在于，
  * 预测问题是**给定一个 policy**，我们要确定它的 value function 是多少。
  * 而控制问题是在**没有 policy 的前提下**，我们要确定最优的 value function 以及对应的决策方案。
* **实际上，这两者是递进的关系，在强化学习中，我们通过解决预测问题，进而解决控制问题。**



Policy evaluation 的核心思想就是把如下式所示的 Bellman expectation backup 拿出来反复迭代，然后就会得到一个收敛的价值函数的值。
$$
v_{t+1}(s)=\sum_{a \in \mathcal{A}} \pi(a \mid s)\left(R(s, a)+\gamma \sum_{s^{\prime} \in \mathcal{S}} P\left(s^{\prime} \mid s, a\right) v_{t}\left(s^{\prime}\right)\right) \tag{15}
$$

### MDP Control

Policy evaluation 是说给定一个 MDP 和一个 policy，我们可以估算出它的价值函数。**还有问题是说如果我们只有一个 MDP，如何去寻找一个最佳的策略，然后可以得到一个`最佳价值函数(Optimal Value Function)`。**

Optimal Value Function 的定义如下式所示：
$$
v^{*}(s)=\max _{\pi} v^{\pi}(s)
$$
Optimal Value Function 是说，我们去搜索一种 policy $\pi$ 来让每个状态的价值最大。$v^*$ 就是到达每一个状态，它的值的极大化情况。

在这种极大化情况上面，我们得到的策略就可以说它是`最佳策略(optimal policy)`，如下式所示：
$$
\pi^{*}(s)=\underset{\pi}{\arg \max }~ v^{\pi}(s)
$$
Optimal policy 使得每个状态的价值函数都取得最大值。所以如果我们可以得到一个 optimal value function，就可以说某一个 MDP 的环境被解。在这种情况下，它的最佳的价值函数是一致的，就它达到的这个上限的值是一致的，但这里可能有多个最佳的 policy，就是说多个 policy 可以取得相同的最佳价值。

![image-20220330184706173](https://cdn.jsdelivr.net/gh/ftfwjft/clouding/data/image-20220330184706173.png)

Q: 怎么去寻找这个最佳的 policy ？

A: 当取得最佳的价值函数过后，我们可以通过对这个 Q 函数进行极大化，然后得到最佳策略。当所有东西都收敛过后，因为 Q 函数是关于状态跟动作的一个函数，所以在某一个状态采取一个动作，可以使得这个 Q 函数最大化，那么这个动作就应该是最佳的动作。所以如果我们能优化出一个 Q 函数，就可以直接在这个 Q 函数上面取一个让 Q 函数最大化的 action 的值，就可以提取出它的最佳策略。



**搜索最佳策略有两种常用的方法：policy iteration 和  value iteration**。

#### **Policy iteration **

**由两个步骤组成：policy evaluation 和 policy improvement。**

* **第一个步骤是 policy evaluation**，当前我们在优化这个 policy $\pi$，在优化过程中得到一个最新的 policy。我们先保证这个 policy 不变，然后去估计它出来的这个价值。给定当前的 policy function 来估计这个 v 函数。
* **第二个步骤是 policy improvement**，得到 v 函数过后，我们可以进一步推算出它的 Q 函数。得到 Q 函数过后，我们直接在 Q 函数上面取极大化，通过在这个 Q 函数上面做一个贪心的搜索来进一步改进它的策略。
* 这两个步骤就一直是在迭代进行，所以在 policy iteration 里面，在初始化的时候，我们有一个初始化的 $V$ 和 $\pi$ ，然后就是在这两个过程之间迭代。
* 左边这幅图上面的线就是我们当前 v 的值，下面的线是 policy 的值。
  * 跟踢皮球一样，我们先给定当前已有的这个 policy function，然后去算它的 v。
  * 算出 v 过后，我们会得到一个 Q 函数。Q 函数我们采取 greedy 的策略，这样就像踢皮球，踢回这个 policy 。
  * 然后进一步改进那个 policy ，得到一个改进的 policy 过后，它还不是最佳的，我们再进行 policy evaluation，然后又会得到一个新的 value function。基于这个新的 value function 再进行 Q 函数的极大化，这样就逐渐迭代，然后就会得到收敛。

![image-20220330184752998](https://cdn.jsdelivr.net/gh/ftfwjft/clouding/data/image-20220330184752998.png)

**你可以把 Q 函数看成一个 Q-table:**

* 横轴是它的所有状态，
* 纵轴是它的可能的 action。

得到 Q 函数后，`Q-table`也就得到了。

那么对于某一个状态，每一列里面我们会取最大的那个值，最大值对应的那个 action 就是它现在应该采取的 action。所以 arg max 操作就说在每个状态里面采取一个 action，这个 action 是能使这一列的 Q 最大化的那个动作。

当一直在采取 arg max 操作的时候，我们会得到一个单调的递增。通过采取这种 greedy，即 arg max 操作，我们就会得到更好的或者不变的 policy，而不会使它这个价值函数变差。所以当这个改进停止过后，我们就会得到一个最佳策略。

当改进停止过后，我们取它最大化的这个 action，它直接就会变成它的价值函数，如下式所示：
$$
q^{\pi}\left(s, \pi^{\prime}(s)\right)=\max _{a \in \mathcal{A}} q^{\pi}(s, a)=q^{\pi}(s, \pi(s))=v^{\pi}(s)
$$
所以我们有了一个新的等式：
$$
v^{\pi}(s)=\max _{a \in \mathcal{A}} q^{\pi}(s, a)
$$
上式被称为  `Bellman optimality equation`。从直觉上讲，Bellman optimality equation 表达了这样一个事实：最佳策略下的一个状态的价值必须等于在这个状态下采取最好动作得到的回报的期望。 

**当 MDP 满足 Bellman optimality equation 的时候，整个 MDP 已经到达最佳的状态。**它到达最佳状态过后，对于这个 Q 函数，取它最大的 action 的那个值，就是直接等于它的最佳的 value function。只有当整个状态已经收敛过后，得到一个最佳的 policy 的时候，这个条件才是满足的。

![image-20220330184908499](https://cdn.jsdelivr.net/gh/ftfwjft/clouding/data/image-20220330184908499.png)

#### **Value Iteration**

**Value iteration 就是把 Bellman Optimality Equation 当成一个 update rule 来进行，**如下式所示：
$$
v(s) \leftarrow \max _{a \in \mathcal{A}}\left(R(s, a)+\gamma \sum_{s^{\prime} \in \mathcal{S}} P\left(s^{\prime} \mid s, a\right) v\left(s^{\prime}\right)\right)
$$
之前我们说上面这个等式只有当整个 MDP 已经到达最佳的状态时才满足。但这里可以把它转换成一个 backup 的等式。Backup 就是说一个迭代的等式。**我们不停地去迭代 Bellman Optimality Equation，到了最后，它能逐渐趋向于最佳的策略，这是 value iteration 算法的精髓。**

为了得到最佳的 $v^*$ ，对于每个状态的 $v^*$，我们直接把这个 Bellman Optimality Equation 进行迭代，迭代了很多次之后，它就会收敛。

![image-20220330185045164](https://cdn.jsdelivr.net/gh/ftfwjft/clouding/data/image-20220330185045164.png)

* 我们使用 value iteration 算法是为了得到一个最佳的策略。
* 解法：我们可以直接把 `Bellman Optimality backup` 这个等式拿进来进行迭代，迭代很多次，收敛过后得到的那个值就是它的最佳的值。
* 这个算法开始的时候，它是先把所有值初始化，通过每一个状态，然后它会进行这个迭代。把等式 (22) 插到等式 (23) 里面，就是 Bellman optimality backup 的那个等式。有了等式 (22) 和等式 (23) 过后，然后进行不停地迭代，迭代过后，然后收敛，收敛后就会得到这个 $v^*$ 。当我们有了 $v^*$ 过后，一个问题是如何进一步推算出它的最佳策略。
* 提取最佳策略的话，我们可以直接用 arg max。就先把它的 Q 函数重构出来，重构出来过后，每一个列对应的最大的那个 action 就是它现在的最佳策略。这样就可以从最佳价值函数里面提取出最佳策略。
* 我们只是在解决一个 planning 的问题，而不是强化学习的问题，因为我们知道环境如何变化。



### Summary for Prediction and Control in MDP

![image-20220330185129665](https://cdn.jsdelivr.net/gh/ftfwjft/clouding/data/image-20220330185129665.png)

总结如上表所示，就对于 MDP 里面的 prediction 和 control  都是用动态规划来解，我们其实采取了**不同的** **Bellman Equation。**

* 如果是一个 prediction 的问题，即 policy evaluation  的问题，直接就是不停地 run 这个 Bellman Expectation Equation，这样我们就可以去估计出给定的这个策略，然后得到价值函数。
* 对于 control，
  * 如果采取的算法是 policy  iteration，那这里用的是 Bellman Expectation Equation 。把它分成两步，先上它的这个价值函数，再去优化它的策略，然后不停迭代。这里用到的只是 Bellman Expectation Equation。
  * 如果采取的算法是 value iteration，那这里用到的 Bellman Equation 就是 Bellman Optimality Equation，通过 arg max 这个过程，不停地去 arg max 它，最后它就会达到最优的状态。