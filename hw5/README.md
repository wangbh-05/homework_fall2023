# 1.Exploration

### 1.1 Random Policy

![1753963774420](image/README/1753963774420.png)

![1753963812470](image/README/1753963812470.png)

![1753963792063](image/README/1753963792063.png)

### 1.2 RND

![1753963861619](image/README/1753963861619.png)

![1753963870087](image/README/1753963870087.png)

![1753963875580](image/README/1753963875580.png)

# 2.Offline RL

### 2.1 CQL

#### 2.1.1 与DQN在easy和medium环境中对比：

![1753971809752](image/README/1753971809752.png)

![1753971910199](image/README/1753971910199.png)

#### 2.1.2 CQL不同alpha参数性能对比

![1754018264858](image/README/1754018264858.png)

alpha=0

![1754032081507](image/README/1754032081507.png)

alpha=0.1

![1754032095283](image/README/1754032095283.png)

#### 2.1.3 IQL & AWAC

环境难，数据集质量低，收集数据少，训练步数少，效果差

###### IQL

step=10000

![1754032528432](image/README/1754032528432.png)

###### AWAC

step=30000

![1754032577898](image/README/1754032577898.png)

#### 2.1.4 Data ablations

Run RND with total_steps 1000, 5000, 10000, and 20000 on Hard environment

use CQL agent compare difference sizes of dataset

rnd 20k:

![1754034261974](image/README/1754034261974.png)

![1754037682915](image/README/1754037682915.png)

rnd 50k:

![1754034753420](image/README/1754034753420.png)

![1754037696358](image/README/1754037696358.png)

compare 10k 20k 50k:

blue: 20k 	 red: 50k	green: 10k

![1754039720752](image/README/1754039720752.png)

### 3.Online Fine-Tuning

#### 3.1 CQL dataset: rnd 20k

![1754111060447](image/README/1754111060447.png)

![1754109713479](image/README/1754109713479.png)

#### 3.2 AWAC dataset:50k

![1754111023115](image/README/1754111023115.png)

![1754109718155](image/README/1754109718155.png)
