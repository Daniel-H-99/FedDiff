import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

# 1. 기본 스타일 설정
plt.style.use('default')
plt.rcParams['figure.figsize'] = (4, 3)
plt.rcParams['font.size'] = 12

# 2. 데이터 준비
np.random.seed(0)

# with open('/home/server36/minyeong_workspace/FL-bench/tested_privacy2_vqfed_cifar10_niid3_client_0.pkl', 'rb') as f:
with open('/home/server36/minyeong_workspace/FL-bench/tested_privacy2_phoenix_cifar10_niid3_client_0.pkl', 'rb') as f:
    data = pkl.load(f)
res = list(data.values())[0]

local_train_list = []
other_train_list = []
global_test_list = []
train_test_list = []
local_other_list = []
for client_id in range(5):
    local_train = res[client_id][f'local_train_client_{client_id}']
    other_train = res[client_id][f'other_train_client_{client_id}']
    global_test = res[client_id][f'global_test_client_{client_id}']
    train_test = global_test / local_train.clip(min=1e-6)
    local_other = other_train / local_train.clip(min=1e-6)
    local_train_list.append(local_train)
    other_train_list.append(other_train)
    global_test_list.append(global_test)
    train_test_list.append(train_test)
    local_other_list.append(local_other)

local_train_list = np.concatenate(local_train_list)
other_train_list = np.concatenate(other_train_list)
global_test_list = np.concatenate(global_test_list)
train_test_list = np.concatenate(train_test_list)
local_other_list = np.concatenate(local_other_list)

print(f'local_train_avg : {local_train_list.mean()}')
print(f'other_train_avg : {other_train_list.mean()}')
print(f'global_test_avg : {global_test_list.mean()}')
print(f'train_test_avg : {train_test_list.mean()}')
print(f'local_other_avg : {local_other_list.mean()}')


# # 3. 그래프 그리기
fig, ax = plt.subplots()

# ax.boxplot([local_train_list, other_train_list, global_test_list])
ax.boxplot([train_test_list])
ax.set_ylim(0, 1.5)
ax.set_xlabel('Data Type')
ax.set_ylabel('Value')
plt.savefig('[Phoenix] Distances.png')
# plt.show()