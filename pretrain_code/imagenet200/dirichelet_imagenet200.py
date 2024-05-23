# -*- coding: gbk -*-
import  numpy as np
import os
import json
import sys
import matplotlib.pyplot as plt
import random
import copy

alpha = 1.0
persudo_num_clients = 30
num_clients = 8
#num_per_class = 500
classes = 200

merge_client_list = [[4 for _ in range(3)],
                     [4 for _ in range(3)],
                     [1 for _ in range(1)],
                     [5 for _ in range(1)]]
random.seed(10)
np.random.seed(10)

def dirichlet_split_iid(train_labels, n_clients):
    
    n_classes = train_labels.max() + 1
    
    l = [1 / n_clients for _ in range(n_clients)] 
    label_distribution = np.array([l for _ in range(n_classes)]) 
    
    #
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]
    for class_idc in class_idcs:
        random.shuffle(class_idc) 

    #
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def dirichlet_split_noniid(train_labels, alpha, n_clients):
    
    n_classes = train_labels.max()+1
    
    label_distribution = np.random.dirichlet([alpha]*n_clients, n_classes)
    
    class_idcs = [np.argwhere(train_labels == y).flatten()
                  for y in range(n_classes)]

    
    client_idcs = [[] for _ in range(n_clients)]
    for k_idcs, fracs in zip(class_idcs, label_distribution):
        # 
        for i, idcs in enumerate(np.split(k_idcs,
                                          (np.cumsum(fracs)[:-1]*len(k_idcs)).
                                          astype(int))):
            client_idcs[i] += [idcs]

    client_idcs = [np.concatenate(idcs) for idcs in client_idcs]

    return client_idcs

def compute_index(index, class_counts):
    total = 0
    for i, count in enumerate(class_counts):
        total += count
        if index < total:
            return i, (index-(total-count))
    return -1 

raw_labels = os.listdir("/home/***/code/src/code and baselines/dataset/imagenet2012/imagenet200/train")
#print(raw_labels)

labels_dict = {}
for i in range(len(raw_labels)):
    labels_dict[i] = raw_labels[i]

# 
folder_path = "/home/***/code/src/code and baselines/dataset/imagenet2012/imagenet200/train"

# 
class_counts = [len(os.listdir(os.path.join(folder_path, folder))) for folder in os.listdir(folder_path)]

file_names = [[file for file in os.listdir(os.path.join(folder_path, subfolder))] for subfolder in os.listdir(folder_path)]
#print(class_counts)


labels = [j for j in range(classes) for num in range(class_counts[j])]
'''
labels = []
for j in range(classes):
    for num in range(class_counts[j]):
        labels.append(j)'''

print(len(labels))
print(labels[0])
print(labels[1])

labels = np.array(labels)

'''client_indexs = dirichlet_split_noniid(labels, alpha, num_clients)
#client_indexs = dirichlet_split_iid(labels, persudo_num_clients)
client_indexs = [i.tolist() for i in client_indexs]
client_indexs_cpy = copy.deepcopy(client_indexs)'''

client_indexs = dirichlet_split_noniid(labels, alpha, persudo_num_clients)
client_indexs = [i.tolist() for i in client_indexs]
client_indexs_cpy = [[] for i in range(num_clients)]
cnt = 0
start_cnt = 0
for i in range(len(merge_client_list)):
    for j in range(len(merge_client_list[i])):
        for k in range(merge_client_list[i][j]):
            client_indexs_cpy[cnt] += client_indexs[start_cnt]
            start_cnt += 1
        cnt+=1
for i in range(num_clients):
    print(len(client_indexs_cpy[i]))

'''plt.figure(figsize=(12, 8))
plt.hist([labels[idc]for idc in client_indexs], stacked=True,
            bins=np.arange(min(labels)-0.5, max(labels) + 1.5, 1),
            label=["Client {}".format(i) for i in range(len(client_indexs))],
            rwidth=0.5)
plt.xticks(np.arange(classes))
plt.xlabel("Label type")
plt.ylabel("Number of samples")
plt.legend(loc="upper right")
plt.title("Display Label Distribution on Different Clients")
plt.show()'''


for i in range(len(client_indexs_cpy)):
    for j in range(len(client_indexs_cpy[i])):
        client_i, client_j = compute_index(client_indexs_cpy[i][j], class_counts)
        client_i_folder = labels_dict[client_i]
        file_name = "./" + client_i_folder + "/" + str(file_names[client_i][client_j])
        client_indexs_cpy[i][j] = file_name

client_indexs_cpy = dict(zip(list(range(num_clients)), client_indexs_cpy))
#print(client_indexs)
np.save("/home/***/code/src/code and baselines/dataset/imagenet2012/imagenet200/index_"+str(num_clients)+"_alpha"+str(alpha)+".npy", client_indexs_cpy)
