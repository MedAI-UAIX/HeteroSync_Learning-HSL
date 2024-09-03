1. The core codes of each learning method are presented. To utilize these methods effectively, the core codes need to be implemented in a distributed manner.

2. In each distributed node, the name of the node where the local data resides is listed first in the node_list. 
For example, in an experiment with five nodes: 
    for Node 1, its node_list is ['Node1', 'Node2', 'Node3', 'Node4', 'Node5']; 
    for Node 2, its node_list is ['Node2', 'Node1', 'Node3', 'Node4', 'Node5']; 
    for Node 5, its node_list is ['Node5', 'Node1', 'Node2', 'Node3', 'Node4']. 
This approach ensures that node_list[0] in the code represents the node where the local data is located.

3. We use a simple parameter averaging method to combine model weights from each node:
    weight= ((weight_1+weight_2+⋯+weight_i))/i

# Code example
dict0=torch.load(swarm_dir+'/_{}_dict_best.pkl'.format(node_list[0]))
for name in dict0:
    for node in node_list[1:]:
        dict0[name]+=torch.load(swarm_dir+'/_{}_dict_best.pkl'.format(node))[name]
    dict0[name]/=len(node_list)

4. For Personalized Learning, besides the current node's model weight contributing 50%, the weights from all other nodes collectively contribute another 50%, and the fusion of these weights across all nodes still follows the averaging method.
    weight= (weight_1*0.5+(weight_2+⋯+weight_i)/(i-1)*0.5

# Code example
dict0=torch.load(swarm_dir+'/_{}_dict_best.pkl'.format(node_list[0]))
for name in dict0:
    dict0[name]/=0.5
    for node in node_list[1:]:
        dict0[name]+=torch.load(swarm_dir+'/_{}_dict_best.pkl'.format(node))[name]*0.5/(len(node_list)-1)
