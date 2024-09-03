import torch
import torch.optim.lr_scheduler as lr_scheduler
from tools import load_HSL_train,load_HSL_test,train_HSL,test_HSL
from MMOE_ResNet18 import MMoE

#Local training
node_list=[] #all distributed nodes are listed here
data_path='' # data source path
swarm_dir='' #parameter file path
savecsv='' #test results path
    
batch_size=200
num_workers=10
wd=1e-4
task_num=2
learning_rate=0.001
device = torch.device("cuda")

loss_local=100
dataset1_path=data_path+'/'+node_list[0]+'/train'
dataset2_path=data_path+'/HS_RSNA_LUNG/train'           
train_data_loader=load_HSL_train(dataset1_path,dataset2_path,batch_size,num_workers)

model = MMoE(tasks=task_num).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay= wd)
scheduler1 = lr_scheduler.CosineAnnealingLR(optimizer,T_max=10, eta_min=1e-4)

for epoch in range(1,6): #multi-round parameter sharing and multi-epoch local tarining in each round of parameter sharing
    loss_train=train_HSL(model, optimizer, train_data_loader, criterion, device)   
    dict_local= model.state_dict()
    if loss_train<loss_local:
        loss_local=loss_train
        torch.save(dict_local, swarm_dir+'/{}_dict_best.pkl'.format(node_list[0]))

# parameter sharing
dict0=torch.load(swarm_dir+'/_{}_dict_best.pkl'.format(node_list[0]))
for name in dict0:
    for node in node_list[1:]:
        dict0[name]+=torch.load(swarm_dir+'/_{}_dict_best.pkl'.format(node))[name]
    dict0[name]/=len(node_list)
torch.save(dict0, swarm_dir+'/{}_dict_best_final.pkl'.format(node_list[0]))

#test
model.load_state_dict(torch.load(swarm_dir+'/all_dict_best_final.pkl' )) 
dataset1_path=data_path+'/'+node_list[0]+'/test'
dataset2_path=data_path+'/HS_RSNA_LUNG/test'           
test_data_loader=load_HSL_test(dataset1_path,dataset2_path,batch_size,num_workers)
auc_value, acc, recall, precision, f1=test_HSL(model, test_data_loader, task_num, device,savecsv+'/'+node_list[0]+'.csv')       