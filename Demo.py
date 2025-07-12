import torch,os,shutil
from SSL_train import load_HSL_train,load_HSL_test,train_HSL,test_HSL
from MMOE_ResNet18 import MMoE
import warnings
warnings.filterwarnings("ignore")
# In[*] 
os.chdir('D:/Code Demo')
node='Node1'
data_path='./Data'
save_dir='./Results'
batch_size=100
num_work=8
  
if __name__ == '__main__':
    device = torch.device("cuda")
                        
    dataset1_path=data_path+'/train'
    dataset2_path='/Data/RSNA_LUNG'     
    train_data_loader=load_HSL_train(dataset1_path,dataset2_path,batch_size,num_work)
    
    dataset_path=data_path+'/val'
    val_data_loader=load_HSL_test(dataset_path,dataset_path,batch_size,num_work)

    auc_val_best=0;auc_all_best=0
    for round_  in range(1,11):               
        model = MMoE(tasks=2).to(device)  
        for epoch in range(1,11):
            train_HSL(model, train_data_loader, device)   
            dict_1= model.state_dict()
        
            node1_auc=test_HSL(model, val_data_loader, device,save_dir+'/val.csv')       
            print('Round {} Epoch {} Auc: {:.4f}'.format(round_,epoch,node1_auc))
            
            if auc_val_best <= node1_auc:
                auc_val_best = node1_auc
                dict_best = dict_1
                dict_best['auc'] = torch.tensor(node1_auc)
                torch.save(dict_best, save_dir+'/Node1_dict_best.pkl')     
                
        if round_==1:
            pass
        
        else:
            dict_2 = torch.load(save_dir+'/Node2_dict_best.pkl')
            node2_auc = dict_2['auc'].item() 

            dict_3 = torch.load(save_dir+'/Node3_dict_best.pkl')
            node3_auc = dict_3['auc'].item() 
            
            for name in dict_1:
                if name != 'auc':
                    dict_1[name] = (dict_1[name]+dict_2[name]+dict_3[name])/3
          
            model_params = {k: v for k, v in dict_1.items() if not k.startswith('auc')}
            model.load_state_dict(model_params)
            torch.save(dict_1, save_dir+'/all_dict_best.pkl')

            auc_all=(node1_auc+node2_auc+node3_auc)/3
            if auc_all_best < auc_all:
                auc_all_best=auc_all
                shutil.copy(save_dir+'/all_dict_best.pkl',save_dir+'/final_dict_best.pkl')

    dict_final = torch.load(save_dir+'/final_dict_best.pkl')
    model_params = {k: v for k, v in dict_final.items() if not k.startswith('auc')}
    model.load_state_dict(model_params)

    dataset_path=data_path+'/test'
    test_data_loader=load_HSL_test(dataset_path,dataset_path,batch_size,num_work)
    node1_test_auc=test_HSL(model, test_data_loader, device,save_dir+'/test.csv')       
    print('Test Auc: {:.4f}'.format(node1_test_auc))
