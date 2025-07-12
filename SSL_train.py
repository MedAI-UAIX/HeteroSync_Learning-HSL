import torch,os,random,tqdm
from PIL import Image
import torchvision.transforms as transforms
import pandas as pd
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

def train_HSL(model, data_loader, device, log_interval=100):
    model.train()
    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.AdamW(model.parameters(),lr=0.0001,weight_decay= 1e-5)
    loader = tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0,leave=False)
    for i, (img_1, img_2,label_1, label_2, img_ID_1, img_ID_2) in enumerate(loader):
        img_1, img_2, label_1, label_2, = img_1.to(device), img_2.to(device), label_1.to(device), label_2.to(device)
        y = model(img_1, img_2)
        labels = torch.cat([label_1.unsqueeze(1), label_2.unsqueeze(1)], 1).view(-1, 2)

        loss1 = criterion(y[0], labels[:, 0].float());loss2=criterion(y[1], labels[:, 1].float())
        loss=loss1*0.5+loss2*0.5
        model.zero_grad()
        loss.backward()
        optimizer.step()

def test_HSL(model, data_loader, device,savecsv):
    model.eval()
    y_label, y_pred, IDs = list(), list(), list()
    with torch.no_grad():
        for img_1, img_2, label_1, label_2, img_ID_1, img_ID_2, in tqdm.tqdm(data_loader, smoothing=0, mininterval=1.0):
            img_1, img_2, label_1, label_2 = img_1.to(device), img_2.to(device), label_1.to(device), label_2.to(device)
            y = model(img_1, img_2)
            labels = torch.cat([label_1.unsqueeze(1), label_2.unsqueeze(1), ], 1).view(-1, 2) 

            y_label.extend(labels[:, 0].tolist())
            y_pred.extend(y[0].tolist())
            IDs.extend(list(img_ID_1))

    auc_value=roc_auc_score(y_label, y_pred)        
    df1 = pd.DataFrame(IDs,columns=['IDs']); df1.set_index('IDs',inplace=True);df1['label'] = y_label; df1['prob'] = y_pred
    df1.to_csv(savecsv, index=False)

    return auc_value


class DatasetImage(torch.utils.data.Dataset):
    def __init__(self, dataset1_path, dataset2_path, transform=None):
        assert (os.path.exists(dataset1_path)), "data_dir:{} 不存在！".format(dataset1_path)
        assert (os.path.exists(dataset2_path)), "data_dir:{} 不存在！".format(dataset2_path)
        
        self.is_balance1 = True
        self.is_balance2 = False
            
        self.transform = transform
        self.dataset1_img_info,self.label_dict1 = self._get_img_info(dataset1_path)
        self.dataset2_img_info,self.label_dict2 = self._get_img_info(dataset2_path)
        
        dataset1_num = self.get_dataset_len(dataset1_path, self.dataset1_img_info)
        dataset2_num = self.get_dataset_len(dataset2_path, self.dataset2_img_info)
        self.dataset_list = [dataset1_num, dataset2_num]
        self.max_num = max(self.dataset_list)
        
        self.score1,self.min_idx1,self.max_idx1 = self.balance_init(self.label_dict1)
        self.score2,self.min_idx2,self.max_idx2 = self.balance_init(self.label_dict2)

    def __len__(self):
        return self.max_num

    def __getitem__(self, index):
        img_path_1, label_1, img_ID_1 = self.random_index(index,self.dataset1_img_info)
        img_path_2, label_2, img_ID_2 = self.random_index(index,self.dataset2_img_info)

        if self.is_balance1:
            p1 = random.random()
            if p1 <= self.score1 and label_1==self.max_idx1:
                img_path_1, label_1, img_ID_1 = random.choice(self.label_dict1[self.min_idx1])
         
        if self.is_balance2:        
            p2 = random.random()
            if p2 <= self.score2 and label_2==self.max_idx2:
                img_path_2, label_2, img_ID_2 = random.choice(self.label_dict2[self.min_idx2])
     
        img_1 = Image.open(img_path_1).convert('RGB')
        img_2 = Image.open(img_path_2).convert('RGB')

        if self.transform is not None:
            img_1 = self.transform(img_1)
            img_2 = self.transform(img_2)

        return img_1, img_2, label_1, label_2, img_ID_1, img_ID_2
    
    def balance_init(self,label_dict, th=0.5):
        list_ = [len(label_dict[0]),len(label_dict[1])]
        p12 = [i/ sum(list_) for i in list_]
        p1,p2 = min(p12),max(p12)
        return (th - p1)/p2, p12.index(p1), p12.index(p2)
        
    def random_index(self, index, dataset_img_info):
        if index < len(dataset_img_info):
            img_path, label, img_ID = dataset_img_info[index]
            
        else:
            n = random.randint(0, len(dataset_img_info)-1)
            img_path, label, img_ID = dataset_img_info[n]
                
        return img_path, label, img_ID


    def get_dataset_len(self, dataset_name, img_info):
        if len(img_info) == 0:
            raise Exception("未获取任何图片路径，请检查{}路径！".format(dataset_name))
        return len(img_info)


    def _get_img_info(self, data_dir):
        sub_dir_ = [name for name in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, name))]
        sub_dir = [os.path.join(data_dir, c) for c in sub_dir_]

        label_dict = {0:[],1:[]}
        img_info = []
        for c_dir in sub_dir:
            for i in os.listdir(c_dir):              
                path_img = os.path.join(c_dir, i)
                path_img = (os.path.join(c_dir, i), int(c_dir.split('\\')[-1]), i) 
                img_info.append(path_img)
                
                if path_img[1]==0:
                    label_dict[0].append(path_img)
                else:
                    label_dict[1].append(path_img)
        
        return [img_info,label_dict]
    
def load_HSL_train(train_dataset_1_path, train_dataset_2_path,batch_size,num_workers):
    train_transform = transforms.Compose([

        transforms.Resize((224,224),3), 
        transforms.RandomRotation(15),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor()])

    train_dataset = DatasetImage(train_dataset_1_path, train_dataset_2_path, train_transform)
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

    return train_data_loader

def load_HSL_test(test_dataset_1_path,test_dataset_2_path,batch_size,num_workers):
    test_transform = transforms.Compose([
        transforms.Resize((224,224),3),
        transforms.ToTensor()])

    test_dataset = DatasetImage(test_dataset_1_path, test_dataset_2_path, test_transform)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

    return test_data_loader
