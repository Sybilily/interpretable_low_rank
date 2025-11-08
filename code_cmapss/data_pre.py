import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.utils.data as data

from sklearn.preprocessing import StandardScaler
def add_operating_condition(df):
    df_op_cond = df.copy()
    
    df_op_cond['context_c1'] = abs(df_op_cond['context_c1'].round())
    df_op_cond['context_c2'] = abs(df_op_cond['context_c2'].round(decimals=2))
    
    # converting settings to string and concatanating makes the operating condition into a categorical variable
    df_op_cond['op_cond'] = df_op_cond['context_c1'].astype(str) + '_' + \
                        df_op_cond['context_c2'].astype(str) + '_' + \
                        df_op_cond['context_c3'].astype(str)
    
    return df_op_cond
 
def condition_scaler(df_train,sensor_names):
    # apply operating condition specific scaling
    scaler = StandardScaler()
    for condition in df_train['op_cond'].unique():
        scaler.fit(df_train.loc[df_train['op_cond']==condition, sensor_names])
        df_train.loc[df_train['op_cond']==condition, sensor_names] = scaler.transform(df_train.loc[df_train['op_cond']==condition, sensor_names])
        #df_test.loc[df_test['op_cond']==condition, sensor_names] = scaler.transform(df_test.loc[df_test['op_cond']==condition, sensor_names])
    return df_train#, df_test


class CmapssData:
    def __init__(self,file_path, file_names, class_dim, latent_dim=0):
        # 2. 定义要读取的列名
        columns = [
            "unit_id", "cycle", "context_c1", "context_c2", "context_c3", 
            *[f"sensor_{i}" for i in range(1, 22)]  # sensor_1 到 sensor_26
        ]
        # 3. 用于存储读取的数据
        data_frames = []

        # 4. 读取每个文件并处理
        for file_name in file_names:
            file_path_full = os.path.join(file_path, file_name)
            
            # 读取文件，假设文件使用空格或制表符分隔
            df = pd.read_csv(file_path_full, sep=r'\s+', header=None, names=columns)
            df['data_set'] = file_name
            # 将读取的数据添加到数据列表中
            data_frames.append(df)
        sensor_idx_ls = [2,3,4,7,8,9,11,12,13,14,15,17,20,21]
        sensor_columns = [f"sensor_{i}" for i in sensor_idx_ls]
        # 5. 合并所有的 DataFrame
        final_df = pd.concat(data_frames, ignore_index=True)
        
        self.sensor_data_original = final_df[[*[f"sensor_{i}" for i in range(1,22)]]]
        
        final_df = add_operating_condition(final_df)
        
        for i in sensor_idx_ls:
            final_df['sensor_' + str(i)] = abs(final_df['sensor_' + str(i)].round(decimals=1))
        
        final_df = condition_scaler(final_df, sensor_columns)

        # 假设 final_df 已经读取并合并完毕
        # 1. 归一化处理 (MinMaxScaler)
        #scaler_s = MinMaxScaler()
        scaler_c = MinMaxScaler()
        # 只对传感器列进行归一化处理
        
        #final_df[sensor_columns] = scaler_s.fit_transform(final_df[sensor_columns])
        context_columns = [f"context_c{i}" for i in range(1, 4)]
        final_df[context_columns] = scaler_c.fit_transform(final_df[context_columns])
        
        context_data = final_df[['context_c1', 'context_c2', 'context_c3']]
        sensor_data = final_df[[*[f"sensor_{i}" for i in sensor_idx_ls]]]
        
        index_data = final_df[['data_set','unit_id', 'cycle','op_cond']]
        context_dim = context_data.shape[1]
        sensor_dim = sensor_data.shape[1]
        
        self.sensor_data = sensor_data
        self.context_data = context_data
        self.index_data = index_data
        self.class_dim = class_dim
        self.latent_dim = latent_dim
        self.sensor_dim = sensor_dim
        self.context_dim = context_dim
        self.sample_size = sensor_data.shape[0]
        n_label = pd.factorize(index_data['data_set'])[0].max()+1
        self.train_label = pd.factorize(index_data['data_set'])[0]
        self.r_input_init = torch.tensor(np.eye(class_dim)[self.train_label*0]).float() # input label, paritally given
        # 如果超出了本身的维度，需要用0补齐
        r_input_init = torch.zeros(self.sample_size, self.class_dim)
        r_input_init[:,:min(class_dim, n_label)] = self.r_input_init[:,:min(class_dim, n_label)]
        self.r_input_init = r_input_init

def get_train_test_valid(train_ratio=0.8, batch_size=256, *data_ls):
    # 传入的dataset数量是可变的，但是要保证前三个都是必须
    torch_dataset = data.TensorDataset( *data_ls)

    # 将数据集分割成训练集和验证集
    train_size = int(train_ratio * len(torch_dataset))
    test_size = int(0.1 * len(torch_dataset))
    valid_size = len(torch_dataset) - train_size - test_size

    train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(torch_dataset, [train_size, valid_size, test_size])

    # 把dataset放入DataLoader
    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,             # 每批提取的数量
        shuffle=True        # 要不要打乱数据（打乱比较好）
        )
    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=int(batch_size/2),
        shuffle=True
    )
    valid_loader = data.DataLoader(
        dataset=valid_dataset,
        batch_size=int(batch_size/2),
        shuffle=True
    )
    train_data = [torch.stack([data[i] for i in train_dataset.indices]) for data in data_ls]
    test_data = [torch.stack([data[i] for i in test_dataset.indices]) for data in data_ls]
    valid_data = [torch.stack([data[i] for i in valid_dataset.indices]) for data in data_ls]
    dataloader_dict = {'train': train_loader, 'test': test_loader, 'valid': valid_loader}
    dataset_dict = {'train': train_data, 'test': test_data, 'valid': valid_data}
    return dataloader_dict, dataset_dict
