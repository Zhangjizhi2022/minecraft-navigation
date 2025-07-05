# !/usr/bin/env python
#  coding: utf-8

#  In[ ]:


import os
import pandas as pd

#  定义要遍历的根目录列表
root_dirs = {
    '1E': 'd:/projects/Gaming/data/1E',
    '2E': 'd:/projects/Gaming/data/2E',
    '3E': 'd:/projects/Gaming/data/3E',
    '4E': 'd:/projects/Gaming/data/4E'
}

#  定义各环境的替换坐标
expected_first_rows = {
    '1E': [159, 68, 1003, 1],
    '2E': [164, 64, -768, 1],
    '3E': [-879, 80, -912, 1],
    '4E': [-251, 71, 1948, 1]
}

#  创建一个字典来存储所有的 DataFrame
dataframes1 = {}
dataframes2 = {}

#  遍历每个根目录
for env, root_dir in root_dirs.items():
    for folder_name, subfolders, filenames in os.walk(root_dir):
        if 'training2.csv' in filenames:
            #  构建file的完整path
            file_path1 = os.path.join(folder_name, 'training1.csv')
            file_path2 = os.path.join(folder_name, 'training2.csv')

            #  Read CSV file
            df1 = pd.read_csv(file_path1, header=None)
            df2 = pd.read_csv(file_path2, header=None)

            #  选择对应环境的替换坐标
            expected_first_row = expected_first_rows[env]

            #  处理 training1.csv
            if not df1.iloc[0].tolist() == expected_first_row:
                df1.iloc[0] = expected_first_row
                df1 = df1.drop(1).reset_index(drop=True)

            #  处理 training2.csv
            if not df2.iloc[0].tolist() == expected_first_row:
                df2.iloc[0] = expected_first_row
                df2 = df2.drop(1).reset_index(drop=True)

            #  以file夹名和环境名命名 DataFrame
            folder = os.path.basename(folder_name)
            dataframe_name = f"{env}_{folder}"

            #  将 DataFrame 存入字典，并记录来源环境
            dataframes1[dataframe_name] = df1
            dataframes2[dataframe_name] = df2


#  In[ ]:


print('Hello world')


#  In[65]:


#  访问特定 DataFrame 并返回对应环境
def get_dataframe_and_env(dataframes, target_folder):
    for key in dataframes.keys():
        if key.endswith(target_folder):
            env = key.split('_')[0]
            return dataframes[key], env
    return None, None

#  调用并返回 某个特定的subject 对应的 DataFrame 和环境
df, env = get_dataframe_and_env(dataframes1, '4015')
print(df, env)


#  获取第四列为1的行的索引
indices = df[df[3] == 1].index

#  分割data框
segments = []
for i in range(len(indices) - 1):
    start_idx = indices[i]
    end_idx = indices[i + 1]
    segment = df.iloc[start_idx:end_idx + 1]  #  包含起始和结束行
    segments.append(segment)

#  初始化存储满足条件的segment索引
selected_segments = []

#  遍历每个segment，计算坐标1的最大值和最小值差值
for idx, segment in enumerate(segments):
    max_val = segment[1].max()
    min_val = segment[1].min()
    if (max_val - min_val) > 5:
        selected_segments.append(idx)

#  输出满足条件的segment标号
print("满足条件的segment标号:", selected_segments)


#  In[71]:


df = segments[16]


import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#  绘制二维图 (x, y)
plt.figure(figsize=(10, 6))
plt.plot(df[0], df[2])
plt.scatter(df[0].iloc[0], df[2].iloc[0], color='green', s=100, label='Start')
plt.scatter(df[0].iloc[-1], df[2].iloc[-1], color='red', s=100, label='End')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2D Plot of X and Y')
plt.legend()
plt.grid(True)
plt.show()

#  绘制三维图 (x, y, z)
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(df[0], df[2], df[1])
ax.scatter(df[0].iloc[0], df[2].iloc[0], df[1].iloc[0], color='green', s=100, label='Start')
ax.scatter(df[0].iloc[-1], df[2].iloc[-1], df[1].iloc[-1], color='red', s=100, label='End')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('3D Plot of X, Y and Z')
ax.legend()
plt.show()


#  In[18]:


df
