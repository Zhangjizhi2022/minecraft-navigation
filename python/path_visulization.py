# !/usr/bin/env python
#  coding: utf-8

#  In[1]:


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


#  In[2]:


#  访问特定 DataFrame 并返回对应环境
def get_dataframe_and_env(dataframes, target_folder):
    for key in dataframes.keys():
        if key.endswith(target_folder):
            env = key.split('_')[0]
            return dataframes[key], env
    return None, None




#  In[14]:





#  In[3]:


import os
import pandas as pd
import matplotlib.pyplot as plt

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

#  定义 unique_locations
unique_locations = {
    '1E': {
        (159, 68, 1003): 'START', (129, 67, 1003): 'Pickaxe', (99, 70, 973): 'Bone',
        (129, 69, 943): 'Egg', (159, 70, 943): 'Diamond', (189, 77, 973): 'Bowl',
        (219, 70, 973): 'Cake', (189, 68, 1003): 'Apple', (219, 71, 1033): 'Bread',
        (159, 67, 1063): 'Book', (189, 70, 1063): 'Brick', (129, 78, 1033): 'Stick',
        (99, 69, 1033): 'Arrow'
    },
    '2E': {
        (164, 64, -768): 'START', (194, 63, -798): 'Cactus', (194, 63, -768): 'Bucket',
        (224, 63, -798): 'Pumpkin', (164, 63, -828): 'Emerald', (134, 63, -828): 'Watermelon',
        (104, 65, -798): 'Carrot', (134, 62, -768): 'Helmet', (104, 68, -738): 'FishingRod',
        (134, 66, -738): 'Ladder', (164, 67, -708): 'Paper', (194, 67, -708): 'Axe',
        (224, 64, -738): 'Wheat',
    },
    '3E': {
        (-879, 80, -912): 'START', (-849, 77, -912): 'Bed', (-879, 79, -852): 'String',
        (-849, 79, -942): 'Saddle', (-819, 78, -942): 'PumpkinPie', (-879, 75, -972): 'Boat',
        (-909, 77, -972): 'Record', (-939, 80, -942): 'Fish', (-909, 75, -912): 'Pants',
        (-909, 73, -882): 'Roses', (-819, 78, -882): 'Shovel', (-939, 77, -882): 'Coal',
        (-849, 74, -852): 'Steak'
    },
    '4E': {
        (-251, 75, 1948): 'START', (-281, 66, 1948): 'FlowerPot', (-281, 63, 1888): 'Feather',
        (-311, 66, 1918): 'Cookie', (-251, 69, 2008): 'SpiderWeb', (-221, 68, 2008): 'Bottle',
        (-281, 68, 1978): 'Sword', (-251, 69, 1888): 'Window', (-221, 70, 1918): 'Potato',
        (-221, 71, 1948): 'Anvil', (-191, 69, 1978): 'Gold', (-191, 65, 1918): 'Mushroom',
        (-311, 68, 1978): 'Torch'
    }
}


#  In[26]:


#  调用并返回 某个特定的subject 对应的 DataFrame 和环境
df, env = get_dataframe_and_env(dataframes2, '2024')
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

# df = segments[12] 




#  In[ ]:


#  调用并返回 某个特定的subject 对应的 DataFrame 和环境
df, env = get_dataframe_and_env(dataframes1, '2025')
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

df = segments[6] 

#  提取前两列作为 x 和 y 坐标
x = df[0]
y = df[2]

unique_x = [location[0] for location in unique_locations[env].keys()]
unique_y = [location[2] for location in unique_locations[env].keys()]
labels = list(unique_locations[env].values())

start_x, start_y = x.iloc[0], y.iloc[0]
end_x, end_y = x.iloc[-1], y.iloc[-1]

plt.figure(figsize=(8, 6))
plt.plot(x, y, linestyle='-', color='b', label='Route')

plt.scatter(unique_x, unique_y, color='r', marker='o', label='Unique Locations')
for i, label in enumerate(labels):
    plt.text(unique_x[i], unique_y[i], label, fontsize=15, ha='right')

plt.scatter(start_x, start_y, color='g', marker='*', s=20, label='Start Point')
plt.scatter(end_x, end_y, color='purple', marker='X', s=20, label='End Point')
plt.text(start_x, start_y, 'Start', fontsize=15, ha='left', color='g')
plt.text(end_x, end_y, 'End', fontsize=15, ha='left', color='purple')

# plt.title('X-Z Route Plot')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.legend()
plt.grid(True)
plt.show()



#  In[28]:


import heapq

#  Modify the function to return both the path and the distance score for the best route
def dijkstra_terrain_with_existence_check_debug(start, goal, terrain_df):
    #  Convert terrain data into a dictionary for fast lookup
    terrain_dict = {(row['x'], row['z']): row['y'] for _, row in terrain_df.iterrows()}

    #  Define possible movement directions (8 directions)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    pq = [(0, start)]  #  Priority queue for nodes to explore
    distances = {start: 0}  #  Distance from start to each node
    previous = {start: None}  #  To reconstruct path

    while pq:
        current_distance, current_node = heapq.heappop(pq)

        if current_node == goal:
            break  #  Stop if goal is reached

        #  Check neighbors
        for direction in directions:
            neighbor = (current_node[0] + direction[0], current_node[1] + direction[1])

            #  Check if neighbor exists in the terrain data
            if neighbor in terrain_dict:
                current_y = terrain_dict[current_node]
                neighbor_y = terrain_dict[neighbor]
                height_diff = neighbor_y - current_y

                #  Define movement cost based on height difference and direction
                if height_diff > 2:
                    continue  #  Skip if height difference for ascending is too large
                elif height_diff < -3:
                    continue  #  Skip if height difference for descending is too large

                #  Movement costs based on height difference
                is_diagonal = direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                base_cost = 1.41 if is_diagonal else 1

                if height_diff == 0:
                    distance = current_distance + base_cost
                elif height_diff < 0:
                    distance = current_distance + base_cost + (1.5 * abs(height_diff))  #  Descend
                elif height_diff > 0:
                    distance = current_distance + base_cost + (2.5 * height_diff)  #  Ascend
                else:
                    distance = current_distance + base_cost + abs(height_diff)  #  Larger jumps

                #  Update distances and paths if a shorter path is found
                if neighbor not in distances or distance < distances[neighbor]:
                    distances[neighbor] = distance
                    priority = distance
                    heapq.heappush(pq, (priority, neighbor))
                    previous[neighbor] = current_node

    #  Reconstruct path from start to goal and return distance score
    path = []
    current = goal
    if current not in previous:
        return "No path found", None  #  Return a message if the goal is unreachable

    while current is not None:
        path.append(current)
        current = previous.get(current)
    path.reverse()  #  Reverse to get path from start to goal

    #  Return the path and the final distance score to the goal
    best_score = distances[goal] if goal in distances else None
    return path, best_score


#  In[64]:


import pandas as pd
import matplotlib.pyplot as plt
import heapq

#  地形图filepath
environment_terrain_paths = {
    '1E': 'd:../data/df_min_y_filtered.csv',
    '2E': 'd:../data/df_min_y_filtered.csv',
    '3E': 'd:../data/df_min_y_filtered.csv',
    '4E': 'd:../data/df_min_y_filtered.csv',
}

#  定义 unique_locations
unique_locations = {
    '1E': {
        (159, 68, 1003): 'START', (129, 67, 1003): 'Pickaxe', (99, 70, 973): 'Bone',
        (129, 69, 943): 'Egg', (159, 70, 943): 'Diamond', (189, 77, 973): 'Bowl',
        (219, 70, 973): 'Cake', (189, 68, 1003): 'Apple', (219, 71, 1033): 'Bread',
        (159, 67, 1063): 'Book', (189, 70, 1063): 'Brick', (129, 78, 1033): 'Stick',
        (99, 69, 1033): 'Arrow'
    },
    '2E': {
        (164, 64, -768): 'START', (194, 63, -798): 'Cactus', (194, 63, -768): 'Bucket',
        (224, 63, -798): 'Pumpkin', (164, 63, -828): 'Emerald', (134, 63, -828): 'Watermelon',
        (104, 65, -798): 'Carrot', (134, 62, -768): 'Helmet', (104, 68, -738): 'FishingRod',
        (134, 66, -738): 'Ladder', (164, 67, -708): 'Paper', (194, 67, -708): 'Axe',
        (224, 64, -738): 'Wheat',
    },
    '3E': {
        (-879, 80, -912): 'START', (-849, 77, -912): 'Bed', (-879, 79, -852): 'String',
        (-849, 79, -942): 'Saddle', (-819, 78, -942): 'PumpkinPie', (-879, 75, -972): 'Boat',
        (-909, 77, -972): 'Record', (-939, 80, -942): 'Fish', (-909, 75, -912): 'Pants',
        (-909, 73, -882): 'Roses', (-819, 78, -882): 'Shovel', (-939, 77, -882): 'Coal',
        (-849, 74, -852): 'Steak'
    },
    '4E': {
        (-251, 75, 1948): 'START', (-281, 66, 1948): 'FlowerPot', (-281, 63, 1888): 'Feather',
        (-311, 66, 1918): 'Cookie', (-251, 69, 2008): 'SpiderWeb', (-221, 68, 2008): 'Bottle',
        (-281, 68, 1978): 'Sword', (-251, 69, 1888): 'Window', (-221, 70, 1918): 'Potato',
        (-221, 71, 1948): 'Anvil', (-191, 69, 1978): 'Gold', (-191, 65, 1918): 'Mushroom',
        (-311, 68, 1978): 'Torch'
    }
}

#  Dijkstra 算法实现
#  包括前面提供的 dijkstra_terrain_with_existence_check_debug function
#  此处省略重复定义

#  Load地形图data
def load_terrain_data(terrain_path):
    return pd.read_csv(terrain_path)

#  绘制每段 segment 及其 Dijkstra 最优path
def plot_segment_with_path(df_segment, env):
    #  提取前两列作为 x 和 y 坐标
    x = df_segment[0]
    y = df_segment[2]

    #  Load地形data
    terrain_path = environment_terrain_paths[env]
    terrain_df = load_terrain_data(terrain_path)

    #  计算起点和终点坐标
    start = (x.iloc[0], y.iloc[0])
    end = (x.iloc[-1], y.iloc[-1])

    #  计算 Dijkstra 最优path
    optimal_path, best_score = dijkstra_terrain_with_existence_check_debug(start, end, terrain_df)

    #  提取path的 x 和 z 坐标
    path_x = [p[0] for p in optimal_path]
    path_y = [p[1] for p in optimal_path]

    #  获取 unique_locations data
    unique_x = [location[0] for location in unique_locations[env].keys()]
    unique_y = [location[2] for location in unique_locations[env].keys()]
    labels = list(unique_locations[env].values())

    #  绘制图表
    plt.figure(figsize=(8, 6))
    #  加粗路线和path
    plt.plot(x, y, linestyle='-', color='b', linewidth=2, label='Route')
    plt.plot(path_x, path_y, linestyle='-', color='red', linewidth=2, label='Optimal Path')

    plt.scatter(unique_x, unique_y, color='r', marker='o', label='Objects ')
    for i, label in enumerate(labels):
        plt.text(unique_x[i], unique_y[i], label, fontsize=20, ha='right')

    #  标出起点和终点，并添加标注文字
    plt.scatter(start[0], start[1], color='g', marker='*', s=100, label='Start Point')
    plt.scatter(end[0], end[1], color='purple', marker='X', s=100, label='End Point')
    plt.text(start[0], start[1], 'Start', fontsize=20, ha='left', color='g')
    plt.text(end[0], end[1], 'End', fontsize=20, ha='left', color='purple')

    #  添加标题和图例
    # plt.title(f'Segment and Optimal Path for Environment {env}')
    # plt.xlabel('X Coordinate')
    # plt.ylabel('Z Coordinate')
    # plt.legend()
    plt.grid(True)
    plt.show()

#  示例：处理某个特定 segment
#  假设我们已经提取了某个环境的特定 segment
df_segment_example, env_example = get_dataframe_and_env(dataframes2, '2300')
indices = df_segment_example[df_segment_example[3] == 1].index
segments = []
for i in range(len(indices) - 1):
    start_idx = indices[i]
    end_idx = indices[i + 1]
    segment = df_segment_example.iloc[start_idx:end_idx + 1]  #  包含起始和结束行
    segments.append(segment)

#  选择某段 segment
example_segment = segments[6]

#  绘制包含 Dijkstra 最优path的图
plot_segment_with_path(example_segment, env_example)



#  In[28]:


import matplotlib.pyplot as plt
import pandas as pd

def plot_paths(terrain_df, path_df, dijkstra_func):
    """
    Plot the given path and optimal sub-paths starting from different points along the main path.

    Parameters:
    - terrain_df: DataFrame containing terrain data.
    - path_df: DataFrame containing the main path with integer columns (e.g., 0 for x, 2 for z).
    - dijkstra_func: Function to compute the optimal path.
    """
    #  Extract x and z coordinates from the main path
    x_main = path_df.iloc[:, 0].tolist()  #  Assuming 'x' is in the first column
    z_main = path_df.iloc[:, 2].tolist()  #  Assuming 'z' is in the third column

    #  Determine key points along the main path
    total_length = len(path_df)
    quarter = total_length // 4
    midpoint = total_length // 2
    three_quarters = 3 * total_length // 4

    key_points = [0, quarter, midpoint, three_quarters, total_length - 1]

    #  Colors for sub-paths
    colors = ['blue', 'green', 'orange', 'purple', 'red']

    plt.figure(figsize=(10, 10))

    #  Plot the main path
    plt.plot(x_main, z_main, label="Main Path", color="black", linestyle="--")

    #  Plot sub-paths starting from key points
    for i, start_index in enumerate(key_points[:-1]):
        start = (path_df.iloc[start_index, 0], path_df.iloc[start_index, 2])  #  Get x, z from integer indices
        goal = (path_df.iloc[-1, 0], path_df.iloc[-1, 2])  #  Always ends at the final point of the main path
        sub_path, _ = dijkstra_func(start, goal, terrain_df)

        #  Extract x and z coordinates for the sub-path
        x_sub = [point[0] for point in sub_path]
        z_sub = [point[1] for point in sub_path]

        plt.plot(x_sub, z_sub, label=f"Sub-path {i+1}", color=colors[i])

    #  Add labels and legend
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title("Optimal Path and Sub-paths")
    plt.legend()
    plt.grid(True)
    plt.show()

#  Example usage
terrain_df = pd.read_csv('d:../data/df_min_y_filtered.csv')
path_df = df  #  Ensure 'df' has integer column indices

def dijkstra_func(start, goal, terrain_df):
    #  Replace this with the actual dijkstra_terrain_with_existence_check_debug function
    return dijkstra_terrain_with_existence_check_debug(start, goal, terrain_df)

plot_paths(terrain_df, path_df, dijkstra_func)


#  In[38]:


import matplotlib.pyplot as plt
import pandas as pd

def plot_paths(terrain_df, path_df, dijkstra_func, unique_locations):
    """
    Plot the given path and optimal sub-paths starting from different points along the main path.

    Parameters:
    - terrain_df: DataFrame containing terrain data.
    - path_df: DataFrame containing the main path with integer columns (e.g., 0 for x, 2 for z).
    - dijkstra_func: Function to compute the optimal path.
    - unique_locations: Dictionary of unique locations with keys as coordinates and values as labels.
    """
    #  Extract x and z coordinates from the main path
    x_main = path_df.iloc[:, 0].tolist()  #  Assuming 'x' is in the first column
    z_main = path_df.iloc[:, 2].tolist()  #  Assuming 'z' is in the third column

    #  Determine key points along the main path
    total_length = len(path_df)
    quarter = total_length // 4
    midpoint = total_length // 2
    three_quarters = 3 * total_length // 4

    key_points = [0, quarter, midpoint, three_quarters, total_length - 1]

    #  Colors for sub-paths
    main_path_color = 'blue'
    sub_path_color = 'red'
    sub_path_style = '--'

    plt.figure(figsize=(10, 10))

    #  Plot the main path
    plt.plot(x_main, z_main, linestyle='-', color=main_path_color, label="Main Path")

    #  Plot sub-paths starting from key points
    for i, start_index in enumerate(key_points[:-1]):
        start = (path_df.iloc[start_index, 0], path_df.iloc[start_index, 2])  #  Get x, z from integer indices
        goal = (path_df.iloc[-1, 0], path_df.iloc[-1, 2])  #  Always ends at the final point of the main path
        sub_path, _ = dijkstra_func(start, goal, terrain_df)

        #  Extract x and z coordinates for the sub-path
        x_sub = [point[0] for point in sub_path]
        z_sub = [point[1] for point in sub_path]

        plt.plot(x_sub, z_sub, linestyle=sub_path_style, color=sub_path_color, label=f"Sub-path {i+1}")


    #  Plot start and end points
    start_x, start_z = x_main[0], z_main[0]
    end_x, end_z = x_main[-1], z_main[-1]
    plt.scatter(start_x, start_z, color='green', marker='*', s=100, label='Start Point')
    plt.scatter(end_x, end_z, color='purple', marker='X', s=100, label='End Point')
    plt.text(start_x, start_z, 'Start', fontsize=10, ha='left', color='green')
    plt.text(end_x, end_z, 'End', fontsize=10, ha='left', color='purple')

    #  Add labels and legend
    plt.xlabel("X Coordinate")
    plt.ylabel("Z Coordinate")
    plt.title("Optimal Path and Sub-paths with Unique Locations")
    plt.legend()
    plt.grid(True)
    plt.show()

#  Example usage
terrain_df = pd.read_csv('d:../data/df_min_y_filtered.csv')
path_df = df

def dijkstra_func(start, goal, terrain_df):
    #  Replace this with the actual dijkstra_terrain_with_existence_check_debug function
    return dijkstra_terrain_with_existence_check_debug(start, goal, terrain_df)

plot_paths(terrain_df, path_df, dijkstra_func, unique_locations)


#  In[46]:


import matplotlib.pyplot as plt
import pandas as pd

#  Readdata
terrain_df = pd.read_csv('d:../data/df_min_y_filtered.csv')
path_df = df


#  模拟 Dijkstra function（请替换为实际function）
def dijkstra_func(start, goal, terrain_df):
    #  模拟path
    return [(start[0], start[1]), (goal[0], goal[1])], 0

#  提取主path的 x 和 z 坐标
x_main = path_df.iloc[:, 0].tolist()  #  假设 'x' 在第一列
z_main = path_df.iloc[:, 2].tolist()  #  假设 'z' 在第三列

#  计算关键点
total_length = len(path_df)
quarter = total_length // 4
midpoint = total_length // 2
three_quarters = 3 * total_length // 4

key_points = [0, quarter, midpoint, three_quarters, total_length - 1]

#  Set颜色和样式
main_path_color = 'blue'
sub_path_color = 'red'
sub_path_style = '--'

#  绘制图形
plt.figure(figsize=(10, 10))

#  绘制主path
plt.plot(x_main, z_main, linestyle='-', color=main_path_color, label="Main Path")

#  绘制从关键点到目标点的子path
for i, start_index in enumerate(key_points[:-1]):
    start = (path_df.iloc[start_index, 0], path_df.iloc[start_index, 2])  #  起点
    goal = (path_df.iloc[-1, 0], path_df.iloc[-1, 2])  #  终点
    sub_path, _ = dijkstra_func(start, goal, terrain_df)

    #  提取子path的 x 和 z 坐标
    x_sub = [point[0] for point in sub_path]
    z_sub = [point[1] for point in sub_path]

    #  绘制子path
    plt.plot(x_sub, z_sub, linestyle=sub_path_style, color=sub_path_color, label=f"Sub-path {i+1}")

#  绘制起点和终点
start_x, start_z = x_main[0], z_main[0]
end_x, end_z = x_main[-1], z_main[-1]
plt.scatter(start_x, start_z, color='green', marker='*', s=100, label='Start Point')
plt.scatter(end_x, end_z, color='purple', marker='X', s=100, label='End Point')
plt.text(start_x, start_z, 'Start', fontsize=10, ha='left', color='green')
plt.text(end_x, end_z, 'End', fontsize=10, ha='left', color='purple')

#  添加坐标轴标签和图例
plt.xlabel("X Coordinate")
plt.ylabel("Z Coordinate")
plt.title("Optimal Path and Sub-paths with Unique Locations")
plt.legend()
plt.grid(True)

#  Show图形
plt.show()
