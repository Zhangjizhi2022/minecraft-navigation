for env in ['1E', '2E', '3E', '4E']:
    print(f'Processing environment: {env}')
    # !/usr/bin/env python
    #  coding: utf-8

    #  In[8]:


    import os
    import pandas as pd

    #  定义根目录
    root_dir = 'd:/projects/Gaming/data/{env}'

    #  创建一个字典来存储所有的 DataFrame
    dataframes1 = {}

    #  遍历根目录下的所有子file夹
    for folder_name, subfolders, filenames in os.walk(root_dir):
        if 'training2.csv' in filenames:
        #  构建file的完整path
            file_path1 = os.path.join(folder_name, 'test1.csv')
            df1 = pd.read_csv(file_path1, header=None)






            folder = os.path.basename(folder_name)
        #  以file夹名命名 DataFrame 
            dataframe_name = f"{folder}"
            #  将 DataFrame 存入字典
            dataframes1[dataframe_name] = df1

    #  输出所有的 DataFrame 名称
    # for name in dataframes:
    #     print(f"DataFrame name: {name}")
    #     print(dataframes[name].head())  # Print每个 DataFrame 的前几行data


    #  In[9]:


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


    #  In[ ]:





    #  In[10]:


    unique_location2 = {
        (-849, 77, -912): 'Bed',
        (-879, 79, -852): 'String',
        (-849, 79, -942): 'Saddle',
        (-819, 78, -942): 'PumpkinPie',
        (-879, 75, -972): 'Boat',
        (-909, 77, -972): 'Record',
        (-939, 80, -942): 'Fish',
        (-909, 75, -912): 'Pants',
        (-909, 73, -882): 'Roses',
        (-819, 78, -882): 'Shovel',
        (-939, 77, -882): 'Coal',
        (-849, 74, -852): 'Steak'
    }


    #  In[11]:


    import pandas as pd
    import numpy as np
    import math
    import os

    #  Define the function to calculate the Euclidean distance using specific dimensions
    def euclidean_distance_modified(point1, point2):
        #  Compare point1 (x, y) with point2 (x, z)
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[2]) ** 2)

    #  找到最近的 unique_location2 作为终点
    def find_nearest_location(last_point, unique_locations):
        distances = {}
        for loc in unique_locations.keys():
            try:
                distances[loc] = euclidean_distance_modified(last_point, loc)
            except Exception as e:
                print(f"Error calculating distance for point {last_point} to location {loc}: {e}")
                distances[loc] = float('inf')  #  Set to a large value if any error occurs

        nearest_location, nearest_distance = min(distances.items(), key=lambda x: x[1])
        return nearest_location, nearest_distance

    #  Define the function to process each segment
    def process_segment(segment, terrain_df, filename_prefix):

        #  找到最近的 unique_location2 作为终点
        last_point = tuple(segment.iloc[-1][[0, 2]])
        try:
            nearest_location, nearest_distance = find_nearest_location(last_point, unique_location2)
        except ValueError:
            print(f"Segment {filename_prefix} has no valid nearest location.")
            return

        #  如果最近的 unique_location2 的距离大于 10，标记为不可达
        if nearest_distance > 10:
            print(f"Segment {filename_prefix} is unreachable due to distant nearest location (distance: {nearest_distance}).")
            return

        end_point = (nearest_location[0], nearest_location[2])

        #  使用单循环计算每个点到目标点的 best_cost_from_current
        cost_differences = []
        segment_is_valid = True  #  Flag to determine if the segment should be skipped

        for i in range(len(segment)):
            current_point = tuple(segment.iloc[i][[0, 2]])

            #  计算从当前点到终点的最佳path成本
            _, best_cost_from_current = dijkstra_terrain_with_existence_check_debug(current_point, end_point, terrain_df)

            #  如果某点的 best_cost_from_current 为 None，标记该 segment 为无效
            if best_cost_from_current is None:
               segment_is_valid = False
               break

            #  将最佳path成本添加到 cost_differences
            cost_differences.append(best_cost_from_current)

        #  如果 segment 有效且可达，Save best_cost_from_current 到 CSV file
        if segment_is_valid:
            cost_diff_df = pd.DataFrame(cost_differences, columns=['Best_Cost_From_Current'])
            filename = os.path.join("d:/projects/Gaming/data/{env}/costdiff", f"{filename_prefix}.csv")
            cost_diff_df.to_csv(filename, index=False)
            print(f"Saved: {filename}")
        else:
            print(f"Skipped segment {filename_prefix} due to unreachable points.")


    #  Load terrain data
    terrain_df = pd.read_csv('d:/projects/Gaming/data/{env}/terrain/df_min_y_filtered.csv')

    #  Iterate over each dataframe in dataframes1
    for df_name, df in dataframes1.items():
        #  获取第四列为 1 的行的索引
        indices = df[df[3] == 1].index.tolist()

        #  如果无法按规则剪切，跳过该 df
        if len(indices) % 2 != 0:
            print(f"Skipping {df_name} due to unmatched segment indices.")
            continue

        #  按新规则剪切 segments
        segments = []
        for i in range(len(indices) // 2):
            start_idx = indices[2 * i]
            end_idx = indices[2 * i + 1]
            segment = df.iloc[start_idx:end_idx + 1]  #  Include start and end rows
            if segment.shape[0] > 1:  #  确保至少有两行
                first_row = segment.iloc[0].to_numpy()
                second_row = segment.iloc[1].to_numpy()
                euclidean_distance = np.linalg.norm(first_row - second_row)

                if euclidean_distance > 5:
                    #  删除第一行
                    segment = segment.drop(index=segment.index[0]).reset_index(drop=True)
            segments.append(segment)

        #  Process each segment
        for segment_index, segment in enumerate(segments):
            filename_prefix = f"t{df_name}_{segment_index}"
            process_segment(segment, terrain_df, filename_prefix)


    #  In[34]:


    #  假设 df 是指定的data框
    df = dataframes1['2081']  #  替换为实际 df 名称

    #  获取第四列为 1 的行的索引
    indices = df[df[3] == 1].index.tolist()

    #  确保有足够的点构成至少一个 segment
    if len(indices) < 2:
        print("Not enough points to form a segment.")
    else:
        #  获取第一个 segment 的起始和结束索引
        start_idx = indices[0]
        end_idx = indices[1]

        #  提取第一个 segment
        first_segment = df.iloc[start_idx:end_idx + 1]  #  包括起始和结束行

        #  提取 segment 的 last_point
        last_point = tuple(first_segment.iloc[-1][[0, 2]])

        #  输出 last_point
        print("Last point of the first segment:", last_point)


    #  In[36]:


    #  找到最近的 unique_location2 作为终点
    last_point = tuple(segment.iloc[-1][[0, 2]])
    nearest_location, nearest_distance = find_nearest_location(last_point, unique_location2)


    end_point = nearest_location

        #  使用单循环计算每个点到目标点的 best_cost_from_current
    cost_differences = []


    for i in range(len(segment)):
            current_point = tuple(segment.iloc[i][[0, 2]])

            #  计算从当前点到终点的最佳path成本
            _, best_cost_from_current = dijkstra_terrain_with_existence_check_debug(current_point, end_point, terrain_df)

            #  如果某点的 best_cost_from_current 为 None，标记该 segment 为无效
            # if best_cost_from_current is None:
            #    segment_is_valid = False
            #     break

            #  将最佳path成本添加到 cost_differences
            cost_differences.append(best_cost_from_current)


    #  In[ ]:





    #  In[35]:


    segment = first_segment