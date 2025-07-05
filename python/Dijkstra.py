for env in ['1E', '2E', '3E', '4E']:
    print(f'Processing environment: {env}')
    # !/usr/bin/env python
    #  coding: utf-8

    #  In[4]:


    import os
    import pandas as pd

    #  定义根目录
    root_dir = 'd:/projects/Gaming/data/{env}'

    #  创建一个字典来存储所有的 DataFrame
    dataframes1 = {}
    dataframes2 = {}

    #  遍历根目录下的所有子file夹
    for folder_name, subfolders, filenames in os.walk(root_dir):
        if 'training2.csv' in filenames:
        #  构建file的完整path
            file_path1 = os.path.join(folder_name, 'training1.csv')
            df1 = pd.read_csv(file_path1, header=None)

            file_path2 = os.path.join(folder_name, 'training2.csv')
            df2 = pd.read_csv(file_path2, header=None)

            expected_first_row = [-879, 80, -912, 1]
            if not df1.iloc[0].tolist() == expected_first_row:
             #  如果第一行不符合预期，则替换第一行并删除原第一行
                df1.iloc[0] = expected_first_row
                df1 = df1.drop(1).reset_index(drop=True)

            if not df2.iloc[0].tolist() == expected_first_row:
             #  如果第一行不符合预期，则替换第一行并删除原第一行
                df2.iloc[0] = expected_first_row
                df2 = df2.drop(1).reset_index(drop=True)    

            folder = os.path.basename(folder_name)
        #  以file夹名命名 DataFrame 
            dataframe_name = f"{folder}"
            #  将 DataFrame 存入字典
            dataframes1[dataframe_name] = df1
            dataframes2[dataframe_name] = df2




    #  In[5]:


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


    #  In[7]:


    import pandas as pd
    import math
    import os

    #  Define the function to process each segment
    def process_segment(segment, terrain_df, filename_prefix):
        #  获取path的起始和结束坐标
        start_point = tuple(segment.iloc[0][[0, 2]])  #  Using 0 and 2 to represent x and z
        end_point = tuple(segment.iloc[-1][[0, 2]])

        #  Calculate the optimal path and cost using the function provided
        optimal_path, optimal_cost = dijkstra_terrain_with_existence_check_debug(start_point, end_point, terrain_df)

        #  预先计算每对相邻点的移动成本
        pairwise_costs = []
        for i in range(len(segment) - 1):
            current_point = tuple(segment.iloc[i][[0, 2]])
            next_point = tuple(segment.iloc[i + 1][[0, 2]])

            try:
                #  获取当前点和下一个点的高度
                current_y = terrain_df[(terrain_df['x'] == current_point[0]) & (terrain_df['z'] == current_point[1])]['y'].values[0]
                next_y = terrain_df[(terrain_df['x'] == next_point[0]) & (terrain_df['z'] == next_point[1])]['y'].values[0]
            except IndexError:
                pairwise_costs.append(None)
                continue

            height_diff = next_y - current_y

            #  计算两点之间的欧氏距离作为 base_cost
            base_cost = math.sqrt((next_point[0] - current_point[0])**2 + (next_point[1] - current_point[1])**2)

            #  计算 step_cost
            if height_diff == 0:
                step_cost = base_cost
            elif height_diff < 0:
                step_cost = base_cost + (1.5 * abs(height_diff))  #  下坡
            elif height_diff > 0:
                step_cost = base_cost + (2.5 * height_diff)  #  上坡

            pairwise_costs.append(step_cost)

        #  使用单循环计算每个点到目标点的 actual_cost，并与最优成本Compare
        cost_differences = []
        segment_is_valid = True  #  Flag to determine if the segment should be skipped

        for i in range(len(segment) - 1):
            current_point = tuple(segment.iloc[i][[0, 2]])

            #  从第 i 点到终点的实际成本
            actual_cost = 0
            is_reachable = True

            #  累加从 i 点到终点的 cost
            for j in range(i, len(segment) - 1):
                step_cost = pairwise_costs[j]
                if step_cost is None:
                    is_reachable = False
                    break  #  如果某一段不可达，则该path不可达
                actual_cost += step_cost

            #  计算最优path成本
            _, best_cost_from_current = dijkstra_terrain_with_existence_check_debug(current_point, end_point, terrain_df)

            #  如果任何一点的 best_cost_from_current 为 None，跳过该 segment
            if best_cost_from_current is None:
                segment_is_valid = False
                break

            #  计算差值
            if is_reachable:
                cost_difference = actual_cost - best_cost_from_current
                cost_differences.append(cost_difference)
            else:
                cost_differences.append(None)  #  无法到达时，标记为 None

        #  如果 segment 有效且可达，Save cost differences 到 CSV file
        if segment_is_valid:
            cost_diff_df = pd.DataFrame(cost_differences, columns=['Cost_Difference'])
            filename = os.path.join("d:/projects/Gaming/data/{env}/costdiff", f"{filename_prefix}.csv")
            cost_diff_df.to_csv(filename, index=False)
            print(f"Saved: {filename}")
        else:
            print(f"Skipped segment {filename_prefix} due to unreachable points.")

    #  Load terrain data
    terrain_df = pd.read_csv('d:/projects/Gaming/data/{env}/terrain/df_min_y_filtered.csv')

    #  Iterate over each dataframe in dataframes2
    for df_name, df in dataframes1.items():
        #  Get the indices where the fourth column is 1
        indices = df[df[3] == 1].index
        segments = []

        for i in range(len(indices) - 1):
            start_idx = indices[i]
            end_idx = indices[i + 1]
            segment = df.iloc[start_idx:end_idx + 1]  #  Include start and end rows
            segments.append(segment)

        #  Process each segment
        for segment_index, segment in enumerate(segments):
            filename_prefix = f"{df_name}_{segment_index}"
            process_segment(segment, terrain_df, filename_prefix)