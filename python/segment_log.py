for env in ['1E', '2E', '3E', '4E']:
    print(f'Processing environment: {env}')
    # !/usr/bin/env python
    #  coding: utf-8

    #  In[1]:


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




    #  In[4]:


    import pandas as pd
    import os

    #  Unique locations mapping
    unique_locations = {
        (-879, 80, -912): 'START',
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

    #  Helper function to find the nearest unique location
    def find_nearest_location(point, unique_locations):
        min_distance = float('inf')
        nearest_location = None

        for loc_coords, loc_name in unique_locations.items():
            distance = ((point[0] - loc_coords[0])**2 + (point[1] - loc_coords[2])**2)**0.5
            if distance < min_distance:
                min_distance = distance
                nearest_location = loc_name

        return nearest_location

    #  Function to process segments and find nearest unique locations
    def process_segments_for_unique_locations(dataframes, unique_locations):
        results = []

        for df_name, df in dataframes.items():
            indices = df[df[3] == 1].index
            segments = []

            for i in range(len(indices) - 1):
                start_idx = indices[i]
                end_idx = indices[i + 1]
                segment = df.iloc[start_idx:end_idx + 1]  #  Include start and end rows
                segments.append(segment)

            for segment_index, segment in enumerate(segments):
                #  Get start and end points
                start_point = tuple(segment.iloc[0][[0, 2]])
                end_point = tuple(segment.iloc[-1][[0, 2]])

                #  Find nearest unique locations
                start_location = find_nearest_location(start_point, unique_locations)
                end_location = find_nearest_location(end_point, unique_locations)

                #  Add to results
                results.append({
                    'df_name': df_name,
                    'segment_index': segment_index,
                    'start_location': start_location,
                    'end_location': end_location
                })

        #  Create and save the DataFrame
        results_df = pd.DataFrame(results)
        output_path = "d:/projects/Gaming/data/{env}/unique_location_segments.csv"
        results_df.to_csv(output_path, index=False)
        print(f"Results saved to {output_path}")

    #  Example call to the function
    #  Assuming `dataframes2` is your dictionary of dataframes
    process_segments_for_unique_locations(dataframes2, unique_locations)