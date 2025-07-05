for env in ['1E', '2E', '3E', '4E']:
    print(f'Processing environment: {env}')
    # !/usr/bin/env python
    #  coding: utf-8

    #  In[2]:


    import heapq  #  Importing heapq as it was missed earlier
    import pandas as pd
    import itertools

    #  Load the terrain data from the uploaded file
    terrain_df = pd.read_csv('d:/projects/Gaming/data/{env}/terrain/df_min_y_filtered.csv')

    #  Unique locations with their labels
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

    #  Define the Dijkstra function
    def dijkstra_terrain_with_existence_check_debug(start, goal, terrain_df):
        terrain_dict = {(row['x'], row['z']): row['y'] for _, row in terrain_df.iterrows()}
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
        pq = [(0, start)]
        distances = {start: 0}
        previous = {start: None}

        while pq:
            current_distance, current_node = heapq.heappop(pq)
            if current_node == goal:
                break
            for direction in directions:
                neighbor = (current_node[0] + direction[0], current_node[1] + direction[1])
                if neighbor in terrain_dict:
                    current_y = terrain_dict[current_node]
                    neighbor_y = terrain_dict[neighbor]
                    height_diff = neighbor_y - current_y
                    if height_diff > 2 or height_diff < -3:
                        continue
                    is_diagonal = direction in [(-1, -1), (-1, 1), (1, -1), (1, 1)]
                    base_cost = 1.41 if is_diagonal else 1
                    if height_diff == 0:
                        distance = current_distance + base_cost
                    elif height_diff < 0:
                        distance = current_distance + base_cost + (1.5 * abs(height_diff))
                    elif height_diff > 0:
                        distance = current_distance + base_cost + (2.5 * height_diff)
                    if neighbor not in distances or distance < distances[neighbor]:
                        distances[neighbor] = distance
                        heapq.heappush(pq, (distance, neighbor))
                        previous[neighbor] = current_node

        path = []
        current = goal
        if current not in previous:
            return "No path found", None
        while current is not None:
            path.append(current)
            current = previous.get(current)
        path.reverse()
        best_score = distances[goal] if goal in distances else None
        return path, best_score

    #  Generate combinations of unique locations
    location_keys = list(unique_locations.keys())
    pairs = list(itertools.combinations(location_keys, 2))

    #  Compute best scores for each pair
    results = []
    for start, goal in pairs:
        path, best_score = dijkstra_terrain_with_existence_check_debug((start[0], start[2]), (goal[0], goal[2]), terrain_df)
        results.append({
            'Start': unique_locations[start],
            'Goal': unique_locations[goal],
            'BestScore': best_score
        })

    #  Create a DataFrame from results
    results_df = pd.DataFrame(results)

    #  Save the results to a CSV file
    output_path = 'best_scores_{env}.csv'
    results_df.to_csv(output_path, index=False)

    output_path


    #  In[3]:


    import matplotlib.pyplot as plt

    #  Load the results CSV file containing the best scores
    results_df = pd.read_csv('d:/projects/Gaming/data/{env}/Analysis/best_scores_{env}.csv')

    #  Extract the 'BestScore' column for the histogram
    best_scores = results_df['BestScore'].dropna()  #  Drop NaN values if any

    #  Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(best_scores, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Best Scores Between Unique Locations')
    plt.xlabel('Best Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


    #  In[4]:


    import pandas as pd

    #  Reload necessary files
    best_scores_df = pd.read_csv('d:/projects/Gaming/data/{env}/Analysis/best_scores_{env}.csv')
    training_data_df = pd.read_csv('d:/projects/Gaming/data/{env}/E3_training_combined.csv')

    #  Add a unique identifier to each row in the training data based on its index
    training_data_df['UniqueID'] = training_data_df.index

    #  Create a reversed version of best_scores_df for matching swapped start and end locations
    best_scores_reversed_df = best_scores_df.rename(columns={"Start": "Goal", "Goal": "Start"})

    #  Combine original and reversed best_scores_df for matching both directions
    combined_best_scores_df = pd.concat([best_scores_df, best_scores_reversed_df], ignore_index=True)

    #  Merge the best scores into the training data based on start and end locations, preserving duplicate rows
    merged_df = training_data_df.merge(
        combined_best_scores_df,
        left_on=['start_location', 'end_location'],
        right_on=['Start', 'Goal'],
        how='left'
    )

    #  Extract the 'BestScore' column for the histogram
    best_scores = merged_df['BestScore'].dropna()  #  Drop NaN values if any

    #  Plot the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(best_scores, bins=30, edgecolor='black', alpha=0.7)
    plt.title('Distribution of Best Scores for Training Data')
    plt.xlabel('Best Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


    #  In[5]:


    #  Merge the best scores into the training data based on start and end locations
    merged_df = training_data_df.merge(
        best_scores_df,
        left_on=['start_location', 'end_location'],
        right_on=['Start', 'Goal'],
        how='left'
    )

    #  Count distances above and below the threshold of 60
    below_60_count = (merged_df['BestScore'] < 60).sum()
    above_60_count = (merged_df['BestScore'] >= 60).sum()

    below_60_count, above_60_count