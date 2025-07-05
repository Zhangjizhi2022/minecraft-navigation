#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import matplotlib.pyplot as plt

# Define unique locations for each environment
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

# File paths for best scores
file_paths = {
    '1E': 'd:../data/best_scores_1E.csv',
    '2E': 'd:../data/best_scores_2E.csv',
    '3E': 'd:../data/best_scores_3E.csv',
    '4E': 'd:../data/best_scores_4E.csv'
}

# Plot function
def plot_environment(env, file_path):
    # Load best scores
    best_scores = pd.read_csv(file_path)

    # Filter scores below 65
    filtered_scores = best_scores[best_scores['BestScore'] < 70]

    # Get unique locations for the environment
    locations = unique_locations[env]
    coord_lookup = {v: k for k, v in locations.items()}

    plt.figure(figsize=(10, 8))

    # Plot connections for filtered scores
    for _, row in filtered_scores.iterrows():
        start = coord_lookup[row['Start']]
        goal = coord_lookup[row['Goal']]
        plt.plot([start[0], goal[0]], [start[2], goal[2]], color='blue')

    # Plot unique locations as red circles
    for (x, y, z), name in locations.items():
        plt.scatter(x, z, color='red', s=100, marker='o')
        plt.text(x, z, name, fontsize=20, ha='right')

    #plt.xlabel('X Coordinate')
    #plt.ylabel('Z Coordinate')
    #plt.title(f'{env} - Connections for Best Scores < 70')
    # Remove tick labels but keep grid


    plt.grid(True)
    plt.show()

# Generate plots for each environment
for env, path in file_paths.items():
    plot_environment(env, path)





# In[6]:


import pandas as pd
import matplotlib.pyplot as plt

# File paths for best scores and training data
file_paths = {
    '1E': {
        'best_scores': 'd:../data/best_scores_1E.csv',
        'training_data': 'd:../data/E1_training_combined.csv'
    },
    '2E': {
        'best_scores': 'd:../data/best_scores_2E.csv',
        'training_data': 'd:../data/E2_training_combined.csv'
    },
    '3E': {
        'best_scores': 'd:../data/best_scores_3E.csv',
        'training_data': 'd:../data/E3_training_combined.csv'
    },
    '4E': {
        'best_scores': 'd:../data/best_scores_4E.csv',
        'training_data': 'd:../data/E4_training_combined.csv'
    }
}

# List to hold all best scores
all_best_scores = []

# Process each environment
for env, paths in file_paths.items():
    # Load data
    best_scores_df = pd.read_csv(paths['best_scores'])
    training_data_df = pd.read_csv(paths['training_data'])

    # Add a unique identifier to each row in the training data based on its index
    training_data_df['UniqueID'] = training_data_df.index

    # Create a reversed version of best_scores_df for matching swapped start and end locations
    best_scores_reversed_df = best_scores_df.rename(columns={"Start": "Goal", "Goal": "Start"})

    # Combine original and reversed best_scores_df for matching both directions
    combined_best_scores_df = pd.concat([best_scores_df, best_scores_reversed_df], ignore_index=True)

    # Merge the best scores into the training data based on start and end locations
    merged_df = training_data_df.merge(
        combined_best_scores_df,
        left_on=['start_location', 'end_location'],
        right_on=['Start', 'Goal'],
        how='left'
    )

    # Extract the 'BestScore' column and add to the list
    best_scores = merged_df['BestScore'].dropna()
    all_best_scores.extend(best_scores)

# Plot the histogram
plt.figure(figsize=(10, 6))
plt.hist(all_best_scores, bins=35, edgecolor='black', alpha=0.7)
#plt.title('Distribution of Best Scores for Combined Training Data (1E-4E)')
plt.xlabel('Dijkstra Distance', fontsize = 20)
plt.ylabel('Frequency', fontsize = 20)
# Adjust tick label sizes
plt.tick_params(axis='both', labelsize=15)  # Adjusts the font size of tick labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()


# In[11]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# File paths for training data, best scores, and costdiff folders
file_paths = {
    '1E': {
        'training_combined': 'd:../data/E1_training_combined.csv',
        'best_scores': 'd:../data/best_scores_1E.csv',
        'costdiff_folder': 'd:/projects/Gaming/data/1E/costdiff'
    },
    '2E': {
        'training_combined': 'd:../data/E2_training_combined.csv',
        'best_scores': 'd:../data/best_scores_2E.csv',
        'costdiff_folder': 'd:/projects/Gaming/data/2E/costdiff'
    },
    '3E': {
        'training_combined': 'd:../data/E3_training_combined.csv',
        'best_scores': 'd:../data/best_scores_3E.csv',
        'costdiff_folder': 'd:/projects/Gaming/data/3E/costdiff'
    },
    '4E': {
        'training_combined': 'd:../data/E4_training_combined.csv',
        'best_scores': 'd:../data/best_scores_4E.csv',
        'costdiff_folder': 'd:/projects/Gaming/data/4E/costdiff'
    }
}

grouped_data = {}

# Function to parse file names and extract details
def parse_filename(file_name):
    match = re.match(r"(\d{4})_(\d+)(-2)?\.csv$", file_name)
    if not match:
        raise ValueError(f"Filename {file_name} does not match the expected format.")
    df_name = int(match.group(1))
    segment_index = int(match.group(2))
    training = 2 if match.group(3) else 1
    return df_name, segment_index, training

# Process each environment
for env, paths in file_paths.items():
    training_combined_df = pd.read_csv(paths['training_combined'])
    best_scores_df = pd.read_csv(paths['best_scores'])
    costdiff_folder = paths['costdiff_folder']

    csv_files = [f for f in os.listdir(costdiff_folder) if f.endswith('.csv') and f[0].isdigit()]

    for file in csv_files:
        try:
            df_name, segment_index, training = parse_filename(file)
            file_path = os.path.join(costdiff_folder, file)

            matching_record = training_combined_df[
                (training_combined_df['df_name'] == df_name) &
                (training_combined_df['segment_index'] == segment_index) &
                (training_combined_df['training'] == training)
            ]

            if matching_record.empty:
                continue

            start_location = matching_record.iloc[0]['start_location']
            end_location = matching_record.iloc[0]['end_location']

            best_score_record = best_scores_df[
                (best_scores_df['Start'] == start_location) &
                (best_scores_df['Goal'] == end_location)
            ]

            if best_score_record.empty or best_score_record.iloc[0]['BestScore'] >= 60:
                continue

            data = pd.read_csv(file_path)
            if data.isna().any().any():
                continue

            curve = data.iloc[:, 0]
            curve = np.interp(np.linspace(0, len(curve) - 1, 100), np.arange(len(curve)), curve)

            if df_name not in grouped_data:
                grouped_data[df_name] = []
            grouped_data[df_name].append(pd.Series(curve))

        except Exception as e:
            print(f"Error processing {file}: {e}")

# Calculate average curves and filter out those with min < -2
average_curves = {}
for df_name, curves in grouped_data.items():
    avg_curve = pd.concat(curves, axis=1).mean(axis=1)
    avg_curve = avg_curve.apply(lambda x: max(x, 0))  # Replace values less than 0 with 0
    if avg_curve.min() >= -2:  # Exclude curves with min value less than -2
        average_curves[df_name] = avg_curve

# Calculate and display the maximum values relative to 0
max_values = {df_name: avg_curve.max() for df_name, avg_curve in average_curves.items()}

# Plotting
plt.figure(figsize=(12, 8))
for df_name, avg_curve in average_curves.items():
    plt.plot(np.linspace(0, 100, 100), avg_curve)

plt.title('Average Curves')
plt.xlabel('Normalized Index')
plt.ylabel('Value')
plt.legend()
plt.grid()
plt.show()





# In[5]:


import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re

# File paths for training data, best scores, and costdiff folders
file_paths = {
    '1E': {
        'training_combined': 'd:../data/E1_training_combined.csv',
        'best_scores': 'd:../data/best_scores_1E.csv',
        'costdiff_folder': 'd:/projects/Gaming/data/1E/costdiff'
    },
    '2E': {
        'training_combined': 'd:../data/E2_training_combined.csv',
        'best_scores': 'd:../data/best_scores_2E.csv',
        'costdiff_folder': 'd:/projects/Gaming/data/2E/costdiff'
    },
    '3E': {
        'training_combined': 'd:../data/E3_training_combined.csv',
        'best_scores': 'd:../data/best_scores_3E.csv',
        'costdiff_folder': 'd:/projects/Gaming/data/3E/costdiff'
    },
    '4E': {
        'training_combined': 'd:../data/E4_training_combined.csv',
        'best_scores': 'd:../data/best_scores_4E.csv',
        'costdiff_folder': 'd:/projects/Gaming/data/4E/costdiff'
    }
}

grouped_data = {}
max_length = 0

# Function to parse file names and extract details
def parse_filename(file_name):
    match = re.match(r"(\d{4})_(\d+)(-2)?\.csv$", file_name)
    if not match:
        raise ValueError(f"Filename {file_name} does not match the expected format.")
    df_name = int(match.group(1))
    segment_index = int(match.group(2))
    training = 2 if match.group(3) else 1
    return df_name, segment_index, training

# Process each environment to determine the maximum curve length
for env, paths in file_paths.items():
    training_combined_df = pd.read_csv(paths['training_combined'])
    best_scores_df = pd.read_csv(paths['best_scores'])
    costdiff_folder = paths['costdiff_folder']

    csv_files = [f for f in os.listdir(costdiff_folder) if f.endswith('.csv') and f[0].isdigit()]

    for file in csv_files:
        try:
            df_name, segment_index, training = parse_filename(file)
            file_path = os.path.join(costdiff_folder, file)

            matching_record = training_combined_df[
                (training_combined_df['df_name'] == df_name) &
                (training_combined_df['segment_index'] == segment_index) &
                (training_combined_df['training'] == training)
            ]

            if matching_record.empty:
                continue

            start_location = matching_record.iloc[0]['start_location']
            end_location = matching_record.iloc[0]['end_location']

            best_score_record = best_scores_df[
                (best_scores_df['Start'] == start_location) &
                (best_scores_df['Goal'] == end_location)
            ]

            if best_score_record.empty or best_score_record.iloc[0]['BestScore'] >= 60:
                continue

            data = pd.read_csv(file_path)
            if data.isna().any().any():
                continue

            curve = data.iloc[:, 0]
            max_length = max(max_length, len(curve))

        except Exception as e:
            print(f"Error processing {file}: {e}")

# Process files again with known max length and store the curves
for env, paths in file_paths.items():
    training_combined_df = pd.read_csv(paths['training_combined'])
    best_scores_df = pd.read_csv(paths['best_scores'])
    costdiff_folder = paths['costdiff_folder']

    csv_files = [f for f in os.listdir(costdiff_folder) if f.endswith('.csv') and f[0].isdigit()]

    for file in csv_files:
        try:
            df_name, segment_index, training = parse_filename(file)
            file_path = os.path.join(costdiff_folder, file)

            matching_record = training_combined_df[
                (training_combined_df['df_name'] == df_name) &
                (training_combined_df['segment_index'] == segment_index) &
                (training_combined_df['training'] == training)
            ]

            if matching_record.empty:
                continue

            start_location = matching_record.iloc[0]['start_location']
            end_location = matching_record.iloc[0]['end_location']

            best_score_record = best_scores_df[
                (best_scores_df['Start'] == start_location) &
                (best_scores_df['Goal'] == end_location)
            ]

            if best_score_record.empty or best_score_record.iloc[0]['BestScore'] >= 70:
                continue

            data = pd.read_csv(file_path)
            if data.isna().any().any():
                continue

            curve = data.iloc[:, 0]
            curve = np.interp(np.linspace(0, len(curve) - 1, max_length), np.arange(len(curve)), curve)

            if df_name not in grouped_data:
                grouped_data[df_name] = []
            grouped_data[df_name].append(pd.Series(curve))

        except Exception as e:
            print(f"Error processing {file}: {e}")

# Calculate average curves and filter out those with min < -2
average_curves = {}
for df_name, curves in grouped_data.items():
    avg_curve = pd.concat(curves, axis=1).mean(axis=1)
    if avg_curve.min() >= -2:  # Exclude curves with min value less than -2
        average_curves[df_name] = avg_curve

# Plotting
plt.figure(figsize=(12, 8))
for avg_curve in average_curves.values():
    plt.plot(avg_curve)

plt.title('Average Curves Distance < 70 (1E-4E Combined)')
plt.xlabel('Index')
plt.ylabel('Average Value')
plt.grid()
plt.show()



# In[26]:


# Prepare data for CSV export
output_data = pd.DataFrame(average_curves).transpose()
output_data.reset_index(inplace=True)
output_data.rename(columns={'index': 'df_name'}, inplace=True)

# Save to CSV
output_csv_path = 'd:../data/combined_average_curves.csv'
output_data.to_csv(output_csv_path, index=False)

print(f"CSV file saved at: {output_csv_path}")


# In[28]:


# Calculate the number of removed curves for each df_name
removed_curves_count = {}

for df_name, curves in grouped_data.items():
    removed_count = sum(curve.min() < -2 for curve in curves)
    removed_curves_count[df_name] = removed_count

# Convert to DataFrame and save as CSV
removed_curves_df = pd.DataFrame(list(removed_curves_count.items()), columns=['df_name', 'Removed_curves'])
removed_curves_df.sort_values(by='df_name', inplace=True)

output_path = 'd:../data/removed_curves_summary.csv'
removed_curves_df.to_csv(output_path, index=False)

print(f"Removed curves summary saved to {output_path}")

