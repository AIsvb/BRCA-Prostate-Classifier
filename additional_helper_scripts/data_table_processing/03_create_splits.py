import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

def create_balanced_patient_train_test_splits_grouped(
    csv_path, 
    n_splits, 
    hospital_groups, 
    label_column='label', 
    test_size=0.2, 
    random_state=None
):
    """
    Creates N balanced train-test splits, ensuring that all slides from a single patient
    are exclusively in either the train or test set. Splits are performed per hospital group.

    Parameters:
        csv_path (str): Path to the CSV file.
        n_splits (int): Number of splits to generate.
        hospital_groups (list of lists): List where each sublist contains hospital codes to group together for splitting.
        label_column (str): Name of the column containing labels (default: 'label').
        test_size (float): Proportion of the dataset to include in the test split (default: 0.2).
        random_state (int or None): Random seed for reproducibility (default: None).

    Returns:
        dict: Dictionary containing train and test splits for each iteration.
    """
    # Load data
    df = pd.read_csv(csv_path)

    # Filter rows where 'use' == 1
    df = df[df['use'] == 1]

    # Extract hospital code from slide_id
    #df['hospital_code'] = df['slide_id'].str.split('_').str[0]
    df['hospital_code'] = df['slide_id'].str[:4]

    # Store splits
    splits = {}
    for split_idx in range(n_splits):
        train_dfs = []
        test_dfs = []

        # Perform split for each hospital group
        for group_idx, hospital_group in enumerate(hospital_groups):
            group_data = df[df['hospital_code'].isin(hospital_group)]

            # Skip group if no data
            if group_data.empty:
                continue

            # Group by patient_id
            patients = group_data['patient_id'].unique()
            patient_data = group_data.groupby('patient_id')

            # Stratify based on patient labels
            patient_labels = patient_data[label_column].first()  # Get one label per patient

            if patient_labels.nunique() > 1:
                # Perform stratified split on patients
                train_patients, test_patients = train_test_split(
                    patients, 
                    test_size=test_size, 
                    random_state=(random_state + split_idx + group_idx) if random_state is not None else None,
                    stratify=patient_labels
                )
            else:
                # If stratification is not possible (only one label), split without it
                train_patients, test_patients = train_test_split(
                    patients, 
                    test_size=test_size, 
                    random_state=(random_state + split_idx + group_idx) if random_state is not None else None
                )

            # Select slides corresponding to train and test patients
            train_dfs.append(group_data[group_data['patient_id'].isin(train_patients)])
            test_dfs.append(group_data[group_data['patient_id'].isin(test_patients)])

        # Aggregate results
        train_split = pd.concat(train_dfs, ignore_index=True)
        test_split = pd.concat(test_dfs, ignore_index=True)

        # Store in dictionary
        splits[f'fold_{split_idx+1}'] = {'train': train_split, 'test': test_split}

    return splits

def save_splits_to_csv(splits, output_dir):
    """
    Saves train-test splits to CSV files in the required format.
    The CSVs will contain three columns: 'train', 'test', and 'val',
    where 'val' is identical to 'test'.

    Parameters:
        splits (dict): Dictionary containing splits with keys like 'split_1', 'split_2', etc.
                       Each value is a dictionary with 'train' and 'test' DataFrames.
        output_dir (str): Directory where the CSV files will be saved.

    Returns:
        None
    """
    import os

    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    for split_name, split_data in splits.items():
        # Prepare data for the CSV
        train_files = split_data['train']['slide_id'].tolist()
        test_files = split_data['test']['slide_id'].tolist()

        # Create a DataFrame with the required columns
        max_len = max(len(train_files), len(test_files))
        train_files.extend([None] * (max_len - len(train_files)))
        test_files.extend([None] * (max_len - len(test_files)))

        split_df = pd.DataFrame({
            'train': train_files,
            'test': test_files,
            'val': test_files  # 'val' column is identical to 'test'
        })

        # Save to CSV
        split_path = os.path.join(output_dir, f"{split_name}.csv")
        split_df.to_csv(split_path, index=False)

        print(f"Saved {split_name} to {split_path}")

def write_split_statistics(splits, output_file):
    """
    Writes the number of cases (unique patient IDs) and slides per label for each split to a file.
    Also prints the statistics to the console.

    Parameters:
        splits (dict): Dictionary containing splits with keys like 'split_1', 'split_2', etc.
                       Each value is a dictionary with 'train' and 'test' DataFrames.
        output_file (str): Path to the output text file.

    Returns:
        None
    """
    with open(output_file, 'w') as f:
        for split_name, split_data in splits.items():
            # Write and print split name
            split_header = f"\nStatistics for {split_name}:\n"
            print(split_header, end='')
            f.write(split_header)
            
            for split_type, df in split_data.items():  # 'train' and 'test'
                split_type_header = f"  {split_type.capitalize()} Split:\n"
                print(split_type_header, end='')
                f.write(split_type_header)
                
                # Count cases and slides per label
                label_stats = df.groupby('label').agg(
                    cases=('patient_id', 'nunique'),  # Unique patient IDs
                    slides=('slide_id', 'count')     # Total slide count
                ).reset_index()
                
                # Write and print statistics
                for _, row in label_stats.iterrows():
                    stats_line = f"    Label {row['label']}: {row['cases']} cases, {row['slides']} slides\n"
                    print(stats_line, end='')
                    f.write(stats_line)


if __name__ == "__main__":

    csv_path = '<PATH_TO_CSV_FILE>'     # Each row should correspond to a slide. Columns must include 'use' (whether to use this slide) and 'patient_id' (for stratification by patient)
    N = 8                               # number of random splits
    test_size = 0.2                     # percentage to keep for testing
    random_state = 9
    output_dir = 'splits'
    hospital_groups = [['NL01', 'NL03', 'NL05', 'NL06', 'NL10', 'NL11', 'NL12', 'NL14', 'NL32'], ['NL02']]  # balance the splits per hostpital (or hospital group)

    splits = create_balanced_patient_train_test_splits_grouped(
        csv_path=csv_path,
        n_splits=N,
        hospital_groups=hospital_groups,
        test_size=test_size,
        random_state=random_state
    )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    write_split_statistics(splits, Path(output_dir) / 'summary.txt')
    save_splits_to_csv(splits, output_dir)