
'''
Legend for Gene status columns:

nan : Not tested / no information available
0   : Tested Wildtype
1   : Tested Mutant
2   : Tested VUS
3   : Tested Mutant + VUS

'''

import pandas as pd
import numpy as np


def slide_id2case_id(slide_id):
    if isinstance(slide_id, str):

        if 'NL32' in slide_id:
            return '-'.join(slide_id.split('-')[0:2])
        
        return '_'.join(slide_id.split('-')[0].split('_')[0:2])

def process_brca_column(df, gene, source_col="BRCA1/2 mutatie aanwezig", vus_col="VUS"):
    """
    Creates a new column for BRCA1 or BRCA2 based on the specified column,
    handling uncertain values and cross-checking with the VUS column.

    Parameters:
    - df: pandas DataFrame
    - gene: str, "BRCA1" or "BRCA2" to indicate which gene to process
    - source_col: str, name of the column indicating BRCA1/2 status
    - vus_col: str, name of the column containing variants of unknown significance

    Returns:
    - pandas DataFrame with an additional column named f"{gene}_status".
    """
    if gene not in ["BRCA1", "BRCA2"]:
        raise ValueError("gene parameter must be either 'BRCA1' or 'BRCA2'")

    if gene == "BRCA1":
        opposite = "BRCA2"
    else:
        opposite = "BRCA1"

    def get_status(row):
        try:
            source_value = row.get(source_col, None)
            vus_value = row.get(vus_col, "")
            vus_present = gene in str(vus_value)  # Check if gene is in VUS column
            
            if pd.isna(source_value):
                return np.nan
            
            # Handle uncertain values
            if isinstance(source_value, str) and "?" in source_value:
                return np.nan
            
            # Check if the gene is present
            if source_value == gene:
                return 3 if vus_present else 1
            elif vus_present:
                return 2
            elif source_value == 0:
                return 0
            elif source_value == opposite:
                return 0
            else:
                return np.nan
        except Exception as e:
            print(f"Error processing row: {row}. Error: {e}")
            return np.nan

    # Apply the function to create the new column
    status_col = f"{gene}_status"
    df[status_col] = df.apply(get_status, axis=1)
    return df

def select_rows_with_specific_mutations(df, include_genes, exclude_genes, include_values=[1,2,3], exclude_values=[1,2,3]):
    """
    Selects rows where mutations are present in some genes (include_genes) 
    but absent in others (exclude_genes).

    Parameters:
    - df: pandas DataFrame
    - include_genes: list of str, genes that must have mutations (value 1 or 3)
    - exclude_genes: list of str, genes that must not have mutations (value 1 or 3)

    Returns:
    - pandas DataFrame with rows meeting the criteria.
    """
    # Check for mutations in include_genes
    include_filter = df[[f"{gene}_status" for gene in include_genes]].apply(
        lambda row: all(val in include_values for val in row), axis=1
    )
    
    # Check for absence of mutations in exclude_genes
    exclude_filter = df[[f"{gene}_status" for gene in exclude_genes]].apply(
        lambda row: all(val not in exclude_values for val in row), axis=1
    )
    
    # Combine filters
    combined_filter = include_filter & exclude_filter
    result = df[combined_filter].dropna(subset=[f"{gene}_status" for gene in include_genes + exclude_genes])

    return result

def remove_LOH_and_amplifications(df, genes, col1='homozygote deletie', col2='amplificatie'):
    """
    Selects rows that do not contain any of the strings in exclude_strings in either of the two columns.
    If the value is a string, it checks that none of the exclude_strings are substrings.
    
    Parameters:
    - df: pandas DataFrame
    - col1, col2: names of the columns to check
    - exclude_strings: list of strings to exclude from both columns

    Returns:
    - pandas DataFrame with rows that do not contain any of exclude_strings in either of the columns.
    """
    # Function to check if any of the exclude_strings are in the value (ignores NaN and non-string values)
    def check_exclude(value):
        if isinstance(value, str):
            return not any(gene in value for gene in genes)
        return True  # For NaN or non-string values, keep the row

    # Apply the check to both columns
    mask = df[col1].apply(check_exclude) & df[col2].apply(check_exclude)

    # Return the filtered rows
    return df[mask]

def write_dataframes_to_excel(file_path, dataframes, sheet_names):
    """
    Writes multiple dataframes to a single Excel file, each in a separate sheet.

    Parameters:
    - file_path: str, the path to save the Excel file.
    - dataframes: list of pandas DataFrame, the dataframes to write.
    - sheet_names: list of str, the names of the sheets.

    Returns:
    - None
    """
    if len(dataframes) != len(sheet_names):
        raise ValueError("The number of dataframes and sheet names must be the same.")
    
    with pd.ExcelWriter(file_path, engine="openpyxl") as writer:
        for df, sheet_name in zip(dataframes, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name, index=False)

def set_label(df, columns, value):
    """
    Adds a column to the dataframe that has value 1 if the specified value
    is found in any of the selected columns, and 0 otherwise.

    Parameters:
    - df: pandas DataFrame
    - columns: list of str, column names to check
    - value: the value to look for in the specified columns

    Returns:
    - pandas DataFrame with the new column added
    """
    df['set_label'] = df[columns].apply(
        lambda row: 1 if value in row.values else 0, axis=1
    )
    return df

## STEP 1: JOIN ERASMUS WITH REST
labels = pd.read_excel('data_processing/data/labels Jacqueline 2025-01-16.xlsx')
labels_sander = pd.read_excel('data_processing/data/labels_230924.xlsx', sheet_name='Erasmus_moleculair')

overlapping_columns = labels.columns.intersection(labels_sander.columns)
labels = pd.concat([labels, labels_sander[overlapping_columns]], ignore_index=True)

# Only keep the needle biopsies
labels = labels[~labels['Soort pathologie '].isin(['Prostatectomie', 'TURP'])].reset_index(drop=True)

## STEP 2: ADD PATIENT ID
labels['patient_id'] = labels['PAI-nummer'].apply(slide_id2case_id)

## STEP 3: ADD SEPERATE COLUMNS FOR STATUS OF SPECIFIED GENES
for gene in ['BRCA1', 'BRCA2']:
    labels = process_brca_column(labels, gene)

## STEP 4: RETRIEVE SUBSET OF THE CASES (here --> only pathogenic BRCA2, no VUS or LOH, no BRCA1 aberrations)

# BRCA2
brca2_set = remove_LOH_and_amplifications(select_rows_with_specific_mutations(labels, include_genes=['BRCA2'], exclude_genes=['BRCA1'], include_values=[0, 1], exclude_values=[1,2,3]), genes=['BRCA2'])
brca2_set = set_label(brca2_set, ['BRCA2_status'], 1)

# Save to excel file
file_path = "<FILENAME>.xlsx"
write_dataframes_to_excel(file_path, [brca2_set], ["BRCA2"])