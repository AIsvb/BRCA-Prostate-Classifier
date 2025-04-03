import shutil
import pandas as pd
from pathlib import Path

def organize_images_by_label(df, image_files, output_dir, no_image_file):

    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)

    label_0_patient_ids = {item for item in set(df[df['set_label'] == 0]['patient_id'].unique()) if isinstance(item, str)}
    label_1_patient_ids = {item for item in set(df[df['set_label'] == 1]['patient_id'].unique()) if isinstance(item, str)}

    print(f'Found {len(label_0_patient_ids)} cases for label 0')
    print(f'Found {len(label_1_patient_ids)} cases for label 1')

    label_0_images = []
    label_1_images = []

    case_0_matched = set()
    case_1_matched = set()

    process_data = {'patient_id': [], 'slide_id': [], 'label':[]}

    for image_file in image_files:

        for id in label_0_patient_ids:
            if isinstance(id, str) and id in image_file:
                label_0_images.append(image_file)
                case_0_matched.update([id])
                process_data['patient_id'].append(id)
                process_data['slide_id'].append(image_file)
                process_data['label'].append(0)
                break

        for id in label_1_patient_ids:
            if isinstance(id, str) and id in image_file:
                label_1_images.append(image_file)
                case_1_matched.update([id])
                process_data['patient_id'].append(id)
                process_data['slide_id'].append(image_file)
                process_data['label'].append(1)
                break
        
    print(f'Found {len(label_0_images)} images for label 0')
    print(f'Found {len(label_1_images)} images for label 1')
    
    with open(no_image_file, 'w') as file:

        file.write('LABEL 0\n')
        for case_ in sorted(label_0_patient_ids):
            if case_ not in case_0_matched:
                file.write(f'{case_}\n')

        file.write('\nLABEL 1\n')
        for case_ in sorted(label_1_patient_ids):
            if case_ not in case_1_matched:
                file.write(f'{case_}\n')

    filtered_df = df[df['patient_id'].isin(case_0_matched.union(case_1_matched))]
    filtered_df.rename(columns={'PAI-nummer': 'slide_id', 'set_label': 'label'}, inplace=True)

    filtered_df.to_csv(dest / 'data_labels.csv')

    process_data = pd.DataFrame(process_data)
    process_data['use'] = 1
    process_data.to_csv(dest / 'tiling_process_list.csv')

if __name__ == "__main__":

    excel_file = ...
    image_files = ...

    with open(image_files, 'r') as f:
        image_files = [i.replace('\n', '') for i in f.readlines()]

    sheet_name = 'BRCA2'
    set_name = 'BRCA2'
    output_dir = f'{set_name}_data'
    no_image_file = f'unmatched_cases_{set_name}.txt'

    frame = pd.read_excel(excel_file, sheet_name=sheet_name)
    organize_images_by_label(frame, image_files, output_dir, no_image_file)
