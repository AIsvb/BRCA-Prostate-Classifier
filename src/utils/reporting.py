
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator
from pathlib import Path
import shutil
import pandas as pd
from reportlab.platypus import SimpleDocTemplate, Paragraph, Image, Table, TableStyle, PageBreak
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from tensorboard.backend.event_processing import event_accumulator
from reportlab.lib.styles import getSampleStyleSheet
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import numpy as np


def csv_to_pdf_table(file, from_csv=True):
    """
    Converts a CSV file into a reportlab Table format suitable for inclusion in a PDF.
    
    Args:
        csv_file (str): Path to the CSV file.
    
    Returns:
        Table: A Table object that can be added to a PDF document.
    """
    if from_csv:
        # Load CSV into DataFrame
        file = pd.read_csv(file)
    
    # Prepare data for Table
    data = [file.columns.tolist()] + file.values.tolist()
    
    # Create the table
    table = Table(data)
    
    # Apply a basic style (optional)
    style = TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black),
    ])
    
    table.setStyle(style)
    
    return table

def plot_tb_scalars(tb_folders):
    """
    Create a single plot for matching scalars found in the TensorBoard logs of the given folders.
    
    Args:
        tb_folders (list): List of folder paths containing TensorBoard files.
    
    Returns:
        List: A list of file paths to saved plots.
    """
    plot_files = []
    
    # Dictionary to store data for matching scalars
    scalar_data = {}

    # Loop through each TensorBoard folder
    for tb_folder in tb_folders:
        # Load TensorBoard events
        ea = event_accumulator.EventAccumulator(tb_folder)
        ea.Reload()

        # Get scalar tags
        scalar_tags = ea.Tags()['scalars']
        
        for scalar in scalar_tags:

            if 'final' in scalar:
                continue 

            # Retrieve scalar values for the tag
            scalar_events = ea.Scalars(scalar)

            # Extract steps and values from ScalarEvent objects
            steps = [event.step for event in scalar_events]
            values = [event.value for event in scalar_events]
            
            if scalar not in scalar_data:
                scalar_data[scalar] = []
            
            # Store the scalar values along with the steps and folder name
            scalar_data[scalar].append({
                "steps": steps,
                "values": values,
                "label": Path(tb_folder).name  # Use the folder name as a label
            })

    dest = Path('temp_folder')
    dest.mkdir(exist_ok=True, parents=True)

    # Now plot the scalars
    for scalar, runs in scalar_data.items():
        plt.figure()
        for run in runs:
            plt.plot(run["steps"], run["values"], label=run["label"])
        
        plt.xlabel('Steps')
        plt.ylabel(scalar)
        plt.title(f'{scalar} across Different Runs')
        plt.legend()
        
        # Save the plot to a file
        plot_filename = f"{scalar.replace('/', '_')}_comparison.png"
        plt.savefig(dest / plot_filename)
        plot_files.append(dest / plot_filename)
        plt.close()
    
    return plot_files

def create_pdf_from_folder(folder, csv_files=['summary.csv', 'eval_overview.csv', 'arguments.csv']):
    """
    Creates a PDF report from TensorBoard logs and CSV files in a given folder.
    
    Args:
        folder (str): Path to the folder containing the necessary files and subfolders.
    """

    # Path to the CSV files
    tables = []
    for csv_ in csv_files:
        tables.append(Path(folder) / csv_)

    # Prepare the PDF document
    pdf_filename = str(Path(folder) / 'training_report.pdf')
    doc = SimpleDocTemplate(pdf_filename, pagesize=letter)
    elements = []
    
    # Title
    styles = getSampleStyleSheet()
    title = Paragraph(f"Folder {Path(folder).name}", styles["Title"])
    elements.append(title)
    
    # Add tables from CSV files
    for table in tables:
        if table.is_file():
            elements.append(csv_to_pdf_table(table))
            elements.append(PageBreak())

    for table in Path(folder).rglob('*pkl'):
        x = csv_to_pdf_table(fold2frame(table), False)
        elements.append(x)
        elements.append(PageBreak())
        
    # Get all subfolders containing TensorBoard files
    tb_folders = [str(p) for p in list(Path(folder).rglob('*events.out.tfevents.*'))]
    
    # Add TensorBoard plots for matching scalars
    if tb_folders:
        plot_files = plot_tb_scalars(tb_folders)
        
        # Add each plot image to the PDF
        for idx, plot_file in enumerate(plot_files):
            img = Image(str(plot_file), width=400, height=300)
            elements.append(img)

            if (idx + 1) % 2 == 0 and idx != 0:
                elements.append(PageBreak())
    
    # Build the PDF document
    doc.build(elements)
    
    shutil.rmtree('temp_folder')
    print(f"PDF report created: {pdf_filename}")

def fold2frame(pickle_file):

    with open(pickle_file, 'rb') as file:
        x = pickle.load(file)

    data = {
        'slide_id': [],
        'class_0_prob': [],
        'class_1_prob': [],
        'label': [],
        'pred': []
    }

    for slide, res in x.items():
        data['class_0_prob'].append(res['prob'][0,0])
        data['class_1_prob'].append(res['prob'][0,1])
        data['pred'].append(int(np.argmax(res['prob'])))
        data['label'].append(res['label'])
        data['slide_id'].append(slide)

    return pd.DataFrame(data)

def aggregate(root):

    frequency = defaultdict(int)
    correct = defaultdict(int)
    labels = dict()
    for pickle_file in Path(root).rglob('*pkl'):

        with open(pickle_file, 'rb') as file:
            x = pickle.load(file)

        for slide, res in x.items():   
            frequency[slide] += 1
            if res['label'] == int(np.argmax(res['prob'])):
                correct[slide] += 1
            labels[slide] = res['label']

    data = defaultdict(list)
    for key, val in frequency.items():
        data['slide_id'].append(key)
        data['frequency'].append(val)
        data['correct'].append(correct[key])
        data['frac_correct'].append(correct[key] / val)
        data['label'].append(labels[key])

    return pd.DataFrame(data)

def process_folder(folder):

    # -- automatic report construction
    aggregate(folder).to_csv(folder / 'eval_overview.csv')

    df = pd.read_csv(folder / 'summary.csv')
    mean = pd.DataFrame([df[['test_auc', 'test_acc']].mean()], index=['Mean'])
    pd.concat([df, mean]).to_csv(folder / 'summary.csv')

    create_pdf_from_folder(folder)
