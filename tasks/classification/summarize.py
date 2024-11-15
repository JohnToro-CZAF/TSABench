import os
import json
from collections import defaultdict
from html import escape

def flatten_dict(d, parent_key='', sep='.'):
    """
    Recursively flattens a nested dictionary.
    """
    items = {}
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.update(flatten_dict(v, new_key, sep=sep))
        else:
            items[new_key] = v
    return items

def collect_experiment_data(exp_folder):
    """
    Collects data from a single experiment folder.
    """
    exp_data = {}
    exp_data['Experiment Name'] = os.path.basename(exp_folder)
    exp_data['Experiment Folder'] = exp_folder  # Store folder path for later use
    # Read config.json
    config_path = os.path.join(exp_folder, 'config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        flat_config = flatten_dict(config)
        exp_data.update(flat_config)
    else:
        print(f'Warning: config.json not found in {exp_folder}')
        return None

    # Read metrics.json
    metrics_path = os.path.join(exp_folder, 'metrics.json')
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        # Extract final metrics
        exp_data['Final Training Loss'] = metrics['train_loss'][-1] if metrics['train_loss'] else None
        exp_data['Final Validation Loss'] = metrics['val_loss'][-1] if metrics['val_loss'] else None
        exp_data['Final Testing Loss'] = metrics['test_loss'][-1] if metrics['test_loss'] else None

        # Assuming 'accuracy' is one of the metrics
        if 'accuracy' in metrics['train_metrics']:
            exp_data['Final Training Accuracy'] = metrics['train_metrics']['accuracy'][-1] if metrics['train_metrics']['accuracy'] else None
            exp_data['Final Validation Accuracy'] = metrics['val_metrics']['accuracy'][-1] if metrics['val_metrics']['accuracy'] else None
            exp_data['Final Testing Accuracy'] = metrics['test_metrics']['accuracy'][-1] if metrics['test_metrics']['accuracy'] else None
        else:
            exp_data['Final Training Accuracy'] = None
            exp_data['Final Validation Accuracy'] = None
    else:
        print(f'Warning: metrics.json not found in {exp_folder}')
        return None

    # Collect image paths
    images = [f for f in os.listdir(exp_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.gif'))]
    exp_data['Images'] = [os.path.join(exp_folder, img) for img in images]

    return exp_data

def generate_experiment_detail_page(exp_data, output_folder):
    """
    Generates a detailed HTML page for a single experiment.
    """
    exp_name = exp_data['Experiment Name']
    detail_filename = f"{exp_name}_details.html"
    detail_filepath = os.path.join(output_folder, detail_filename)

    # Start building the HTML content
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Details - {exp_name}</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 20px;
            }}
            h1 {{
                text-align: center;
            }}
            .section {{
                margin-bottom: 30px;
            }}
            .section h2 {{
                border-bottom: 1px solid #ddd;
                padding-bottom: 5px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
            }}
            th, td {{
                border: 1px solid #dddddd;
                text-align: left;
                vertical-align: top;
                padding: 8px;
                word-wrap: break-word;
            }}
            th {{
                background-color: #f2f2f2;
            }}
            img {{
                max-width: 600px;
                height: auto;
                margin: 10px 0;
                border: 1px solid #ccc;
            }}
            .back-link {{
                margin-top: 20px;
            }}
        </style>
    </head>
    <body>
        <h1>Experiment Details - {exp_name}</h1>
        <div class="section">
            <h2>Configuration</h2>
            <table>
                <tbody>
    '''.format(exp_name=escape(exp_name))

    # Add configuration details
    config_keys = [key for key in exp_data.keys() if not key.startswith('Final ') and key not in ['Experiment Name', 'Experiment Folder', 'Images']]
    config_keys.sort()
    for key in config_keys:
        value = exp_data[key]
        if isinstance(value, (float, int)):
            value = f'{value:.4f}'
        else:
            value = str(value)
        html_content += f'''
                    <tr>
                        <th>{escape(key)}</th>
                        <td>{escape(value)}</td>
                    </tr>
        '''

    html_content += '''
                </tbody>
            </table>
        </div>
        <div class="section">
            <h2>Final Metrics</h2>
            <table>
                <tbody>
    '''

    # Add final metrics
    metric_keys = [key for key in exp_data.keys() if key.startswith('Final ')]
    metric_keys.sort()
    for key in metric_keys:
        value = exp_data[key]
        if isinstance(value, (float, int)):
            value = f'{value:.4f}'
        else:
            value = str(value)
        html_content += f'''
                    <tr>
                        <th>{escape(key)}</th>
                        <td>{escape(value)}</td>
                    </tr>
        '''

    html_content += '''
                </tbody>
            </table>
        </div>
        <div class="section">
            <h2>Images</h2>
    '''

    # Add images
    for img_path in exp_data.get('Images', []):
        # Use relative paths
        relative_img_path = os.path.relpath(img_path, start=os.path.dirname(detail_filepath))
        html_content += f'''
            <img src="{escape(relative_img_path)}" alt="{escape(os.path.basename(img_path))}">
        '''

    html_content += '''
        </div>
        <div class="back-link">
            <a href="experiments_summary.html">&larr; Back to Experiment Summary</a>
        </div>
    </body>
    </html>
    '''

    # Write the HTML content to the file
    with open(detail_filepath, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Experiment detail page saved to {detail_filepath}')

    return detail_filename

def generate_html_table(experiments_data, output_file, important_columns):
    """
    Generates an HTML file containing a table of experiment data with important columns and links to detail pages.
    """
    # Start building the HTML content
    html_content = '''
    <!DOCTYPE html>
    <html>
    <head>
        <title>Experiment Summary</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
            }
            .table-wrapper {
                overflow-x: auto;
                max-width: 100%;
            }
            table {
                border-collapse: collapse;
                width: 100%;
                min-width: 800px;
            }
            th, td {
                border: 1px solid #dddddd;
                text-align: left;
                vertical-align: top;
                padding: 8px;
                word-wrap: break-word;
            }
            th {
                background-color: #f2f2f2;
                position: sticky;
                top: 0;
                z-index: 2;
            }
            tr:nth-child(even) {
                background-color: #fafafa;
            }
            tr:hover {
                background-color: #f1f1f1;
            }
            .experiment-name {
                font-weight: bold;
            }
            /* Add sorting arrows */
            th.sortable:hover {
                cursor: pointer;
            }
            th.sortable::after {
                content: '\\25B4\\25BE'; /* Up and down arrows */
                font-size: 0.6em;
                padding-left: 5px;
                color: #aaa;
            }
        </style>
        <script>
            // Function to sort table columns
            function sortTable(n) {
                var table, rows, switching, i, x, y, shouldSwitch, dir, switchcount = 0;
                table = document.getElementById("experimentTable");
                switching = true;
                dir = "asc";
                while (switching) {
                    switching = false;
                    rows = table.rows;
                    for (i = 1; i < (rows.length - 1); i++) {
                        shouldSwitch = false;
                        x = rows[i].getElementsByTagName("TD")[n];
                        y = rows[i + 1].getElementsByTagName("TD")[n];
                        if (dir == "asc") {
                            if (x.textContent.toLowerCase() > y.textContent.toLowerCase()) {
                                shouldSwitch = true;
                                break;
                            }
                        } else if (dir == "desc") {
                            if (x.textContent.toLowerCase() < y.textContent.toLowerCase()) {
                                shouldSwitch = true;
                                break;
                            }
                        }
                    }
                    if (shouldSwitch) {
                        rows[i].parentNode.insertBefore(rows[i + 1], rows[i]);
                        switching = true;
                        switchcount++;
                    } else {
                        if (switchcount == 0 && dir == "asc") {
                            dir = "desc";
                            switching = true;
                        }
                    }
                }
            }
        </script>
    </head>
    <body>
        <h1>Experiment Summary</h1>
        <div class="table-wrapper">
            <table id="experimentTable">
                <thead>
                    <tr>
    '''
    # Add table headers
    for idx, key in enumerate(important_columns):
        html_content += f'<th class="sortable" onclick="sortTable({idx})">{escape(key)}</th>\n'
    html_content += '<th>Details</th>\n'  # Column for details button
    html_content += '''
                    </tr>
                </thead>
                <tbody>
    '''
    # Add table rows
    for exp_data in experiments_data:
        html_content += '<tr>\n'
        for key in important_columns:
            value = exp_data.get(key, '')
            if isinstance(value, (float, int)):
                value = f'{value:.4f}'
            else:
                value = str(value)
            # Highlight the experiment name
            if key == 'Experiment Name':
                value = f'<span class="experiment-name">{escape(value)}</span>'
            else:
                value = escape(value)
            html_content += f'<td>{value}</td>\n'
        # Add Details button/link
        detail_filename = exp_data['Detail Page']
        html_content += f'<td><a href="{escape(detail_filename)}">View Details</a></td>\n'
        html_content += '</tr>\n'
    html_content += '''
                </tbody>
            </table>
        </div>
    </body>
    </html>
    '''
    # Write the HTML content to the file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    print(f'Experiment summary saved to {output_file}')

def main(args):
    output_folder = args.output_folder
    experiments_data = []

    # List all experiment folders
    experiment_folders = [os.path.join(output_folder, d) for d in os.listdir(output_folder) if os.path.isdir(os.path.join(output_folder, d))]

    for exp_folder in experiment_folders:
        exp_data = collect_experiment_data(exp_folder)
        if exp_data:
            experiments_data.append(exp_data)

    # Define the important columns to display in the main table
    important_columns = [
        'Experiment Name',
        'model_config.model_type',
        'model_config.args.embedding_strategy',
        'model_config.args.dim_input',
        'model_config.args.dim_hidden',
        'Final Validation Accuracy',
        'Final Validation Loss',
        'Final Testing Accuracy',
        'Final Testing Loss',
        # Add or remove columns as needed
    ]

    # Generate detail pages for each experiment
    for exp_data in experiments_data:
        detail_filename = generate_experiment_detail_page(exp_data, output_folder)
        exp_data['Detail Page'] = detail_filename  # Add the detail page filename to the experiment data

    # Specify the output HTML file
    output_file = os.path.join(output_folder, 'experiments_summary.html')

    # Generate the HTML table
    generate_html_table(experiments_data, output_file, important_columns)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Generate a summary of experiment results in HTML format.")
    parser.add_argument("-o", "--output_folder", type=str, help="Path to the folder containing experiment results.")
    args = parser.parse_args()
    main(args)
