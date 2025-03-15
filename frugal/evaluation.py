import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    classification_report
    )
import json
import os
import seaborn as sns
import pandas as pd
import datetime
import numpy as np
import ast

# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# def confusion_matrix_plot(y_test, y_pred):
#     cm = confusion_matrix(y_test, y_pred)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm) # ajouter si besoin le paramètre : display_labels=ordinal_values
#     disp.plot()
#     plt.show()

def evaluate(model_name, y_test, y_pred, model_emissions):
    # ----------------
    # Prepare evaluation framework

    # Ensure evaluation directory exists otherwise it will be created
    eval_dir = "apps/evaluations"
    os.makedirs(eval_dir, exist_ok=True)

    # Define file name and path
    filename = model_name
    filename_path = os.path.join(eval_dir, filename + ".json")


    # ---------------
    # Make evaluation data

    if not (len(y_test) == len(y_pred)):
        raise ValueError(f"Length mismatch: y_test({len(y_test)}), y_pred({len(y_pred)})")

    # Compute ALL metrics
    report = (classification_report(y_test, y_pred, output_dict=True))
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

    # Extract MACRO metrics from the report
    accuracy = report['accuracy']
    macro_precision =  report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']
    # Store MACRO metrics
    macro_metrics = {
        "Model name": model_name,
        "Timestamp": timestamp,
        "Model Emissions": model_emissions, # in gCO2eq
        "Accuracy": accuracy,
        "Macro Precision": macro_precision,
        "Macro Recall": macro_recall,
        "Macro F1 Score": macro_f1
        }

    # Get category labels (sorted for consistency)
    category_names = sorted(pd.Series(y_test).unique())
    # Compute per-CATEGORY metrics --- maybe better to reuse 'report' to ensure harmonised results, but this is a quick fix
    precision = precision_score(y_test, y_pred, average=None, labels=category_names, zero_division=0)
    recall = recall_score(y_test, y_pred, average=None, labels=category_names, zero_division=0)
    f1 = f1_score(y_test, y_pred, average=None, labels=category_names, zero_division=0)
    # Store per-CATEGORY metrics
    category_df = pd.DataFrame({
        "Cat Names": category_names,
        "Cat Precision": precision,
        "Cat Recall": recall,
        "Cat F1 Score": f1
    })
    # Store category performance metrics
    category_data = {
        "Category performance" : category_df.to_dict()
    }

    # Compile macro metrics & category metrics
    macro_metrics.update(category_data)


    # ---------------
    # Save evaluation data

    # Convert and save the metrics to a JSON
    with open(filename_path, "w") as f:
        json.dump(macro_metrics, f, indent=4, default=str)

    # Confirm evaluation and saving
    print("✅ Evaluation done")
    print(f"Report saved to {filename}.json'")



def concatenate_evaluations():
    concatenated_data = {}

    for filename in os.listdir("apps/evaluations"):
        if filename.endswith(".json"):
            file_path = os.path.join("apps/evaluations", filename)

            # Read the JSON file
            with open(file_path, 'r') as f:
                data = json.load(f)

                # Concatenate the data
                for key, value in data.items():
                    if key not in concatenated_data:
                        concatenated_data[key] = []
                    concatenated_data[key].append(value)

    # Convert the concatenated dictionary to a DataFrame
    concatenated_data_df = pd.DataFrame.from_dict(concatenated_data)

    print(concatenated_data_df)
    concatenated_data_df.to_csv('concatenated_data.csv')

    return concatenated_data_df


def global_model_comparison(df):
    # Set baseline
    baseline = 0.125 # corresponds to 1/8

    plt.figure(figsize=(8, 5))
    plt.plot(df['Model name'], df['Macro F1 Score'], marker='o', label='F1 Score',linestyle='')
    plt.plot(df['Model name'], df['Macro Precision'], marker='s', label='Precision', linestyle='')
    plt.plot(df['Model name'], df['Macro Recall'], marker='^', label='Recall', linestyle='')

    # Add Baseline values
    plt.axhline(y=baseline, color='r', linestyle='--', label=f'Baseline ({baseline})')
    plt.xlabel("Models")
    plt.ylabel("Score")
    plt.title("Performance Metrics per Model")
    plt.legend()
    plt.grid(True)

    print()
    plt.show()


def global_class_comparison(df):
    # MAKE heatmaps for each performance metric, for each class and models
    precision_data_corrected = {}
    recalll_data_corrected = {}
    f1_data_corrected = {}

    for index, row in df.iterrows():
        model_name = row['Model name']
        category_precision = row['Category performance']['Cat Precision']
        category_recall = row['Category performance']['Cat Recall']
        category_f1 = row['Category performance']['Cat F1 Score']

        # Extract precision values for categories 0 to 7
        precision_values = {}
        recalll_values = {}
        f1_values = {}

        for cat in range(8):
            cat_str = str(cat)
            if cat_str in category_precision.keys():
                precision_values[cat] = category_precision[cat_str]
            if cat_str in category_recall.keys():
                recalll_values[cat] = category_recall[cat_str]
            if cat_str in category_f1.keys():
                f1_values[cat] = category_f1[cat_str]
            else:
                precision_values[cat] = None  # In case the category is missing

        # Store the values in a dictionary using the model name as the key
        precision_data_corrected[model_name] = precision_values
        recalll_data_corrected[model_name] = recalll_values
        f1_data_corrected[model_name] = f1_values

    # Convert the corrected precision data to a DataFrame for plotting
    precision_df_corrected = pd.DataFrame(precision_data_corrected).T
    recalll_df_corrected = pd.DataFrame(recalll_data_corrected).T
    f1_df_corrected = pd.DataFrame(f1_data_corrected).T


    # Plot the heatmap
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    sns.heatmap(precision_df_corrected, ax=axes[0], cmap='coolwarm', annot=True)
    axes[0].set_title('Precision 🎯')

    sns.heatmap(recalll_df_corrected, ax=axes[1], cmap='coolwarm', annot=True)
    axes[1].set_title('Recall ✅')

    sns.heatmap(f1_df_corrected, ax=axes[2], cmap='coolwarm', annot=True)
    axes[2].set_title('F1 🏎️')

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    global_data_df = concatenate_evaluations()
    # Global model and class performance comparison
    global_model_comparison(global_data_df)
    global_class_comparison(global_data_df)
