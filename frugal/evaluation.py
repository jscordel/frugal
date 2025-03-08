import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import json
import os
import seaborn as sns
import pandas as pd


def evaluate(model_name, y_test, y_pred):
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

    # Generate performance metrics using the classification report.
    # Other possibility is to use specific sklearn metrics functions (precision_score, recall_score, f1_score, accuracy_score)
    report = (classification_report(y_test, y_pred, output_dict=True))
    # Extract specific metrics from the report
    accuracy = report['accuracy']
    macro_precision =  report['macro avg']['precision']
    macro_recall = report['macro avg']['recall']
    macro_f1 = report['macro avg']['f1-score']

    # Store selected data in a custom dictionary that includes efficiency and performance metrics
    meta_data = {
        "model_name": model_name,
        "accuracy": round(accuracy, 2),
        # "timestamp": timestamp,

        # "model_details": model_memory_need(model=model),
        # "pipeline_kwargs": pipeline_kwargs,

        # "sample_size": N,
        # "quote_len_truncated": False if (max(results_df['X_test'].apply(len)) <= threshold) == False else threshold,

        # "prompt_template": prompt_template,

        # "efficiency": tracker.get_metrics().to_dict(),
        # "performance": performance.to_dict(),
        # 'category_performance' : metrics_df.to_dict(),
        # "model_pipeline" : model
    }

    # Confusion matrix
    # cm = confusion_matrix(y_test, y_pred)
    # disp = ConfusionMatrixDisplay(confusion_matrix=cm) # ajouter si besoin le paramètre : display_labels=ordinal_values
    # disp.plot()
    # plt.show()


    # ---------------
    # Save evaluation data

    # Convert and save the metrics to a JSON
    with open(filename_path, "w") as f:
        json.dump(meta_data, f, indent=4, default=str)

    # Confirm evaluation and saving
    print("✅ Evaluation done")
    print(f"Classification report saved to {filename}.json'")


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
    df = pd.DataFrame.from_dict(concatenated_data)

    print(df)
    return df


def global_model_comparison():

    plt.figure(figsize=(5, 3))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=np.unique(y_test),
        yticklabels=np.unique(y_test)
        )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.show()
