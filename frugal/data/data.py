from datasets import load_dataset
from frugal.config.environment import DATA_DIR

def load_data():
    # Load train and test datasets from HF and make X and y as lists, for train & test
    train_dataset, test_dataset = datasets()
    X_train, X_test, y_train, y_test = make_data_lists(train_dataset, test_dataset)
    return X_train, X_test, y_train, y_test


def datasets():
    # Load the train and test datasets from HF
    train_dataset = load_dataset('QuotaClimat/frugalaichallenge-text-train', split = 'train', cache_dir=DATA_DIR)
    test_dataset = load_dataset('QuotaClimat/frugalaichallenge-text-train', split = 'test', cache_dir=DATA_DIR)

    print("✅ Data loaded from 🤗 ")
    return train_dataset, test_dataset


def make_data_lists(train_dataset, test_dataset):
    # Make X and y as lists, for train & test
    # only keeps the 'texts' and 'categories' of the dataset
    X_train = train_dataset['quote']
    X_test = test_dataset['quote']
    y_train = train_dataset['label']
    y_test = test_dataset['label']

    print("✅ X, y test & train lists made")
    return X_train, X_test, y_train, y_test
