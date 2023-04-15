import pandas as pd
from sklearn.model_selection import train_test_split


class FinancialDataset:
    """Financial Simulated Dataset"""

    def __init__(self, data_dir, train_val_split, input_dim, linear):
        self.x = self._read_file(data_dir / f"c_{input_dim}.csv")
        if linear:
            self.y = self._read_file(data_dir / f"r1_{input_dim}.csv")
        else:
            self.y = self._read_file(data_dir / f"r2_{input_dim}.csv")
        x_train, x_val, y_train, y_val = train_test_split(
            self.x, self.y, test_size=train_val_split, shuffle=False
        )
        self.train_dataset = (x_train, y_train)
        self.val_dataset = (x_val, y_val)

    def _read_file(self, filepath):
        return pd.read_csv(filepath, header=None)

    def __len__(self):
        return len(self.x)

    def __call__(self):
        return self.train_dataset, self.val_dataset
