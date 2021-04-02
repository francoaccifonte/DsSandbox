from sklearn.datasets import load_iris
from pdb import set_trace as st
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mglearn


class flowers:
    display_plot = True
    store_results = False

    def __init__(self):
        self.iris_dataset = load_iris()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.iris_dataset["data"], self.iris_dataset["target"], random_state=0
        )

    def preview_data(self):
        dataset = pd.DataFrame(self.X_train, columns=self.iris_dataset.feature_names)
        pd.plotting.scatter_matrix(
            dataset,
            c=self.y_train,
            figsize=(15, 15),
            marker="o",
            hist_kwds={"bins": 20},
            s=60,
            alpha=0.8,
            cmap=mglearn.cm3,
        )
        if flowers.display_plot:
            plt.show()

    def train_model(self):
        knn = KNeighborsClassifier(n_neighbors=1)
        knn.fit(self.X_train, self.y_train)
        self.model = knn

    def predict(self, sw, sl, pw, pl):
        X_new = np.array([[sw, sl, pw, pl]])
        prediction = self.model.predict(X_new)
        print("Prediction: {}".format(prediction))
        print(
            "Predicted target name: {}".format(
                self.iris_dataset["target_names"][prediction]
            )
        )

    def score(self):
        pass  # TODO


if __name__ == "__main__":
    model = flowers()
    model.preview_data()
    model.train_model()
    model.predict(5, 2.9, 1, 0.2)
