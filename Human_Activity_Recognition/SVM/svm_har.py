import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

data_path = "../dataset/"

train_X = np.loadtxt(data_path + "train/X_train.txt")
train_y = np.loadtxt(data_path + "train/y_train.txt") - 1

test_X = np.loadtxt(data_path + "test/X_test.txt")
test_y = np.loadtxt(data_path + "test/y_test.txt") - 1

svm_clf = SVC()
cv_results = cross_validate(svm_clf, train_X, train_y, scoring="accuracy", cv=5, return_estimator=True, verbose=1)

best_i = np.argmax(cv_results["test_score"])
best_clf = cv_results["estimator"][best_i]

test_preds = best_clf.predict(test_X)

print(classification_report(test_y, test_preds))

fig2, ax = plt.subplots()
fig2.set_size_inches(5, 5)
ax.set_title("Validation Accuracy", fontsize = 20)

ax.boxplot(np.multiply(cv_results["test_score"], 100))
ax.set_ylabel("Validation Accuracy (%)", fontsize = 18)
ax.set_xticklabels(["CNN"], fontsize = 18)

for label in ax.get_yticklabels():
    label.set_fontsize(14)

plt.tight_layout()
plt.show()