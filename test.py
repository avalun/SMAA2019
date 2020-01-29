from pprint import pprint

from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from movies import load

if __name__ == '__main__':
    grid: GridSearchCV = load('grid.p')
    train, test, train_labels, test_labels = load('dataset.p')
    y_pred = grid.predict(test)
    r2 = r2_score(test_labels, y_pred)
    pprint(grid.cv_results_)
    print('SVR - R2 metric:', r2)
