import numpy as np
from collections import Counter


def find_best_split(feature_vector, target_vector):
    """
    Под критерием Джини здесь подразумевается следующая функция:
    $$Q(R) = -\frac {|R_l|}{|R|}H(R_l) -\frac {|R_r|}{|R|}H(R_r)$$,
    $R$ — множество объектов, $R_l$ и $R_r$ — объекты, попавшие в левое и правое поддерево,
     $H(R) = 1-p_1^2-p_0^2$, $p_1$, $p_0$ — доля объектов класса 1 и 0 соответственно.

    Указания:
    * Пороги, приводящие к попаданию в одно из поддеревьев пустого множества объектов, не рассматриваются.
    * В качестве порогов, нужно брать среднее двух сосдених (при сортировке) значений признака
    * Поведение функции в случае константного признака может быть любым.
    * При одинаковых приростах Джини нужно выбирать минимальный сплит.
    * За наличие в функции циклов балл будет снижен. Векторизуйте! :)

    :param feature_vector: вещественнозначный вектор значений признака
    :param target_vector: вектор классов объектов,  len(feature_vector) == len(target_vector)

    :return thresholds: отсортированный по возрастанию вектор со всеми возможными порогами, по которым объекты можно
     разделить на две различные подвыборки, или поддерева
    :return ginis: вектор со значениями критерия Джини для каждого из порогов в thresholds len(ginis) == len(thresholds)
    :return threshold_best: оптимальный порог (число)
    :return gini_best: оптимальное значение критерия Джини (число)
    """
    feature_vector = np.array(feature_vector, dtype=np.float64)
    target_vector = np.array(target_vector, dtype=int)

    sort_indexes = np.argsort(feature_vector)

    feature_vector = feature_vector[sort_indexes]
    target_vector = target_vector[sort_indexes]

    feature_vector_unique = np.unique(feature_vector)

    rolled_one_left = feature_vector_unique[1:]
    feature_vector_wo_last = feature_vector_unique[:-1]

    thresholds = (rolled_one_left + feature_vector_wo_last) / 2

    _, unique_indexes_reversed = np.unique(
        feature_vector[::-1], return_index=True)

    unique_indexes = len(feature_vector) - unique_indexes_reversed

    length_left = unique_indexes[:-1]

    p1_for_lefts = np.cumsum(target_vector)[length_left - 1]

    p1 = p1_for_lefts / length_left

    HR_l = 1 - p1 ** 2 - (1 - p1) ** 2

    length_rigth = len(feature_vector) - length_left

    p1_for_rights = np.sum(target_vector) - p1_for_lefts

    p12 = p1_for_rights / length_rigth

    HR_r = 1 - p12 ** 2 - (1 - p12) ** 2

    Qs = -length_left / len(target_vector) * HR_l - length_rigth / len(target_vector) * HR_r

    return thresholds, Qs, thresholds[np.argmax(Qs)], np.max(Qs)


class DecisionTree:
    def __init__(
            self, feature_types, max_depth=np.inf, min_samples_split=None,
            min_samples_leaf=None):
        if np.any(list(map(lambda x: x != "real" and x != "categorical", feature_types))):
            raise ValueError("There is unknown feature type")

        self._tree = {}
        self._feature_types = feature_types
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._min_samples_leaf = min_samples_leaf

    def _fit_node(self, sub_X, sub_y, node, depth=0):
        if np.all(sub_y == sub_y[0]):
            node["type"] = "terminal"
            node["class"] = sub_y[0]
            return

        feature_best, threshold_best, gini_best, split = None, None, None, None
        continue_split_condition = depth < self._max_depth
        if self._min_samples_split is not None:
            continue_split_condition = depth < self._max_depth and sub_X.shape[0] >= self._min_samples_split

        if continue_split_condition:
            for feature in range(sub_X.shape[1]):
                feature_type = self._feature_types[feature]
                categories_map = {}

                if feature_type == "real":
                    feature_vector = sub_X[:, feature]
                elif feature_type == "categorical":
                    counts = Counter(sub_X[:, feature])
                    clicks = Counter(sub_X[sub_y == 1, feature])
                    ratio = {}
                    for key, current_count in counts.items():
                        if key in clicks:
                            current_click = clicks[key]
                        else:
                            current_click = 0
                        ratio[key] = current_click / current_count
                    sorted_categories = list(
                        map(lambda x: x[0], sorted(ratio.items(), key=lambda x: x[1])))
                    categories_map = dict(
                        zip(sorted_categories, list(range(len(sorted_categories)))))

                    feature_vector = np.array(
                        list(map(lambda x: categories_map[x],
                                 sub_X[:, feature])))
                else:
                    raise ValueError

                if len(np.unique(feature_vector)) <= 1:
                    continue

                _, _, threshold, gini = find_best_split(feature_vector, sub_y)

                left_size = sub_X[feature_vector < threshold].shape[0]
                right_size = sub_X[feature_vector >= threshold].shape[0]
                subtrees_are_valid = True
                if self._min_samples_leaf is not None:
                    subtrees_are_valid = left_size >= self._min_samples_leaf and right_size >= self._min_samples_leaf

                if subtrees_are_valid and (gini_best is None or gini > gini_best):
                    feature_best = feature
                    gini_best = gini
                    split = feature_vector < threshold

                    if feature_type == "real":
                        threshold_best = threshold
                    elif feature_type == "categorical":
                        threshold_best = list(map(lambda x: x[0], filter(
                            lambda x: x[1] < threshold, categories_map.items())))
                    else:
                        raise ValueError

        if self._min_samples_split is not None:
            condition_terminal = feature_best is None or sub_X.shape[0] < self._min_samples_split
        else:
            condition_terminal = feature_best is None

        if condition_terminal:
            node["type"] = "terminal"
            node["class"] = Counter(sub_y).most_common(1)[0][0]
            return

        node["type"] = "nonterminal"

        node["feature_split"] = feature_best
        if self._feature_types[feature_best] == "real":
            node["threshold"] = threshold_best
        elif self._feature_types[feature_best] == "categorical":
            node["categories_split"] = threshold_best
        else:
            raise ValueError
        node["left_child"], node["right_child"] = {}, {}
        self._fit_node(sub_X[split], sub_y[split], node["left_child"], depth + 1)
        self._fit_node(sub_X[np.logical_not(split)], sub_y[np.logical_not(split)], node["right_child"], depth + 1)

    def _predict_node(self, x, node):
        if node['type'] == 'terminal':
            return node['class']

        feature = node['feature_split']

        if self._feature_types[feature] == 'real':
            if x[feature] < node['threshold']:
                return self._predict_node(x, node['left_child'])
            return self._predict_node(x, node['right_child'])

        if np.isin(x[feature], node['categories_split']):
            return self._predict_node(x, node['left_child'])
        return self._predict_node(x, node['right_child'])

    def fit(self, X, y):
        self._fit_node(X, y, self._tree)

    def predict(self, X):
        predicted = []
        for x in X:
            predicted.append(self._predict_node(x, self._tree))
        return np.array(predicted)

    def get_params(self, deep=False):
        params = {
            'feature_types': self._feature_types,
            'max_depth': self._max_depth,
            'min_samples_split': self._min_samples_split,
            'min_samples_leaf': self._min_samples_leaf
        }
        return params
