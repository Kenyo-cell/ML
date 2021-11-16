import numpy as np
import struct
from functools import reduce
from scipy.optimize import minimize
from matplotlib import pyplot as plt

# Вам предлагается обучить трехслойную нейронную сеть прямого распространения с сигмоидальными функциями активации на
# наборе данных MNIST - множестве изображений рукописных цифр и соответствующих метках 0-9.

# Ваша задача заполнить код в следующих функциях:
# logistic - вычисляет логистическую функцию от аргумента,
# compute_hypothesis - вычисляет выходы сети для заданного набора данных,
# compute_cost - вычисляет функцию стоимости,
# compute_cost_grad - вычисляет градиент функции стоимости с помощью алгоритма обратного распространения ошибки.

# По ходу работы смотрите в консоль. Там будут выводиться результаты написанных вами функций и ожидаемые результаты.
# Так вы сможете проверить себя.

# Ниже заданы "константы", с которыми можно "поиграть" - поварьировать их и посмотреть, что будет.
HIDDEN_LAYER_SIZE = 64  # Количество нейронов в скрытом слое.

# Всего у вас есть 60 000 изображений, поэтому сумма следующих двух констант не должна быть больше 60 000.
SAMPLES_TO_TRAIN = 1000  # Количество примеров в обучающей выборка, на ней будет производить подбор параметров модели.
SAMPLES_TO_TEST = 100  # Количество примеров в тестовой выборке, на ней будет оцениваться качество модели.


def one_hot_encode(y):

    # Функция кодирует вектор чисел с метками классов в матрицу, каждая строка которой является унитарным кодом
    # соответствующей метки.

    return np.array(np.arange(0, 10, 1) == y, dtype=np.int32)


def get_random_matrix(l_in, l_out):

    # Функция для генерации матрицы случайных параметров.

    return np.random.uniform(-0.5, 0.5, (l_out, l_in + 1))


def compute_cost_grad_approx(X, y, theta, lamb):

    # Фукнция для численного расчета градиента фукници стоимости. Такой метод полезно использовать при проверке
    # правильности аналитического расчета.

    eps = 1e-5
    grad = np.zeros(theta.size)
    perturb = np.zeros(theta.size)
    for i in range(theta.shape[0]):
        if i % 1000 == 0:
            print(f'{round(i / theta.shape[0] * 100, 2)}%')
        perturb[i] = eps
        grad[i] = (compute_cost(X, y, theta + perturb, lamb) - compute_cost(X, y, theta - perturb, lamb)) / (2 * eps)
        perturb[i] = 0
    return grad


def logistic(z):

    # Функция принимает аргумент z - скаляр, вектор или матрицу в виде объекта numpy.array()
    # Должна возвращать скяляр, вектор или матрицу (в зависимости от размерности z)
    # результатов вычисления логистической функции от элементов z

    return 1 / (1 + np.exp(-z))


def logistic_grad(z):

    # Функция принимает аргумент z - скаляр, вектор или матрицу в виде объекта numpy.array()
    # Должна возвращать скяляр, вектор или матрицу (в зависимости от размерности z)
    # результатов вычисления производной логистической функции от элементов z

    e_z = np.exp(-z)

    return e_z / (1 + e_z) ** 2


def add_fictive(X):
    return np.column_stack((np.ones(X.shape[0]), X))


def compute_hypothesis(X, theta):

    # Функция принимает матрицу данных X без фиктивного признака и вектор параметров theta.
    # Вектор theta является "разворотом" мартиц Theta1 и Theta2 в плоский вектор.
    # Функция должна возвращать матрицу выходов сети для каждого примера из обучающего набора,
    # а также матрицу входов нейронов скрытого слоя и матрицу выходов скрытого слоя (это понадобится для реализации
    # алгоритма обратного распространения ошибки).

    # Восстановление матриц Theta1 и Theta2 из ветора theta:
    Theta1 = theta[:HIDDEN_LAYER_SIZE * (X.shape[1] + 1)].reshape(HIDDEN_LAYER_SIZE, X.shape[1] + 1)
    Theta2 = theta[HIDDEN_LAYER_SIZE * (X.shape[1] + 1):].reshape(10, HIDDEN_LAYER_SIZE + 1)

    X_fictive = add_fictive(X)

    Z2 = X_fictive @ Theta1.T # Будущая матрица входов скрытого слоя, нужно заполнить

    A2 = logistic(Z2) # Будущая матрица выходов скрытого слоя, нужно заполнить

    A2_fictive = add_fictive(A2)

    output_in = A2_fictive @ Theta2.T

    H = logistic(output_in) # Будущая матрица выходов, нужно заполнить 

    return H, Z2, A2


def compute_cost(X, y, theta, lamb):

    # Функция принимает матрицу данных X без фиктивного признака, вектор верных классов y,
    # вектор параметров theta и параметр регуляризации lamb.
    # Вектор theta является "разворотом" мартиц Theta1 и Theta2 в плоский вектор.
    # Должна вернуть значение функции стоимости в точке theta.

    m, n = X.shape  # m - количество примеров в выборке, n - количество признаков у каждого примера
    cost = 0  # значение функции стоимости при заданных параметрах, его нужно посчитать

    # Восстановление матриц Theta1 и Theta2 из ветора theta:
    Theta1 = theta[:HIDDEN_LAYER_SIZE * (X.shape[1] + 1)].reshape(HIDDEN_LAYER_SIZE, X.shape[1] + 1)
    Theta2 = theta[HIDDEN_LAYER_SIZE * (X.shape[1] + 1):].reshape(10, HIDDEN_LAYER_SIZE + 1)

    # ВАШ КОД ЗДЕСЬ
    H, Z, A = compute_hypothesis(X, theta)

    for i in range(m):
        y_i = one_hot_encode(y[i])
        cost += sum(y_i * np.log(H[i]) + (1 - y_i) * np.log(1 - H[i]))

    # ==============

    return -cost / m + lamb / (2 * m) * np.sum(theta ** 2)


def compute_cost_grad(X, y, theta, lamb):

    # Функция принимает матрицу данных X без фиктивного признака, вектор верных классов y,
    # вектор параметров theta и параметр регуляризации lamb.
    # Вектор theta является "разворотом" мартиц Theta1 и Theta2 в плоский вектор.
    # Функция должна реализовать алгоритм обратного распространения ошибки и вернуть матрицы градиентов весов связей.

    m, n = X.shape  # m - количество примеров в выборке, n - количество признаков у каждого примера

    # Восстановление матриц Theta1 и Theta2 из ветора theta:
    Theta1 = theta[:HIDDEN_LAYER_SIZE * (X.shape[1] + 1)].reshape(HIDDEN_LAYER_SIZE, X.shape[1] + 1)
    Theta2 = theta[HIDDEN_LAYER_SIZE * (X.shape[1] + 1):].reshape(10, HIDDEN_LAYER_SIZE + 1)

    Theta1_grad = None  # Будущая матрица градиентов весов между входным и скрытым слоем, ее нужно заполнить
    Theta2_grad = None  # Будущая матрица градиентов весов между скрытым и выходным слоем, ее нужно заполнить

    # ВАШ КОД ЗДЕСЬ
    H, Z2, A2 = compute_hypothesis(X, theta)

    X = add_fictive(X)
    A2 = add_fictive(A2)
    Y = np.array([one_hot_encode(i) for i in y])

    d3 = H - Y

    # Native
    # d2 = np.array([(Theta2.T @ d3[i]) * logistic_grad(add_fictive(Z2)[i]) for i in range(d3.shape[0])])[ : , 1:]

    # Tested
    d2 = ((Theta2.T @ d3.T) * logistic_grad(add_fictive(Z2).T)).T[ : , 1:]

    
    d1 = (X.T @ d2).T
    d1 = np.column_stack((d1[ :, 0 ], d1[ :, 1: ] + lamb * Theta1[ :, 1: ]))

    d2 = (A2.T @ d3).T
    d2 = np.column_stack((d2[ :, 0 ], d2[ :, 1: ] + lamb * Theta2[ :, 1: ]))

    Theta1_grad = d1 / m
    Theta2_grad = d2 / m
    # ==============

    return np.concatenate((Theta1_grad, Theta2_grad), axis=None)


def load_data_idx(file_path):
    with open(file_path, 'rb') as input_file:
        magic = input_file.read(4)
        dims = int(magic[3])
        sizes = [struct.unpack('>L', input_file.read(4))[0] for _ in range(dims)]
        size = reduce(lambda x, y: x * y, sizes)
        data = np.array(list(input_file.read(size)), dtype=float)
        data = data.reshape(sizes)
        return data


# Загрузка данных
images = load_data_idx('images.idx')
features = images.reshape((images.shape[0], images.shape[1] * images.shape[2])) / 128 - 1.0

labels = load_data_idx('labels.idx')

X = features[:SAMPLES_TO_TRAIN]
y = labels[:SAMPLES_TO_TRAIN]

X_test = features[SAMPLES_TO_TRAIN: SAMPLES_TO_TRAIN + SAMPLES_TO_TEST]
y_test = labels[SAMPLES_TO_TRAIN: SAMPLES_TO_TRAIN + SAMPLES_TO_TEST]

print(f'logistic(0) = {logistic(np.array(0))} (должно быть 0.5)\n'
      f'logistic(-10) = {logistic(np.array(-10))} (должно быть ~0)\n'
      f'logistic(10) = {logistic(np.array(10))} (должно быть ~1)')
print()
print(f'logistic_grad(0) = {logistic_grad(np.array(0))} (должно быть ~0.25)\n'
      f'logistic_grad(-10) = {logistic_grad(np.array(-10))} (должно быть ~0)\n'
      f'logistic_grad(10) = {logistic_grad(np.array(10))} (должно быть ~0)')

im_size = int(features.shape[1] ** 0.5)
hm = np.zeros((im_size * 10, im_size * 10))
for i in range(100):
    im_x = i % 10 * im_size
    im_y = i // 10 * im_size
    for j in range(features.shape[1]):
        px_x = im_x + j % im_size
        px_y = im_y + j // im_size
        hm[10 * im_size - 1 - px_x, px_y] = images[i, j % im_size, j // im_size]
plt.pcolor(hm, cmap='Greys')
plt.show()

lamb = 0.15  # можно поварьировать параметр регуляризации и посмотреть, что будет

init_Theta1 = np.random.uniform(-0.12, 0.12, (HIDDEN_LAYER_SIZE, X.shape[1] + 1))
init_Theta2 = np.random.uniform(-0.12, 0.12, (10, HIDDEN_LAYER_SIZE + 1))

init_theta = np.concatenate((init_Theta1, init_Theta2), axis=None)

back_prop_grad = compute_cost_grad(X[:10], y[:10], init_theta, 0)

print('Запуск численного расчета градиента (может занять какое-то время)')
num_grad = compute_cost_grad_approx(X[:10], y[:10], init_theta, 0)

print('Градиент, посчитанный аналитически: ')
print(back_prop_grad)

print('Градиент, посчитанный численно (должен примерно совпадать с предыдущим): ')
print(num_grad)

print()
print('Относительное отклоение градиентов (должно быть < 1e-7): ')
print((np.linalg.norm(back_prop_grad - num_grad) / np.linalg.norm(back_prop_grad + num_grad)))

print()
print('Запуск оптимизации параметров сети (может занять какое-то время)')
opt_theta_obj = minimize(lambda th: compute_cost(X, y, th, lamb), init_theta,
                         method='CG',
                         jac=lambda th: compute_cost_grad(X, y, th, lamb),
                         options={'gtol': 1e-3, 'maxiter': 3000, 'disp': False})

print('Минимизация функции стоимости ' + ('прошла успешно.' if opt_theta_obj.success else 'не удалась.'))

opt_theta = opt_theta_obj.x
print('Функция стоимости при оптимальных параметрах (должна быть < 0.5): ')
print(compute_cost(X, y, opt_theta, lamb))

hypos_train = compute_hypothesis(X, opt_theta)[0]

true_preds_train_num = 0
for k in range(hypos_train.shape[0]):
    max_el = np.max(hypos_train[k, :])
    true_preds_train_num += list(np.array(hypos_train[k, :] == max_el, dtype=np.int32)) == list(one_hot_encode(y[k]))
print('Доля верно распознаных цифр в обучающем наборе:', true_preds_train_num / hypos_train.shape[0] * 100, '%')

hypos_test = compute_hypothesis(X_test, opt_theta)[0]

true_preds_test_num = 0
for k in range(hypos_test.shape[0]):
    max_el = np.max(hypos_test[k, :])
    true_preds_test_num += list(np.array(hypos_test[k, :] == max_el, dtype=np.int32)) == list(one_hot_encode(y_test[k]))
print('Доля верно распознаных цифр в тестовом наборе:', true_preds_test_num / hypos_test.shape[0] * 100, '%')

if HIDDEN_LAYER_SIZE ** 0.5 % 1.0 == 0:
    Theta1 = opt_theta[:HIDDEN_LAYER_SIZE * (X.shape[1] + 1)].reshape(HIDDEN_LAYER_SIZE, X.shape[1] + 1)
    Theta1 = Theta1[:, 1:]
    Theta1 = (Theta1 - Theta1.mean(axis=0)) / (Theta1.std(axis=0))
    Theta1 = Theta1.reshape(HIDDEN_LAYER_SIZE, 28, 28)
    im_size = int(features.shape[1] ** 0.5)
    size_imgs = int(HIDDEN_LAYER_SIZE ** 0.5)
    hm = np.zeros((im_size * size_imgs, im_size * size_imgs))
    for i in range(HIDDEN_LAYER_SIZE):
        im_x = i % size_imgs * im_size
        im_y = i // size_imgs * im_size
        for j in range(features.shape[1]):
            px_x = im_x + j % im_size
            px_y = im_y + j // im_size
            hm[size_imgs * im_size - 1 - px_x, px_y] = Theta1[i, j % im_size, j // im_size]
    plt.pcolor(hm, cmap='Greys')
    plt.show()
