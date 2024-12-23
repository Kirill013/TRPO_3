import numpy as np
import numpy.linalg as ln
import scipy as sp
import scipy.optimize
import matplotlib.pyplot as plt
import logging

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Целевая функция
def objective_function(point):
    x1, x2 = point
    return (1 - x1)**2 + 100 * (x2 - x1**2)**2

# Градиент целевой функции
def objective_gradient(point):
    x1, x2 = point
    grad_x1 = -2 * (1 - x1) - 400 * x1 * (x2 - x1**2)
    grad_x2 = 200 * (x2 - x1**2)
    return np.array([grad_x1, grad_x2])

# Метод BFGS
def bfgs_method(f, grad_f, initial_point, max_iterations=200, tolerance=1e-3):
    """
    :param f: целевая функция
    :param grad_f: градиент целевой функции
    :param initial_point: начальная точка
    :param max_iterations: int, максимальное количество итераций
    :param tolerance: float, порог точности
    :return: минимум функции, число итераций, история точек, история значений
    """
    iteration = 0
    current_gradient = grad_f(initial_point)
    dimension = len(initial_point)
    identity_matrix = np.eye(dimension)
    hessian_approx = identity_matrix
    current_point = initial_point

    # Для хранения данных логирования
    points_history = [initial_point]
    values_history = [f(initial_point)]

    logging.info(f"Начальная точка: {initial_point}, начальное значение функции: {f(initial_point):.4f}")

    while ln.norm(current_gradient) > tolerance and iteration < max_iterations:
        # Направление поиска
        search_direction = -np.dot(hessian_approx, current_gradient)

        # Линия поиска
        line_search = sp.optimize.line_search(f, grad_f, current_point, search_direction)
        step_size = line_search[0]

        if step_size is None:
            raise ValueError(f"Линия поиска не смогла найти шаг на итерации {iteration}.")

        # Обновление точки и градиента
        next_point = current_point + step_size * search_direction
        point_difference = next_point - current_point
        next_gradient = grad_f(next_point)
        gradient_difference = next_gradient - current_gradient

        current_gradient = next_gradient
        current_point = next_point

        # Сохранение истории
        points_history.append(current_point)
        values_history.append(f(current_point))

        # Обновление аппроксимации Гессиана
        rho = 1.0 / (np.dot(gradient_difference, point_difference))
        identity_update_1 = np.eye(dimension) - rho * np.outer(point_difference, gradient_difference)
        identity_update_2 = np.eye(dimension) - rho * np.outer(gradient_difference, point_difference)
        hessian_approx = np.dot(identity_update_1, np.dot(hessian_approx, identity_update_2)) + rho * np.outer(point_difference, point_difference)

        iteration += 1
        logging.info(f"Итерация {iteration}: точка = {current_point}, f(x) = {f(current_point):.4f}, ||grad|| = {ln.norm(current_gradient):.4f}")

    if ln.norm(current_gradient) > tolerance:
        raise RuntimeError(
            f"Метод BFGS не достиг сходимости за {max_iterations} итераций. "
            f"Текущая точка: {current_point}, значение функции: {f(current_point):.4f}, ||grad|| = {ln.norm(current_gradient):.4f}"
        )

    return current_point, iteration, points_history, values_history

# Функция визуализации
def plot_optimization(points_history, values_history, f):
    """
    :param points_history: список точек, посещённых методом
    :param values_history: значения функции в этих точках
    :param f: целевая функция
    """
    points_history = np.array(points_history)
    X, Y = points_history[:, 0], points_history[:, 1]

    plt.figure(figsize=(10, 6))
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 3, 400)
    X_grid, Y_grid = np.meshgrid(x, y)
    Z = f([X_grid, Y_grid])

    plt.contour(X_grid, Y_grid, Z, levels=30, cmap='viridis')
    plt.plot(X, Y, 'ro-', label='Траектория')
    plt.scatter(X[-1], Y[-1], color='blue', label='Итоговая точка', zorder=5)
    plt.title('Траектория оптимизации методом BFGS')
    plt.xlabel('$x_1$')
    plt.ylabel('$x_2$')
    plt.legend()
    plt.grid()
    plt.colorbar(label='$f(x)$')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(values_history, 'bo-', label='Значение функции')
    plt.title('Значение функции по итерациям')
    plt.xlabel('Итерация')
    plt.ylabel('$f(x)$')
    plt.grid()
    plt.legend()
    plt.show()

# Пример вызова метода
initial_point = np.array([-1.2, 1.0])  # Начальная точка

try:
    # Применение метода BFGS
    result, iterations, points_history, values_history = bfgs_method(objective_function, objective_gradient, initial_point)

    print('Результаты работы метода BFGS:')
    print(f'Итоговая точка (минимум функции): {result}')
    print(f'Количество итераций: {iterations}')

    plot_optimization(points_history, values_history, objective_function)

except RuntimeError as e:
    logging.error(f"Ошибка: {e}")
