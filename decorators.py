from functools import wraps, lru_cache, singledispatch
from deprecated import deprecated
import warnings
from memory_profiler import profile as memit  # готовый декоратор для замера использования памяти
from dataclasses import dataclass
import time
from colorama import Fore

# https://nuancesprog.ru/p/17759/

"""
Декораторы Python - это эффективные инструменты, 
помогающие создавать чистый, многократно используемый и легко сопровождаемый код.

Можно считать декораторы функциями, 
которые принимают другие функции в качестве входных данных 
и расширяют их функциональность без изменения основного назначения.

Чтобы написать этот декоратор, нужно сначала подобрать ему подходящее имя.

Затем передать ему другую функцию в качестве входной и вернуть ее в качестве выходной. 
Выходная функция обычно является расширенной версией входной. 

Поскольку нам неизвестно, какие аргументы использует входная функция, 
можем передать их из функции-обертки с помощью выражений *args и **kwargs. 

Теперь можно применить декоторатор к любой другой функции,
прописав перед ее объявлением @ и имя декоратора.
"""

"""
Встроенный декоратор @wraps из functools обновляет функцию-обертку, 
чтобы она выглядела как оригинальная функция и наследовала ее имя и свойства, 
а не документацию и имя декоратора.
"""


def logger(function):
    """Регистрация начала и окончания выполнения ф-и"""

    @wraps(function)
    def wrapper(*args, **kwargs):
        """wrapper documentation"""
        print(f"----- {function.__name__}: start -----")
        result = function(*args, **kwargs)
        print(f"----- {function.__name__}: end -----")
        return result

    return wrapper


def ignore_warnings(function):
    """Игнорирование предупреждений"""

    @wraps(function)
    def wrapper(*args, **kwargs):
        warnings.filterwarnings('ignore')
        result = function(*args, **kwargs)
        warnings.filterwarnings('default')
        return result

    return wrapper


def timeit(rnd=4):
    """Измерение времени выполнения ф-и"""

    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            tic = time.perf_counter()
            value = function(*args, **kwargs)
            tac = time.perf_counter()
            elapsed_time = tac - tic
            print(Fore.YELLOW + f'"{function.__name__}" elapsed {round(elapsed_time, rnd)} seconds' + Fore.RESET)
            return value

        return wrapper

    return decorate


def cache(function):
    """Кэширование ф-и"""

    @wraps(function)
    def wrapper(*args, **kwargs):
        cache_key = args + tuple(kwargs.items())
        if cache_key in wrapper.cache:
            output = wrapper.cache[cache_key]
        else:
            output = function(*args)
            wrapper.cache[cache_key] = output
        return output

    wrapper.cache = dict()
    return wrapper


def countcall(function):
    """Подсчитывает количество вызовов функции"""

    @wraps(function)
    def wrapper(*args, **kwargs):
        wrapper.count += 1
        result = function(*args, **kwargs)
        print(f'{function.__name__} has been called {wrapper.count} times')
        return result

    wrapper.count = 0
    return wrapper


def repeat(number_of_times: int):
    """Вызов ф-и несколько раз подряд"""

    def decorate(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            res = [None] * number_of_times
            for i in range(number_of_times):
                res[i] = function(*args, **kwargs)
            return res

        return wrapper

    return decorate


def retry(num_retries, exception_to_check, sleep_time=0):
    """Заставляет функцию, которая сталкивается с исключением, совершить несколько повторных попыток"""

    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for i in range(1, num_retries + 1):
                try:
                    return func(*args, **kwargs)
                except exception_to_check as e:
                    print(f"{func.__name__} raised {e.__class__.__name__}. Retrying...")
                    if i < num_retries:
                        time.sleep(sleep_time)
            # Инициирование исключения, если функция оказалось неуспешной после указанного количества повторных попыток
            raise e

        return wrapper

    return decorate


def rate_limited(max_per_second):
    """Ограничивает частоту вызова функции"""
    min_interval = 1.0 / float(max_per_second)

    def decorate(function):
        last_time_called = [0.0]

        @wraps(function)
        def wrapper(*args, **kwargs):
            elapsed = time.perf_counter() - last_time_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0: time.sleep(left_to_wait)
            ret = function(*args, **kwargs)
            last_time_called[0] = time.perf_counter()
            return ret

        return wrapper

    return decorate


"""
Встроенный декоратор @lru_cache из functools кэширует возвращаемые значения функции, 
используя при заполнении кэша алгоритм LRU - алгоритм замещения наименее часто используемых значений.

Применим этот декоратор для длительно выполняющихся задач, 
которые не меняют результат при одинаковых входных данных, 
например для запроса к базе данных, запроса статической удаленной веб-страницы и выполнения сложной обработки.
"""


@repeat(3)
@timeit
@lru_cache(maxsize=None)
def heavy_processing(n, show=False):
    time.sleep(n)
    return 'pip'


class Movie:
    def __init__(self, r):
        self._rating = r

    """используется для определения свойств класса, 
    которые по сути являются методами getter, setter и deleter для атрибута экземпляра класса.

    Используя декоратор @property, можно определить метод как свойство класса и получить к нему доступ, 
    как к атрибуту класса, без явного вызова метода.

    Это полезно, если нужно добавить некоторые ограничения и логику проверки 
    в отношении получения и установки значения."""

    @property
    def rating(self):
        return self._rating

    @rating.setter
    def rating(self, r):
        if 0 <= r <= 5:
            self._rating = r
        else:
            raise ValueError("The movie rating must be between 0 and 5!")


# позволяет функции иметь различные реализации для разных типов аргументов.
@singledispatch
def fun(arg):
    print("Called with a single argument")


@fun.register(int)
def _(arg):
    print("Called with an integer")


@fun.register(list)
def _(arg):
    print("Called with a list")


@deprecated('Этой функции не будет в следующей версии!')
def _foo(n):
    s = 0
    for i in range(n):
        s += n ** (0.5 + i)
    return s


"""
Декоратор @dataclass в Python используется для декорирования классов.

Он автоматически генерирует магические методы, 
такие как __init__, __repr__, __eq__, __lt__ и __str__ для классов, 
которые в основном хранят данные. 
Это позволяет сократить объем кода и сделать классы более читаемыми и удобными для сопровождения.

Он также предоставляет готовые методы для элегантного представления объектов, 
преобразования их в формат JSON, обеспечения их неизменяемости и т.д.
"""


@dataclass
class Person:
    first_name: str
    last_name: str
    age: int
    job: str

    def __eq__(self, other):
        if isinstance(other, Person):
            return self.age == other.age
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Person):
            return self.age < other.age
        return NotImplemented


if __name__ == '__main__':
    print(heavy_processing(2))
    heavy_processing(2)

    print(_foo(10))

    exit()

    heavy_processing(3)
    batman = Movie(2.5)
    batman.rating

    fun(1)  # Выводит "Called with an integer"

    john = Person(first_name="John",
                  last_name="Doe",
                  age=30,
                  job="doctor", )

    anne = Person(first_name="Anne",
                  last_name="Smith",
                  age=40,
                  job="software engineer", )

    print(john == anne)
