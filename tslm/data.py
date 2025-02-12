"""
Generate a calculus dataset of differentiation, and save it to parquet format
"""

import os
import re
import random
from enum import Enum
from multiprocessing import Pool
from random import randint

from datasets import Dataset, load_dataset
import sympy
from sympy import symbols, diff, sympify, latex
from tqdm import tqdm
import typer

from tslm.utils.timeout import process_timeout
from tslm.utils.sympy import is_finite, is_valid_expression
from tslm.reward import parse_str_to_expr

def expr_to_str(f: sympy.Expr) -> str:
    f_str = str(f)
    f_str = f_str.replace('**', '^') # replace x**2 with x^2
    return f_str

def expr_to_latex(f: sympy.Expr) -> str:
    f_str = latex(f)
    f_str = re.sub(r"\\[Bb]ig[lr]?", "", f_str) # Remove \big, \Big, etc.
    f_str = re.sub(r"\\[Ll]eft[lr]?", "", f_str) # Remove \left, \Left, etc.
    f_str = re.sub(r"\\[Rr]ight[lr]?", "", f_str) # Remove \right, \Right, etc.
    return f_str

def simple_simplify(f: sympy.Expr) -> sympy.Expr:
    # Apply the following simplifications, 3 times each:
    # logcombine
    # signsimp
    # collect
    # rcollect
    # trigsimp
    # exptrigsimp
    # powsimp

    for _ in range(3):
        f = sympy.logcombine(f)
        f = sympy.signsimp(f)
        f = sympy.collect(f, sympy.Symbol('x'))
        f = sympy.rcollect(f, sympy.Symbol('x'))
        f = sympy.trigsimp(f)
        f = sympy.exptrigsimp(f)
        f = sympy.powsimp(f)

    return f

def normalize_weights(weights: list[float]) -> list[float]:
    """Normalize weights so that they sum to 1."""
    sum_weights = sum(weights)
    if sum_weights == 0:
        raise ValueError("Weights sum to 0")
    return [w / sum_weights for w in weights]

def apply_unary_op(f: sympy.Expr, max_coefficient: int = 5) -> sympy.Expr:
    """Apply a unary operator to a function."""
    unary_ops = [
        (0.15, lambda f: apply_binary_op(f, randint(-max_coefficient, max_coefficient))),
        # Trigonometric functions
        (0.15, lambda f: sympy.sin(f)),
        (0.15, lambda f: sympy.cos(f)),
        (0.08, lambda f: sympy.tan(f)),
        (0.04, lambda f: sympy.sec(f)),
        (0.04, lambda f: sympy.csc(f)),
        (0.04, lambda f: sympy.cot(f)),
        # Exponential and logarithmic
        (0.12, lambda f: sympy.exp(f)),
        (0.08, lambda f: sympy.log(f)),
        # Hyperbolic functions
        (0.04, lambda f: sympy.sinh(f)),
        (0.04, lambda f: sympy.cosh(f)),
        (0.04, lambda f: sympy.tanh(f)),
        # Other functions
        (0.08, lambda f: sympy.sqrt(f)),
        (0.04, lambda f: 1/f),
        (0.04, lambda f: f**2),
    ]
    weights, ops = zip(*unary_ops)
    weights = normalize_weights(weights)
    retval = random.choices(ops, weights=weights)[0](f)
    if not is_finite(retval):
        return f
    return retval

def apply_binary_op(f1: sympy.Expr, f2: sympy.Expr) -> sympy.Expr:
    """Apply a binary operator to two functions."""
    binary_ops = [
        (0.30, lambda a, b: a + b),
        (0.30, lambda a, b: a * b),
        (0.30, lambda a, b: a - b),
        (0.09, lambda a, b: a/b),
        (0.01, lambda a, b: a**b),
    ]
    weights, ops = zip(*binary_ops)
    weights = normalize_weights(weights)
    retval = random.choices(ops, weights=weights)[0](f1, f2)
    if not is_finite(retval):
        return f1
    return retval

def compose_functions(x: sympy.Symbol, depth: int = 0, max_depth: int = 3, max_coefficient: int = 20, max_power: int = 5) -> sympy.Expr:
    """Recursively compose functions using unary and binary operations."""
    if depth >= max_depth or random.random() < 0.2:
        return x

    def has_x(f):
        return x in f.free_symbols

    # Choose between unary and binary composition
    if random.random() < 0.4:  # 40% chance for unary
        f = compose_functions(x, depth + 1, max_depth, max_coefficient, max_power)
        if has_x(f):
            return apply_unary_op(f)
        else:
            return f
    else:  # 60% chance for binary
        f1 = compose_functions(x, depth + 1, max_depth, max_coefficient, max_power)
        f2 = compose_functions(x, depth + 1, max_depth, max_coefficient, max_power)
        if has_x(f1) or has_x(f2):
            return apply_binary_op(f1, f2)
        else:
            return f1

def generate_expression_simple(max_coefficient: int = 5, max_power: int = 5, is_task_integration: bool = False) -> tuple[sympy.Expr, str]:
    templates_all = {
        'polynomial': [
            '{a}*x**{n} + {b}*x**{m}',
            '{a}*x**{n} + {b}*x**{m} + {c}*x**{o} + {d}',
        ],
        'trigonometric': [
            '{a}*sin({b}*x + {c})',
            '{a}*cos({b}*x + {c})',
            '{a}*tan({b}*x + {c})',
            '{a}*cot({b}*x + {c})',
            '{a}*sec({b}*x + {c})',
            '{a}*csc({b}*x + {c})',
            '{a}*sin({b}*x)*cos({c}*x)',
            '{d}*sin({a}*x)*sin({b}*x + {c})',
            '{d}*cos({a}*x)*cos({b}*x + {c})',
            '{d}*sin({a}*x)*cos({b}*x + {c})',
            '{d}*cos({a}*x)*sin({b}*x + {c})',
        ],
        'exponential': [
            '{a}*E^({b}*x + {c})',
            '{a}*{b}^({c}*x + {d})',
            '{a}*x*E^({b}*x + {d})',
            '{a}*E^({b}*x) + {c}*E^({d}*x)',
        ],
        'logarithmic': [
            '{a}*log({b}*x + {c})',
            '{a}*x*log({b}*x + {d})',
            '{a}*log({b}*x^2 + {c}*x + {d})',
            '{a}*log({b}*x)/{c}*x + {d}',
        ],
        'composite': [
            '{a}*sin({b}*x)*E^({c}*x + {d})',
            '{a}*E^({b}*x + {d})*x^{n}',
            '{a}*sin({b}*x)*cos({c}*x) + {d}*E^x',
        ]
    }
    templates_differentiation_only = {
        'trigonometric': [
            '{a}*sin({b}*x^2 + {c}*x + {d})',
            '{a}*cos({b}*x^2 + {c}*x + {d})',
        ],
        'exponential': [
            '{a}*E^({b}*x^2 + {c}*x + {d})',
        ],
        'composite': [
            '{a}*log({b}*x + {d})*cos({c}*x)',
            '{a}*sin({b}*x^2 + {d})*log({c}*x)',
        ]
    }

    if not is_task_integration:
        for k in templates_all:
            if k in templates_differentiation_only:
                templates_all[k].extend(templates_differentiation_only[k])

    # Randomly select function type and template
    function_type = random.choice(list(templates_all.keys()))
    template = random.choice(templates_all[function_type])

    # Generate random coefficients
    params = {
        'a': random.randint(-max_coefficient, max_coefficient),
        'b': random.randint(-max_coefficient, max_coefficient),
        'c': random.randint(-max_coefficient, max_coefficient),
        'd': random.randint(-max_coefficient, max_coefficient),
        'n': random.randint(1, max_power),
        'm': random.randint(1, max_power),
        'o': random.randint(1, max_power)
    }

    # Create the function string
    func_str = template.format(**params)
    func = simple_simplify(sympify(func_str))

    return func, function_type

def generate_expression_complex(max_coefficient: int = 20, max_power: int = 5) -> tuple[sympy.Expr, str]:
    MAX_DEPTH = 4
    x = symbols('x')
    try:
        func = compose_functions(x, max_depth=MAX_DEPTH, max_coefficient=max_coefficient, max_power=max_power)
        func = simple_simplify(func)
        if is_finite(func):
            return func, 'composite-complex'
        else:
            return None, None
    except Exception as e:
        return None, None

def generate_expression(max_coefficient: int = 5, max_power: int = 5,
                        simplicity: float = 0.01, is_task_integration: bool = False) -> tuple[sympy.Expr, str]:
    if random.random() < simplicity:
        return generate_expression_simple(max_coefficient=max_coefficient, max_power=max_power, is_task_integration=is_task_integration)
    else:
        return generate_expression_complex(max_coefficient=max_coefficient, max_power=max_power)

## Differentiation ##
def generate_differentiation_example(max_coefficient=5, max_power=5, simplicity=0.01, max_length=None):
    func, function_type = generate_expression(
                                    max_coefficient=max_coefficient,
                                    max_power=max_power,
                                    simplicity=simplicity,
                                    is_task_integration=False
                                )
    if func is None or not is_valid_expression(func) or not is_finite(func):
        return None

    try:
        # Compute derivative
        derivative = diff(func, symbols('x'))
        if not is_valid_expression(derivative) or not is_finite(derivative):
            return None

        # Get string expression
        func_str = expr_to_latex(func)
        derivative_str = expr_to_latex(derivative)

        # Skip examples that are too long/hard
        if max_length is not None and len(func_str) > max_length:
            return None
        if max_length is not None and len(derivative_str) > max_length:
            return None

        example = {
            'function_type': function_type,
            'function': func_str,
            'derivative': derivative_str,
        }
        return example
    except Exception as e:
        return None

def _worker_function_differentiation(args):
    """Helper function to unpack arguments for multiprocessing"""
    try:
        # Timeout after 5 seconds
        return process_timeout(5, generate_differentiation_example, args=args)
    except TimeoutError:
        return None
    except Exception:
        raise

def generate_differentiation_dataset_parallel(num_examples=1000, max_coefficient=5, max_power=5, 
                                            simplicity=0.01, num_workers=10, max_length=None):
    # Create arguments tuple for each call
    args_iter = ((max_coefficient, max_power, simplicity, max_length) for _ in range(num_examples))

    with Pool(num_workers) as p:
        try:
            # Use imap for interruptible iteration
            data = []
            for result in tqdm(
                p.imap_unordered(_worker_function_differentiation, args_iter),
                total=num_examples,
                desc="Generating differentiation dataset"
            ):
                if result is not None:
                    data.append(result)
        except KeyboardInterrupt:
            p.terminate()  # Terminate all worker processes
            p.join()       # Wait for them to exit
            raise
    return data



def main_generate(
    local_dir: str = "./data/calculus-differentiation",
    simplicity: float = 1.0,
    N_train: int = 327680,
    N_test: int = 4096,
    N_dedup_factor: float = 1.5,
    workers: int = 20,
):
    """Generate calculus dataset for differentiation."""
    MAX_COEFFICIENT = 40
    MAX_POWER = 5
    MAX_LENGTH = 100

    # Generate more to account for deduplication
    N = int((N_train + N_test) * N_dedup_factor) 

    dataset = generate_differentiation_dataset_parallel(
            num_examples=N,
            max_coefficient=MAX_COEFFICIENT,
            max_power=MAX_POWER,
            simplicity=float(simplicity),
            num_workers=workers,
            max_length=MAX_LENGTH,
        )

    # Keep only unique func_str (data[1])
    filtered_dataset = []
    all_func_str = set()
    for item in dataset:
        if item['function'] not in all_func_str:
            all_func_str.add(item['function'])
            filtered_dataset.append(item)
    dataset = filtered_dataset

    # Split dataset into train and test
    if len(dataset) < N_train + N_test:
        raise ValueError(
            f'{len(dataset)=} {N_train=} {N_test=}.'
            f'Increase --n-dedup-factor={N_dedup_factor}.'
        )
    train_dataset = dataset[:N_train]
    test_dataset = dataset[-N_test:]

    def to_dataset(dataset_list):
        dataset_dict = {
            "function_type": [],
            "function": [],
            "derivative": [],
        }
        for dp in dataset_list:
            dataset_dict["function_type"].append(dp['function_type'])
            dataset_dict["function"].append(dp['function'])
            dataset_dict["derivative"].append(dp['derivative'])
        return Dataset.from_dict(dataset_dict)

    train_dataset = to_dataset(train_dataset)
    test_dataset = to_dataset(test_dataset)
    train_dataset.to_parquet(os.path.join(local_dir, 'train.raw.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.raw.parquet'))


# Prompts. Multiple prompts are used to test OOD generalization.
SYSTEM_PROMPT = f"""Please reason step by step, and put your final answer within \\boxed{{}}."""
from tslm.user_prompts import USER_PROMPTS, SINGLE_USER_PROMPT

def dropout_char(s: str, char_to_drop: str, prob: float = 0.5) -> str:
    s_chars = list(s)
    for i, c in enumerate(s_chars):
        if c == char_to_drop and random.random() < prob:
            s_chars[i] = ""
    return "".join(s_chars)

def vary_latex_str(latex_str: str) -> str:
    """Generate a variation of the latex string."""
    # 99% chance to remove the \big, \Big, etc.
    if random.random() < 0.99:
        latex_str = re.sub(r"\\[Bb]ig[lr]?", "", latex_str) # Remove \big, \Big, etc.
        latex_str = re.sub(r"\\[Ll]eft[lr]?", "", latex_str) # Remove \left, \Left, etc.
        latex_str = re.sub(r"\\[Rr]ight[lr]?", "", latex_str) # Remove \right, \Right, etc.

    # 50% chance to replace ^ with **
    if random.random() < 0.5:
        latex_str = latex_str.replace("^", "**")

    # 50% chance to replace cdot with *
    if random.random() < 0.5:
        latex_str = latex_str.replace("\\cdot", "*")

    # 50% chance to replace {} with ()
    if random.random() < 0.5:
        latex_str = latex_str.replace("{", "(").replace("}", ")")

    # 50% chance to remove extra spaces near brackets
    if random.random() < 0.5:
        latex_str = latex_str.replace("( ", "(").replace(" )", ")")
        latex_str = latex_str.replace("{ ", "{").replace(" }", "}")

    # Each space has a 50% chance to be removed
    latex_str = dropout_char(latex_str, " ", prob=0.5)

    # 50% chance to remove all extra spaces
    if random.random() < 0.1:
        latex_str = latex_str.replace("  ", " ")
    # 5% cahance to remove all spaces
    if random.random() < 0.05:
        latex_str = latex_str.replace(" ", "")

    # 50% chance to remove all \ i.e. \sin -> sin
    if random.random() < 0.5:
        # Remove all 'cdot'
        latex_str = latex_str.replace("\\cdot", "")
        latex_str = latex_str.replace("\\", "")

    return latex_str

def vary_sympy_expr_str(expr_str: str) -> str:
    """Vary the expression string."""
    # 50% chance to convert ** to ^
    if random.random() < 0.5:
        expr_str = expr_str.replace("**", "^")

    # Space drop out. Each space has a 50% chance to be removed
    expr_str = dropout_char(expr_str, " ", prob=0.5)

    # 10% chance to remove all spaces
    if random.random() < 0.1:
        expr_str = expr_str.replace(" ", "")

    return expr_str

def generate_expression_variation(expr_str: str) -> str:
    """Generate a variation of the expression string."""
    expr = parse_str_to_expr(expr_str)

    # Randomly choose between latex and sympy
    if random.random() < 0.5:
        expr_str = expr_to_latex(expr)
        expr_str = vary_latex_str(expr_str)
    else:
        expr_str = expr_to_str(expr)
        expr_str = vary_sympy_expr_str(expr_str)

    return expr_str

def main_format(
    local_dir: str = "./data/calculus-differentiation",
    filesuffix: str = '',
    apply_variation: bool = True,
    workers: int = 20,
):
    """Format the generated dataset with prompts."""

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            # Make prompt dict
            if apply_variation:
                expression_variation = generate_expression_variation(example['function'])
                user_prompt_template = random.choice(USER_PROMPTS)
            else:
                expression_variation = example['function']
                user_prompt_template = SINGLE_USER_PROMPT
            user_prompt = user_prompt_template.format(function=expression_variation)
            prompt = [
                {
                    "role": "system",
                    "content": SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": user_prompt,
                }
            ]
            prompt_user_only = [{
                "role": "user",
                "content": SYSTEM_PROMPT + " " + user_prompt,
            }]

            data = {
                "data_source": "calculus-differentiation",
                "prompt": prompt,
                "prompt_user_only": prompt_user_only,
                "ability": "math",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": example['derivative']
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                    'function': example['function'],
                    'derivative': example['derivative'],
                }
            }
            return data

        return process_fn

    dataset = load_dataset(local_dir, data_files={'train': 'train.raw.parquet', 'test': 'test.raw.parquet'})
    train_dataset = dataset['train'].map(function=make_map_fn('train'), with_indices=True, num_proc=workers)
    test_dataset = dataset['test'].map(function=make_map_fn('test'), with_indices=True, num_proc=workers)

    train_path = os.path.join(local_dir, f'train{filesuffix}.parquet')
    test_path = os.path.join(local_dir, f'test{filesuffix}.parquet')
    if os.path.exists(train_path) or os.path.exists(test_path):
        raise ValueError(f'{train_path=} or {test_path=} already exists')
    train_dataset.to_parquet(train_path)
    test_dataset.to_parquet(test_path)

def main_generate_and_format(
    local_dir: str = "./data/calculus-differentiation",
    simplicity: float = 1.0,
    N_train: int = 327680,
    N_test: int = 4096,
    N_dedup_factor: float = 1.5,
    workers: int = 20,
):
    main_generate(local_dir, simplicity, N_train, N_test, N_dedup_factor, workers)
    main_format(local_dir, workers=workers)


if __name__ == "__main__":
    app = typer.Typer()
    app.command()(main_generate)
    app.command()(main_format)
    app.command()(main_generate_and_format)
    app()
