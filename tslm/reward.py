import re
import random
from typing import Optional
from sympy import E, exp, log
import traceback
import numpy as np
import sympy
from sympy.parsing.latex import parse_latex
from tslm.utils.parsing.parsers import parse_latex0, parse_latex1, parse_latex2, parse_latex3, parse_latex4

from verl.utils.reward_score.math import remove_boxed, last_boxed_only_string

from tslm.utils.timeout import process_timeout
from tslm.utils.sympy import is_finite, is_valid_expression




def fix_log_arguments(content: str) -> str:
    log_pattern = r'\\log\s*{([^}]+)}'
    def log_replace(match):
        arg = match.group(1)
        # Clean up the argument
        arg = arg.replace('\\', '')
        return r'\log(' + arg + ')'
    
    content = re.sub(log_pattern, log_replace, content)
    content = re.sub(r'\\log\s*\\\{([^}]+)\\\}', r'\\log(\1)', content)
    return content

def fix_trig_functions(content: str) -> str:
    trig_funcs = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc']
    for func in trig_funcs:
        # Fix function arguments with curly braces
        pattern = fr'\\{func}\s*{{([^}}]+)}}'
        def fix_trig_arg(match):
            arg = match.group(1)
            arg = re.sub(r'\s+', ' ', arg.strip())
            return f'\\{func}({arg})'
        # Fix function arguments with parentheses
        pattern = fr'\\{func}\s*\(([^)]+)\)'
        content = re.sub(pattern, fr'\\{func}(\1)', content)
    return content

def sanitize_latex_str(content: str) -> str:
    """
    Sanitize a LaTeX string by removing \big, \Big, etc.

    """
    content = content.strip().rstrip('.')

    content = re.sub(r"\\[Bb]igg[lr]?", "", content) # Remove \biggl, \Biggl, etc.
    content = re.sub(r"\\[Bb]ig[lr]?", "", content) # Remove \big, \Big, etc.
    content = re.sub(r"\\[Ll]eft[lr]?", "", content) # Remove \left, \Left, etc.
    content = re.sub(r"\\[Rr]ight[lr]?", "", content) # Remove \right, \Right, etc.
    content = re.sub(r"\\[,;]", " ", content) # Replace , and ; with a space.
    content = content.replace(r'\,', r'\cdot') 

    content = content.replace(r"\ln", r"\log") # ln -> log
    content = content.replace("dfrac", "frac") # dfrac -> frac

    power_pattern = r'(\d+)\s*\^{([^}]+)}'
    content = re.sub(power_pattern, r'(\1^{\2})', content)  

    content = fix_trig_functions(content)
    content = fix_log_arguments(content)
    char_map = {
        # powers (1-9)
        "¹": r"^1",
        "²": r"^2",
        "³": r"^3",
        "⁴": r"^4",
        "⁵": r"^5",
        "⁶": r"^6",
        "⁷": r"^7",
        "⁸": r"^8",
        "⁹": r"^9",
        # sqrts
        "√": r"\sqrt",
        "∛": r"\sqrt[3]",
        "∜": r"\sqrt[4]",
    }

    for char, latex in char_map.items():
        content = content.replace(char, latex)

    select_functions = [
        'sin', 'cos', 'tan', 'cot', 'sec', 'csc',
        'sinh', 'cosh', 'tanh', 'coth', 'sech', 'csch',
        'exp', 'log', 'ln', 'sqrt', 'frac'
    ]
    for func in select_functions:
        # Change occurrences of func that aren't preceded by \ by adding a \ before them
        content = re.sub(rf"(?<!\\)\b{func}\b", rf"\\{func}", content)

    content = content.split("=")[-1].strip()
    return content

def extract_boxed_content(text: str) -> Optional[str]:
    """
    Extract the content inside a LaTeX \\boxed{} expression.
    """
    boxed_content = last_boxed_only_string(text)
    if boxed_content is not None:
        try:
            content = remove_boxed(boxed_content)
        except Exception:
            return None

        return content
    return None

def convert_exp_notation(expr: sympy.Expr) -> sympy.Expr:
    """
    Convert expressions like E**x to exp(x).
    """
    from sympy import E, exp

    expr = expr.replace(
        lambda x: x.is_Pow and (x.base == E or (x.base.is_Symbol and x.base.name == "e")),
        lambda x: exp(x.exp),
    )

    expr = expr.replace(
        lambda x: x.func == log,
        lambda x: log(x.args[0])
    )

    return expr

def parse_str_to_expr(expr_str, latex_parser_func = parse_latex):
    """Parse a LaTeX string into a sympy expression.
    """
    expr_str = sanitize_latex_str(expr_str)
    expr = latex_parser_func(expr_str)
    expr = convert_exp_notation(expr)
    return expr

def validate_answer_format(pred_expr: sympy.Expr, gt_expr: sympy.Expr):
    # If symbols are not the same, return False
    return pred_expr.free_symbols == gt_expr.free_symbols

def trig_and_simplify(expr):
    diff_expr_trig = sympy.expand_trig(expr)
    return sympy.trigsimp(diff_expr_trig)

def expand_and_simplify(expr):
    return sympy.expand(sympy.simplify(expr))

def is_expr_equal(pred_expr: sympy.Expr, gt_expr: sympy.Expr, timeout: float = 10):
    try:

        simplifiers = [          
            sympy.simplify,
            trig_and_simplify,
            expand_and_simplify
        ]

        diff_expr = pred_expr - gt_expr
        for simplifier in simplifiers:
            diff_expr_simple = process_timeout(timeout, simplifier, args=(diff_expr,))
            if diff_expr_simple.is_zero or diff_expr_simple == 0 or process_timeout(timeout, is_expr_equal_numeric, args=(diff_expr_simple, sympy.S.Zero)):
                return True
    except TimeoutError:
        try:
            return process_timeout(timeout, is_expr_equal_numeric, args=(pred_expr, gt_expr))
        except TimeoutError:
            return False

def is_expr_equal_numeric(pred_expr: sympy.Expr, gt_expr: sympy.Expr):
    """ Evaluate the 2 expressions at 100 points and check if they are equal.
    """
    SAMPLE_RANGE = 1
    EVAL_POINTS = 100
    num_close = 0
    for _ in range(EVAL_POINTS):
        subs = {symbol: random.uniform(-SAMPLE_RANGE, SAMPLE_RANGE)  for symbol in (pred_expr.free_symbols | gt_expr.free_symbols)}
        pred_val = pred_expr.subs(subs)
        gt_val = gt_expr.subs(subs)
        try:
            if is_finite(pred_val) and is_finite(gt_val):
                pred_val = complex(pred_val.evalf())
                gt_val = complex(gt_val.evalf())
                if not np.isclose(pred_val, gt_val, equal_nan=True):
                    return False
                else:
                    num_close += 1
        except Exception:
            continue
    return num_close >= (EVAL_POINTS * 0.1)



def compute_score(solution_str, ground_truth, format_score=0.1, score=1., do_print=True, latex_parser_func = parse_latex3):
    """The scoring function for calculus task.

    Args:
        solution_str: the solution text
        ground_truth: the ground truth sympy expression, as a string
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    predicted_answer = extract_boxed_content(solution_str)
    do_print = do_print and random.randint(1, 64) == 1

    if do_print:
        print(f"--------------------------------")
        print(f"Ground truth: {ground_truth} | Extracted answer: {predicted_answer}")
        print(f"Solution string: {solution_str}", flush=True)

    if predicted_answer is None or predicted_answer == "":
        if do_print:
            print(f"No boxed content in model output.", flush=True)
        return 0

    # Check if they're equivalent using sympy
    try:

        if do_print:
            print(f'parsing {ground_truth=}')
        target= parse_str_to_expr(ground_truth, latex_parser_func=parse_latex)

        pred= parse_str_to_expr(predicted_answer, latex_parser_func=latex_parser_func)

        if not validate_answer_format(pred, target) or not is_valid_expression(pred):
            if do_print:
                print(f"Invalid format: {predicted_answer}", flush=True)
            return 0
        if do_print:
            print(f'validated answer format and expression')


        if is_expr_equal(target, pred, timeout=1):
            if do_print:
                print(f"Correct answer: {predicted_answer}", flush=True)
            return score
        else:   
            if do_print:
                print(f"Incorrect answer: {predicted_answer}", flush=True)
            return format_score

    except Exception as e:
        if do_print:
            print(f"Parsing error", flush=True)
            print(f'Exception {e}', flush=True)
            traceback.print_exc()
        return 0


parsing_func_list = [parse_latex,parse_latex0, parse_latex1, parse_latex2, parse_latex3, parse_latex4]

def compute_score_cascade(solution_str, ground_truth, format_score=0.1, score=1., do_print=True, latex_parser_func_list = parsing_func_list):
    # Itearate over the latex parser functions and return the best score
    pred_str = extract_boxed_content(solution_str)
    best_score = 0
    for i,latex_parser_func in enumerate(latex_parser_func_list):
        cur_score = compute_score(solution_str, ground_truth, format_score=format_score, score=score, do_print=do_print, latex_parser_func=latex_parser_func)
    
        if cur_score > 0.9:
            best_score = cur_score
            break
        else:
            best_score = cur_score       
    return best_score