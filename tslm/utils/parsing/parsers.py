import os

import lark
import sympy
from sympy.parsing.latex import parse_latex
from sympy.parsing.latex.lark.latex_parser import LarkLaTeXParser


class myLatexParser(LarkLaTeXParser):
    def __init__(self, grammar_file, parser="earley", ambiguity="explicit"):
        super().__init__(grammar_file=grammar_file)

        if grammar_file is None:
            raise ValueError("Grammar file is required")
        else:
            with open(grammar_file, encoding="utf-8") as f:
                latex_grammar = f.read()

        # Passing different path isn't working.
        # grammar_dir_path = os.path.dirname(grammar_file)
        # print(f'{grammar_file=} {grammar_dir_path=}')

        import sympy.parsing.latex.lark.latex_parser
        fpath = sympy.parsing.latex.lark.latex_parser.__file__
        # Using a hack to pass the same path that LarkLaTeXParser does.
        grammar_dir_path = os.path.join(os.path.dirname(fpath), "grammar/")


        self.parser = lark.Lark(
            latex_grammar,
            source_path=grammar_dir_path,
            parser=parser,
            start="latex_string",
            lexer="auto",
            ambiguity=ambiguity,
            propagate_positions=False,
            maybe_placeholders=False,
            keep_all_tokens=True)


grammar_dir = os.path.join(os.path.dirname(__file__), "grammar")
grammar_file1 = f"{grammar_dir}/latex.lark"
grammar_file2 = f"{grammar_dir}/latex_restricted.lark"

larklatexparser1 = LarkLaTeXParser(grammar_file=grammar_file1)
larklatexparser2 = LarkLaTeXParser(grammar_file=grammar_file2)
larklatexparser3 = myLatexParser(grammar_file=grammar_file2, parser="earley", ambiguity="explicit")
larklatexparser4 = myLatexParser(grammar_file=grammar_file2, parser="earley", ambiguity="resolve")

def parse_latex0(s: str) -> sympy.Expr:
    """Original parser (with antlr backend)"""
    return parse_latex(s, backend="antlr")  

def parse_latex1(s: str) -> sympy.Expr:
    """grammar1-lark-earley-explicit (Sympy's default parser)"""
    pp = larklatexparser1.doparse(s)
    if isinstance(pp, lark.Tree):
        return pp.children[0]
    else:
        return pp

def parse_latex2(s: str) -> sympy.Expr:
    """grammar2-lark-earley-explicit (Restricted the grammar)"""
    pp = larklatexparser2.doparse(s)
    if isinstance(pp, lark.Tree):
        return pp.children[0]
    else:
        return pp

def parse_latex3(s: str) -> sympy.Expr:
    """grammar2-lark-earley-explicit (should exactly match `parse_latex2`)"""
    pp = larklatexparser3.doparse(s)
    if isinstance(pp, lark.Tree):
        return pp.children[0]
    else:
        return pp

def parse_latex4(s: str) -> sympy.Expr:
    """grammar2-lark-earley-resolve (Lark resolves ambiguities automatically)"""
    return larklatexparser4.doparse(s)
