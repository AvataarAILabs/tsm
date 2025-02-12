import sympy
import sympy.logic.boolalg

forbidden_sympy_nodes = [
    # Comparison/Relational/Logical Nodes (all inherit from Bool)
    # But cannot use bool because symbols also inherit from bool, which are allowed.
    # sympy.logic.boolalg.Boolean,  # Parent class for all logical operations
    sympy.Rel,  # Parent class for all relational operations
    sympy.logic.boolalg.BooleanFunction,

    # Unevaluated Functions
    sympy.Derivative, sympy.Integral, sympy.Lambda, 
    sympy.Sum, sympy.Product, sympy.Limit, sympy.Subs,

    # MinMax
    sympy.Min, sympy.Max,
    
    # Special Functions/Operations
    sympy.DiracDelta, sympy.Heaviside, sympy.SingularityFunction, 
    sympy.Subs, sympy.Lambda, sympy.WildFunction, 
    
    # Set Operations (all inherit from Set)
    sympy.Set,  # Parent class for all set operations
    
    # Too hard to handle, or irrelevant functions
    sympy.factorial, sympy.factorial2, sympy.rf, sympy.ff, sympy.binomial,
    sympy.RisingFactorial, sympy.FallingFactorial, sympy.subfactorial, sympy.carmichael,
    sympy.fibonacci, sympy.lucas, sympy.motzkin, sympy.tribonacci, sympy.harmonic, sympy.bernoulli, sympy.bell, sympy.euler,
    sympy.catalan, sympy.genocchi, sympy.andre, sympy.partition,
    sympy.conjugate, sympy.arg, sympy.polar_lift,
    sympy.periodic_argument, sympy.unbranched_argument, sympy.principal_branch, sympy.transpose,
    sympy.adjoint, sympy.polarify, sympy.unpolarify,
    sympy.floor, sympy.ceiling, sympy.frac, sympy.Piecewise, sympy.piecewise_fold,
    sympy.piecewise_exclusive, sympy.erf, sympy.erfc, sympy.erfi, sympy.erf2, sympy.erfinv, sympy.erfcinv, sympy.erf2inv,
    sympy.Ei, sympy.expint, sympy.E1, sympy.li, sympy.Li, sympy.Si, sympy.Ci, sympy.Shi, sympy.Chi, sympy.fresnels, sympy.fresnelc, sympy.gamma,
    sympy.lowergamma, sympy.uppergamma, sympy.polygamma, sympy.loggamma, sympy.digamma, sympy.trigamma,
    sympy.multigamma, sympy.dirichlet_eta, sympy.zeta, sympy.lerchphi, sympy.polylog, sympy.stieltjes, sympy.Eijk,
    sympy.LeviCivita, sympy.KroneckerDelta, sympy.SingularityFunction, sympy.DiracDelta, sympy.Heaviside,
    sympy.bspline_basis, sympy.bspline_basis_set, sympy.interpolating_spline, sympy.besselj,
    sympy.bessely, sympy.besseli, sympy.besselk, sympy.hankel1, sympy.hankel2, sympy.jn, sympy.yn, sympy.jn_zeros, sympy.hn1,
    sympy.hn2, sympy.airyai, sympy.airybi, sympy.airyaiprime, sympy.airybiprime, sympy.marcumq, sympy.hyper,
    sympy.meijerg, sympy.appellf1, sympy.legendre, sympy.assoc_legendre, sympy.hermite, sympy.hermite_prob,
    sympy.chebyshevt, sympy.chebyshevu, sympy.chebyshevu_root, sympy.chebyshevt_root, sympy.laguerre,
    sympy.assoc_laguerre, sympy.gegenbauer, sympy.jacobi, sympy.jacobi_normalized, sympy.Ynm, sympy.Ynm_c,
    sympy.Znm, sympy.elliptic_k, sympy.elliptic_f, sympy.elliptic_e, sympy.elliptic_pi, sympy.beta, sympy.mathieus,
    sympy.mathieuc, sympy.mathieusprime, sympy.mathieucprime, sympy.riemann_xi, sympy.betainc, sympy.betainc_regularized,
]

def is_finite(f: sympy.Expr) -> bool:
    return not (f.has(sympy.nan, sympy.oo, sympy.zoo) or f.is_infinite)

def is_valid_expression(f: sympy.Expr) -> bool:
    # Check if f is a valid expression, 
    # without any comparisons, derivatives, integrals, equalities etc.
    return not f.has(*forbidden_sympy_nodes)
