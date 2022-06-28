{}# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time
import numpy as np
from fractions import Fraction

VERBOSE      = False
TPL_VERBOSE  = False
CHECK        = True
DEBUG        = False
DEBUG_INDENT = 0

# ----------------------------------------------------------------------------------------------------
# Templates management
#
# Templates are expressions where variables start with $ sign
#
# When comparing a template with a regular expression, we use a variable instancer XVars
# Note that XVars is also used to compute the expressions
#
# Template examples:
# $a^2 - $a^2           = ($a - $b)($a + $b)
# cos($a)^2 + sin($b)^2 = 1
# sin($a)/cos($a)       = tan($a)
#
# ----- SUM template
# A SUM template applies only to SUM expressions. The number of terms in the expression
# must be greated than the number of terms in the template.
# The first term of the template is matched with each possible term in the sum. This gives
# an expression like:
#
# expr = rem + factor*template_first_term
#
# The remainder is then divied by factor and compared with the rest of the template.
#
# ----- PROD template
# A PROD template applies only to PROD expressions. The number of temrs in the epxression
# must be greated than the number of terms in the template.
#
# - Template is a single variable: $a with 7.cos(a).sin(a)^2
#   All the combinations of the expressions are tried: the whole terme and and the partial products:
#   $a = [cos(a), sin(a), sin(a)^2, cos(a).sin(a), cos(a).sin(a)^2]
#
# - Template is a power of single variable: $a^2 with 7.cos(a)^2.sin(a)^2
#   All the combinations of terms with the same exponent which is a mulitple of the template exponent.
#   $a = [cos(a), sin(a), cos(a).sin(a)]
#
# - Template is a function : sin(2*$a)/cos(2*$a) with sin(a)^2/cos(a)^2
#   The argument of the function
#   $a = [a/2]
#
# - The values are not used but taking into account as factors in the result

# ----------------------------------------------------------------------------------------------------
# Template variables instancer

class XVars(dict):
    
    def __init__(self, xvars={}):
        super().__init__(xvars)
    
    @property
    def completed(self):
        for v in self.values():
            if v is None:
                return False            
        return True

    def add_variable(self, name):
        
        # None if doesn't exist or initialized to None
        v = self.get(name)
        
        # Make sure it exists
        if v is None:
            self[name] = None
    
    def set_values(self, seed=None):

        if seed is not None:
            np.random.seed(seed)
        
        for k in self:
            self[k] = np.random.randint(2, 20)
            
    # ----------------------------------------------------------------------------------------------------
    # Exact match between a template and an expression
    # Update the variables values to find an exact match
    
    def template_match(self, template, expression):
        
        def ok_exp():
            if template.e * expression.e < 0:
                return False
            if abs(expression.e.numerator) % abs(template.e.numerator) != 0:
                return False
            return True
        
        if VERBOSE:
            global DEBUG_INDENT
            print(f"{'   '*DEBUG_INDENT}TEMPLATE MATCH: '{template}' in '{expression}', xvars: {self}, ok_exp = {ok_exp()}")
        
        if template.type == VALUE:
            return template == expression
        
        elif template.type == VAR:
            name = template.v
            tmpl = self[name]
            
            if tmpl is None:
                if not ok_exp():
                    return False
                
                self[name] = expression ** (1/template.e)
                return True
            
            else:
                tmpl = tmpl ** template.e
                return tmpl == expression
            
        elif template.type == FUNC:
            
            if expression.type != FUNC or template.v != expression.v or not ok_exp():
                return False
            
            if VERBOSE:
                DEBUG_INDENT += 1

            ok = self.template_match(template.xes[0], expression.xes[0])

            if VERBOSE:
                DEBUG_INDENT -= 1

            return ok
        
        else:
            if expression.type != template.type or expression.e != template.e or len(expression.xes) != len(template.xes):
                return False
            
            xvs = XVars(self)
            matches = []
            for t_i, t_xe in enumerate(template.xes):
                found = False
                for e_i, e_xe in enumerate(expression.xes):
                    if e_i in matches:
                        continue
                    if xvs.template_match(t_xe, e_xe):
                        matches.append(e_i)
                        found = True
                        break
                if not found:
                    return False
                
            # ----- Update self with the new vars from xvs
                
            for k, v in xvs.items():
                self[k] = v
                
            return True

# ----------------------------------------------------------------------------------------------------
# When matching a product template with a product expression, we need to loop on the possible
# combinations. We use a, iteratore for that

class TCompactIter:
    
    def __init__(self, template, expression, xvars):
        
        # --------------------------------------------------
        # Template terms
        
        if template.type == PROD:
            self.t_terms = template.xes
        else:
            self.t_terms = [template]

        # --------------------------------------------------
        # Expression terms
            
        self.expression = expression
        
        if expression.type == PROD:
            self.e_terms = expression.xes
        else:
            self.e_terms = [expression]
            
            
        self.xvars = XVars(xvars)
            
        self.index = 0
        self.total = 0
        self.shape = ()
        self.poss  = []
        
        # ---------------------------------------------------------------------------
        # for each term in the template, check which terms in the expression
        # can be matched
        
        for t_i, t_xe in enumerate(self.t_terms):
            
            ps = []
            if t_xe.type == VALUE:
                pass
            
            elif t_xe.type == VAR:
                ps = [i for i in range(len(self.e_terms))]
                
            elif t_xe.type == FUNC:
                for e_i, e_xe in enumerate(self.e_terms):
                    if e_xe.type == FUNC and e_xe.v == t_xe.v and e_xe.e * t_xe.e > 0 and e_xe.e.numerator % t_xe.e.numerator == 0:
                        ps.append(e_i)
                        
            elif t_xe.type == PROD:
                pass
            
            elif t_xe.type == SUM:
                pass
            
            if len(ps) == 0:
                return
            
            self.poss.append(ps)
            
        self.shape = tuple([len(ps) for ps in self.poss])
        self.total = int(np.product(self.shape))
        
        
    def __repr__(self):
        s  = "TCompactIter:\n"
        s += f"   expression: {self.e_terms}\n"
        s += f"   template:   {self.t_terms}\n"
        s += f"   shape:      {self.shape}\n"
        s += f"   poss:       {self.poss}\n"
        s += f"   index:      {self.index}/{self.total}"
        return s

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        
        for idx in range(self.index, self.total):
            
            # ----- Indices of the current permutation
            # - key:   the index of the template term
            # - value: index in possibilities giving the index of the expression term
            
            poss_indices = np.unravel_index(self.index, self.shape)
            indices = [self.poss[i][poss_indices[i]] for i in range(len(self.t_terms))]
            
            if VERBOSE:
                print("indices", indices)
                for i, ind in enumerate(indices):
                    print(f"   {i:2d}> '{self.t_terms[i]}' --> term {ind} : '{self.e_terms[ind]}'")
            
            # ----- Indices must be different
            
            if len(np.unique(indices)) != len(indices):
                continue
            
            # ----- Ok, let's match the template
            
            xvars = XVars(self.xvars)
            match = None
            for t_i, t_xe in enumerate(self.t_terms):
                term = self.e_terms[indices[t_i]]
                
                if VERBOSE:
                    print(f"template {t_i}: '{t_xe}' match with '{term}' ?")
                
                if not xvars.template_match(t_xe, term):
                    if VERBOSE:
                        print(f"    KO")
                    match = None
                    break
                
                m = t_xe.set_vars(xvars)
                
                if match is None:
                    match = m
                else:
                    match *= m
                    
                if VERBOSE:
                    print(f"    ok: '{m}' --> '{match}'")
                
            if match is None:
                continue
                
            factor = self.expression / match
            self.index = idx + 1
            
            if VERBOSE:
                print(f"COMPACT ITERATION, match proposal: {self.index}/{self.total}")
                print(f"   expression:    {self.expression}")
                print(f"   template:      {self.t_terms}")
                print(f"   match:         {match}")
                print(f"   factor:        {factor}")
                print(f"   xvars:         {xvars}")
            
            return factor, xvars
            

        self.index = self.total        
        raise StopIteration


# ----------------------------------------------------------------------------------------------------
# Expression types


VALUE = 0
VAR   = 1
FUNC  = 2
PROD  = 3
SUM   = 4

TYPES = ['VALUE', 'VAR', 'FUNC', 'PRODUCT', 'SUM']

FUNCTIONS = {
    'sin': 'arcsin',
    'cos': 'arccos',
    'tan': 'arctan',
    }

# ----------------------------------------------------------------------------------------------------
# An expression x^e

class Xe:
    
    def __init__(self, type, *args, e=1, reduce=True):
        
        self.type = type

        self.v    = None
        self.xes  = []

        if self.type in [VALUE, VAR, FUNC]:
            self.v = args[0]
            
        if self.type == FUNC:
            for xe in args[1:]:
                self.xes.append(Xe.xe(xe, clone=True))
                
        elif self.type in [PROD, SUM]:
            for xe in args:
                self.xes.append(Xe.xe(xe, clone=True))
                
        self.e  = e
        self.s_ = None

        
        if self.type in [PROD, SUM] and reduce:
            self.reduce()
            
    # ---------------------------------------------------------------------------
    # The value
    
    @property
    def v(self):
        return self.v_
    
    @v.setter
    def v(self, value):
        if value is None:
            self.v_ = None
            
        elif type(value) is str:
            self.v_ = value
            
        else:
            self.v_ = Fraction(value)
    
    # ---------------------------------------------------------------------------
    # Set the exponent
    
    @property
    def e(self):
        return self.e_
    
    @e.setter
    def e(self, exp):
        
        if self.type == VALUE:
            self.v **= exp
            self.e_ = Fraction(1)
            
        elif self.type == PROD:
            for i in range(len(self.xes)):
                self.xes[i].e *= Fraction(exp)
            self.e_ = Fraction(1)
        
        else:
            self.e_ = Fraction(exp)
            
        self.s_ = None
        
    # ---------------------------------------------------------------------------
    # decompose value * expression

    def value_xe(self):
        
        if self == VALUE:
            return self.clone(), None
        
        elif self.type == PROD:
            val  = 1
            expr = None
            for xe in self.xes:
                if xe.type == VALUE:
                    val *= xe.v
                else:
                    if expr is None:
                        expr = xe.clone()
                    else:
                        expr = expr * xe
                        
            return val**self.e, expr**self.e
        
        else:
            return 1, self.clone()
            
    # ---------------------------------------------------------------------------
    # The x attribute of x^e
        
    @property
    def x(self):
        x    = self.clone()
        x.e  = 1
        return x
    
    @x.setter
    def x(self, value):        
        e = self.e
        self.change_to(value)
        self.e = e
        
        
    # ---------------------------------------------------------------------------
    # Clone
    
    def clone(self):
        
        clone = Xe.Zero() # To avoid calling init which calls reduce()
        
        clone.type  = self.type
        clone.v     = self.v
        clone.xes   = [xe.clone() for xe in self.xes]
        clone.e_    = self.e_
        clone.s_    = self.s_
        
        return clone
        
    # ---------------------------------------------------------------------------
    # Conversion to expression
    
    @staticmethod
    def xe(thing, clone=False):
        if type(thing) is Xe:
            if clone:
                return thing.clone()
            else:
                return thing
            
        elif type(thing) is str:
            return Xe(VAR, thing)
        
        else:
            return Xe(VALUE, thing)
        
    # ---------------------------------------------------------------------------
    # Sub expressions without values
    
    @property
    def noval_xes(self):
        xes = []
        for xe in self.xes:
            if xe.type != VALUE:
                xes.append(xe)
        return xes

    # ---------------------------------------------------------------------------
    # Sorting str
    
    @property
    def str_sort(self):
        s = f"{self.type}_{self.v}"
        if self.type == FUNC:
            sep = '('
            for xe in self.xes:
                s += sep + xe.str_sort
                sep = ','
            s += ')'
            
        elif self.type in [PROD, SUM]:
            op = ' * ' if self.type == PROD else ' + '
            sxs = [xe.str_sort for xe in self.xes]
            sxs.sort()
            sep = ' ('
            for sx in sxs:
                s += sep + '(' + sx + ')'
                sep = op
            s += ')'
            
        s += f"^({self.e})"
        
        return s
    
    # ---------------------------------------------------------------------------
    # Match string
    # Simplified string version to see if detailed match is worth doint
    
    @property
    def str_match(self):
        
        if self.type == VALUE:
            return ""
        
        elif self.type == VAR:
            return self.v

        elif self.type  == FUNC:
            return self.v
        
        strs = []
        for xe in self.xes:
            sx = xe.str_match
            if sx != "":
                strs.append(sx)
                
        if len(strs) == 0:
            return ""
            
        strs.sort()
        
        op = '.' if self.type == PROD else '+'
        
        for i, sx in enumerate(strs):
            if i == 0:
                s = sx
            else:
                s += op + sx
        return s
    
    # ---------------------------------------------------------------------------
    # Match template string
    # Only functions
    
    @property
    def str_template_match(self):

        if self.type in [VALUE, VAR]:
            return ""
        
        elif self.type == FUNC:
            return self.v
        
        elif self.type == PROD:
            nums = []
            dens = []
            
            for i, xe in self.xes:
                sx = xe.str_template_match
                if sx != "":
                    if xe.e > 0:
                        nums.append(sx)
                    else:
                        dens.append(sx)
            nums.sort()
            dens.sort()
            s = ""
            for sx in nums:
                if s == "":
                    s = sx
                else:
                    s += '.' + sx
            for sx in dens:
                s += '/' + sx
                
            return s
        
        elif self.type == SUM:
            strs = []
            
            for i, xe in self.xes:
                sx = xe.str_template_match
                if sx != "":
                    strs.append(sx)

            strs.sort()
            for sx in strs:
                if s == "":
                    s = sx
                else:
                    s += '+' + sx
                
            return s
        
    # ---------------------------------------------------------------------------
    # Conversion to expression
        
    def change_to(self, other):
        
        if type(other) is Xe:
            self.type  = other.type
            self.v     = other.v
            self.xes   = [xe.clone() for xe in other.xes]
            self.e_    = other.e_
            self.s_    = other.s_
            
            return self
        
        else:
            return self.change_to(Xe.xe(other))
        
    # ---------------------------------------------------------------------------
    # Reduction
    
    def reduced(self):
        return self.clone().reduce()
    
    def reduce(self):
        
        check_ = self.clone()
        
        # --------------------------------------------------
        # Value
        
        if self.type == VALUE:
            self.e = 1
            return self.check(check_)
        
        # --------------------------------------------------
        # Variable
        
        elif self.type == VAR:
            return self
        
        # --------------------------------------------------
        # Reduces the child expressions
        
        for xe in self.xes:
            xe.reduce()
            
        # --------------------------------------------------
        # Function
            
        if self.type == FUNC:
            return self.check(check_)
        
        # --------------------------------------------------
        # Product
        
        if self.type == PROD:
            
            # ----- Prod of prods

            i = 0
            while i < len(self.xes):
                xe = self.xes[i]
                if xe.type == PROD:
                    xe.reduce()
                    self.xes.extend(xe.xes)
                    del self.xes[i]
                else:
                    i += 1
                    
            # ----- Propagate the exp

            e = self.e
            if e != 1:
                for i in range(self.xes):
                    self.xes[i].e *= e
                self.e = 1
                
            # ----- Merge the values
            
            val = 1
            i = 0
            while i < len(self.xes):
                if self.xes[i].type == VALUE:
                    val *= self.xes[i].v
                    del self.xes[i]
                else:
                    i += 1
            if val != 1:
                self.xes.insert(0, Xe.Value(val))
                
            # ----- Multiply identical terms
            
            i = 0
            while i < len(self.xes):
                termx = self.xes[i].x
                j = i+1
                while j < len(self.xes):
                    if self.xes[j].x == termx:
                        self.xes[i].e += self.xes[j].e
                        del self.xes[j]
                    else:
                        j += 1
                        
                if self.xes[i].e == 0:
                    del self.xes[i]
                else:
                    i += 1
                    
            # ----- Full reduction
                    
            if len(self.xes) == 0:
                self.change_to(1)
                
            elif len(self.xes) == 1:
                self.change_to(self.xes[0] ** self.e).reduce()
                    
                    
        # --------------------------------------------------
        # Sum
        
        if self.type == SUM:
            
            # ----- Sum of sums
            
            i = 0
            while i < len(self.xes):
                xe = self.xes[i]
                if xe.type == SUM and xe.e == 1:
                    self.xes.extend(xe.xes)
                    del self.xes[i]
                else:
                    i += 1

            # ----- Merge the values
            
            val = 0
            i = 0
            while i < len(self.xes):
                if self.xes[i].type == VALUE:
                    val += self.xes[i].v
                    del self.xes[i]
                else:
                    i += 1
            if val != 0:
                self.xes.insert(0, Xe.Value(val))
                    
                
            # ----- Multiply identical terms
            
            i = 0
            while i < len(self.xes):
                val0, termx = self.xes[i].value_xe()
                val = val0
                j = i+1
                while j < len(self.xes):
                    v, x = self.xes[j].value_xe()
                    if x == termx:
                        val += v
                        del self.xes[j]
                    else:
                        j += 1
                        
                if val == 0:
                    del self.xes[i]
                else:
                    if val != val0:
                        self.xes[i] = val * termx
                    i += 1
                    
            # ----- Full reduction
            
            if len(self.xes) == 0:
                self.change_to(0)
                
            elif len(self.xes) == 1:
                self.change_to(self.xes[0] ** self.e).reduce()
                
        """
                    
        # --------------------------------------------------
        # No sub expressions
        
        if len(self.xes) == 0:
            if self.type == PROD:
                return self.One().check(check_)
            else:
                return self.Zero().check(check_)
            
        # --------------------------------------------------
        # One sub expression
        
        if len(self.xes) == 1:
            e = self.e
            self.change_to(self.xes[0])
            self.e *= e
            self.check(check_)
            return self.reduce()
        
        """
        
        return self.check(check_)        
        
        
    # ---------------------------------------------------------------------------
    # Initializers
    
    @classmethod
    def Value(cls, value):
        return cls(VALUE, value)
    
    @classmethod
    def Zero(cls):
        return cls.Value(0)
    
    @classmethod
    def One(cls):
        return cls.Value(1)
    
    @classmethod
    def Var(cls, name):
        return cls(VAR, name)
    
    @staticmethod
    def Vars(*names):
        vs = []
        for name in names:
            vs.append(Xe.Var(name))
        return tuple(vs)
    
    @classmethod
    def Prod(cls, *args):
        return cls(PROD, *args)

    @classmethod
    def Sum(cls, *args):
        return cls(SUM, *args)

    @classmethod
    def Func(cls, name, *args):
        return cls(FUNC, name, *args)
    
    @classmethod
    def cos(cls, var):
        return cls.Func('cos', var)
    
    @classmethod
    def sin(cls, var):
        return cls.Func('sin', var)
    
    @classmethod
    def tan(cls, var):
        return cls.Func('tan', var)
    
    @property
    def func_inv(self):
        if self.type == FUNC:
            for name, inv in FUNCTIONS.items():
                if name == self.v:
                    return inv
                elif inv == self.v:
                    return name
                
        raise RuntimeError(f"Expression {self} is not a function. Impossible to inverse it!")
            
    # ---------------------------------------------------------------------------
    # Expression counts
    
    @property
    def count(self):
        count = 1
        for xe in self.xes:
            count += xe.count
        return count
    
    # ---------------------------------------------------------------------------
    # Check if contains a variable
    
    def contains(self, var_name):
        if self.type == VAR and self.v == var_name:
            return 1
        else:
            count = 0
            for xe in self.xes:
                count += xe.contains(var_name)
            return count

    # ---------------------------------------------------------------------------
    # Dump
    
    def dump(self, space=""):
        
        name = TYPES[self.type]
        if self.v is not None:
            name = f"({self.v})"
        
        print(space + f"type: {name}^{self.e} -> {self}")
        if len(self.xes) > 0:
            for i, xe in enumerate(self.xes):
                print(space + "> x^e", i)
                xe.dump(space + "   ")
            print()
            
    # ---------------------------------------------------------------------------
    # Extract the variables
    
    def get_vars(self, xvars=None):
        
        if xvars is None: xvars = XVars()
        
        if self.type == VAR:
            xvars.add_variable(self.v)
        else:
            for xe in self.xes:
                xe.get_vars(xvars)
                
        return xvars
    
    # ---------------------------------------------------------------------------
    # Variables to template
    
    @property
    def is_template(self):
        if self.type == VAR:
            return self.v[0] == '$'
        else:
            for xe in self.xes:
                if xe.is_template:
                    return True
            return False
    
    def to_template(self):
        if self.type == VAR:
            if self.v[0] != '$':
                self.v = '$' + self.v
        else:
            for xe in self.xes:
                xe.vars_to_template()

        return self
    
    # ---------------------------------------------------------------------------
    # Template to Xe
    
    def set_vars(self, xvars):
        
        clone = self.clone()
        
        if clone.type == VAR:
            x = xvars.get(clone.v)
            if x is None:
                raise RuntimeError(f"set_vars error: var name '{self.v}' not found in {xvars} to set vars in '{self}'")
            
            clone.x = x
                
        else:
            for i in range(len(clone.xes)):
                clone.xes[i] = clone.xes[i].set_vars(xvars)
                
        return clone
                
    # ---------------------------------------------------------------------------
    # Evaluate
    
    def compute(self, xvars=None):
        if xvars is None:
            xvars = self.get_vars()
            xvars.set_values()

        clone = self.set_vars(xvars)
        
        return eval(clone.python, None, xvars)
    
    # ---------------------------------------------------------------------------
    # Compare two computations
    
    def check(self, other, seed=None):
        
        if not CHECK:
            return self
        
        xvars = self.get_vars()
        other.get_vars(xvars)
        xvars.set_values(seed=seed)
            
        v0 = self.compute(xvars)
        v1 = other.compute(xvars)
        
        if np.isnan(v0) or np.isnan(v1):
            return self
        
        if abs(v0 - v1) < 1e-5:
            return self
        
        print('-'*80)
        print("Error when comparing two expressions")
        print("1>", self, '-->', self.python)
        print("2>", other, '-->', other.python)
        print("v>", xvars)
        print(f"{v0} != {v1} , diff = {abs(v0 - v1)}")
        print()
        raise RuntimeError("Comparison error")

    # ---------------------------------------------------------------------------
    # To string
    
    def to_str(self, source_code=False):
        
        if source_code:
            f_pref = "np."
            EXP    = "**"
            MUL0   = "*"
            MUL1   = "*"
        else:
            f_pref = ""
            EXP    = "^"
            MUL0   = ""
            MUL1   = "."
            
        # ---------------------------------------------------------------------------
        # Simple cases
        
        if self.type == VALUE:
            return f"{self.v}"
        
        if self.e == 0:
            return "1"

        # ---------------------------------------------------------------------------
        # x to str
        
        if self.type == VAR:
            s = self.v
            
        elif self.type == FUNC:
            
            # special cases
            
            asin = self.v == 'arcsin' and source_code
            atan = self.v == 'arctan' and source_code and self.axes[0].atype == PROD
            
            # Arctan in arctan2
            
            if atan:
                arg = self.xes[0].reduced()
                
                s = "np.arctan2("
                num = Xe.One()
                den = Xe.One()
                for xe in arg.axes:
                    if Xe.e < 0:
                        den = den/xe
                    else:
                        num = num*xe
                        
                s += num.to_str(True) + ", " + den.to_str(True) + ")" 
                
            # Standard cases
                
            else:
            
                s = f_pref + self.v + "("
                if asin:
                    s += "np.clip("
                
                for i, xe in enumerate(self.xes):
                    if i != 0:
                        s += ", "
                    s += xe.to_str(source_code)
                
                if asin:
                    s += ", -1, 1)"
                
                s += ")"

        elif self.type == SUM:
            s = ""
            for i, xe in enumerate(self.xes):
                sx = xe.to_str(source_code)
                if xe.type == SUM and xe.e != 1:
                    sx = '(' + sx + ')'
                    
                if i == 0:
                    s = sx
                else:
                    if sx[0] == "-":
                        s += " - " + sx[1:]
                    else:
                        s += " + " + sx

        elif self.type == PROD:
            
            num = []
            den = []
            for xe in self.xes:
                if xe.e < 0:
                    den.append(xe)
                else:
                    num.append(xe)
            num.extend(den)
            
            s = ""
            for i, xe in enumerate(num):
                
                # ---------------------------------------------------------------------------
                # x to string
                
                sx = xe.x.to_str(source_code)
                if xe.type == SUM or (xe.type == PROD and xe.e != 1) or (xe.type != VALUE and sx[0] == '-'):
                    sx = '(' + sx + ')'
                    
                # ---------------------------------------------------------------------------
                # Exponent part = x^e
                
                if abs(xe.e) != 1:
                    if xe.e.denominator == 1:
                        sx += EXP + f"{abs(xe.e)}"
                    else:
                        sx += EXP + f"({abs(xe.e)})"
                    
                # ---------------------------------------------------------------------------
                # chain of terms x^e

                oper = '/' if xe.e < 0 else MUL0
                if i == 0:
                    if oper == '/':
                        s = '1/' + sx
                    else:
                        if len(self.xes) > 1 and sx == '-1' and self.xes[1].e > 0:
                            s = '-'
                        else:
                            s = sx
                else:
                    if oper != '/':
                        oper = MUL1 if s != '-' else ""
                    
                    s += oper + sx
                            
        # ---------------------------------------------------------------------------
        # Quick
        
        if self.e == 1:
            return s
                            
        # ---------------------------------------------------------------------------
        # Exponent
        
        if s[0] == '-' or self.type in [PROD, SUM]:
            s = '(' + s + ')'
            
        if self.e < 0:
            s = "1/" + s
            
        if self.e != -1:
            if abs(self.e).denominator == 1:
                s = s + f"{EXP}{abs(self.e)}"
            else:
                s = s + f"{EXP}({abs(self.e)})"
            
        return s
    
    # ---------------------------------------------------------------------------
    # Representation

    def __repr__(self):
        return self.to_str(source_code=False)
    
    @property
    def python(self):
        return self.to_str(source_code=True)
    
    # ---------------------------------------------------------------------------
    # Development
    
    def developed(self):
        return self.clone().develop()
    
    def develop(self):
        
        check_ = self.clone()
        
        # ---------------------------------------------------------------------------
        # Develop two lists of terms
        
        def dev(sum0, sum1):
            res = []
            for s0 in sum0:
                for s1 in sum1:
                    res.append(s0*s1)
            return res
                    
        # ---------------------------------------------------------------------------
        # Nothing to do
        
        if self.type in [VALUE, VAR, FUNC]:
            return self
        
        # ---------------------------------------------------------------------------
        # Develop the terms
        
        for xe in self.xes:
            if xe.type in [SUM, PROD]:
                xe.develop()
                
        self.reduce()
        
        # ---------------------------------------------------------------------------
        # Develop exponent for sums
        
        if self.type == SUM:
            n = abs(self.e.numerator)
            if n > 1:
                xes = list(self.xes)
                for i in range(1, n):
                    self.xes = dev(self.xes, xes)
                self.e = 1
                self.reduce()
                
            return self.check(check_)
        
        # ---------------------------------------------------------------------------
        # Develop the product

        factors  = []
        the_sums = {}
        
        for xe in self.xes:
            if xe.type == SUM:
                den = xe.e.denominator * (-1 if xe.e < 0 else 1)
                sm  = the_sums.get(den)
                if sm is None:
                    sm = [1]
                    
                n = abs(xe.e.numerator)
                for i in range(n):
                    sm = dev(sm, xe.xes)
                    
                the_sums[den] = sm
                
            else:
                factors.append(xe)
                    
        # ---------------------------------------------------------------------------
        # Develop the factor to den = 1
        
        if len(factors) > 0:
            sm = the_sums.get(1)
            if sm is not None:
                for i in range(len(sm)):
                    fsm = Xe(PROD, *factors, reduce=False)
                    fsm.xes.append(sm[i])
                    sm[i] = fsm.reduce()
                factors = []
            
        # ---------------------------------------------------------------------------
        # Result
        
        if len(factors) == 0:
            res = None
        else:
            res = Xe(PROD, *factors, reduce=False)
            
        for den, sm in the_sums.items():
            if res is None:
                res = Xe(SUM, *sm, e=Fraction(1, den))
            else:
                res *= Xe(SUM, *sm, e=Fraction(1, den))
                
        if res is None:
            res = Xe.Zero()
        else:
            res.reduce()

        return self.change_to(res).check(check_)
    
    
    # ---------------------------------------------------------------------------
    # Compare two arrays of axes
    
    @staticmethod
    def comp_xes(xes0, xes1, sort=True):
        
        if len(xes0) > len(xes1):
            return -1
        elif len(xes0) < len(xes1):
            return 1

        # ----- Arrays have the same size
        
        if sort:
            ax0 = [xe.clone() for xe in xes0]
            ax1 = [xe.clone() for xe in xes1]
            ax0.sort()
            ax1.sort()
        else:
            ax0 = xes0
            ax1 = xes1
        
        for x0, x1 in zip(ax0, ax1):
            cmp = Xe.compare(x0, x1)
            if cmp != 0:
                return cmp
            
        return 0   
    
    
    # ---------------------------------------------------------------------------
    # Compare two axes
    
    @staticmethod
    def compare(xe0, xe1):
        
        s0 = xe0.str_sort
        s1 = xe1.str_sort
        
        if s0 < s1:
            return -1
        elif s0 > s1:
            return 1
        else:
            return 0
          
        # ---------------------------------------------------------------------------
        # Comparizon with developed expressions
        
        xe0 = xe0.developed()
        xe1 = xe1.developed()
        
        # ---------------------------------------------------------------------------
        # Types
        
        if xe0.type < xe1.type:
            return -1
        if xe0.type > xe1.type:
            return 1
        
        # ---------------------------------------------------------------------------
        # Values
        
        if xe0.type in [VALUE, VAR, FUNC]:
            if xe0.v < xe1.v:
                return -1
            if xe0.v > xe1.v:
                return 1
            
            if xe0.type == VALUE:
                return 0
            
            if xe0.type == FUNC:
                cmp = Xe.comp_axes(xe0.xes, xe1.xes, sort=False)
                if cmp != 0:
                    return cmp
                
        # ---------------------------------------------------------------------------
        # Sums or products
            
        elif xe0.type in [SUM, PROD]:
            cmp = Xe.comp_axes(xe0.xes, xe1.xes, sort=True)
            if cmp != 0:
                return cmp
            
        # ---------------------------------------------------------------------------
        # Exponent and factor to compare
        
        if xe0.e < xe1.e:
            return -1
        if xe0.e > xe1.e:
            return 1
        
        # ---------------------------------------------------------------------------
        # Equal !!!!!
        
        return 0
    
    # ---------------------------------------------------------------------------
    # Comparison
    
    def __eq__(self, other):
        return Xe.compare(self, Xe.xe(other)) == 0
    
    def __lt__(self, other):
        return Xe.compare(self, Xe.xe(other)) < 0
    
    def __let__(self, other):
        return Xe.compare(self, Xe.xe(other)) <= 0
    
    def __gt__(self, other):
        return Xe.compare(self, Xe.xe(other)) > 0
    
    def __get__(self, other):
        return Xe.compare(self, Xe.xe(other)) >= 0
    
    # ---------------------------------------------------------------------------
    # Signs
    
    def __pos__(self):
        return self.clone()
    
    def __neg__(self):
        return -1 * self
    
    # ---------------------------------------------------------------------------
    # Operations
    
    def __add__(self, other):
        o = Xe.xe(other, True)
        r = self.clone()
        
        if r.type == SUM and r.e == 1:
            if o.type == SUM and o.e == 1:
                r.xes.extend(o.xes)
                return r.reduce()
            else:
                r.xes.append(o)
                return r
            
        elif o.type == SUM and o.e == 1:
            o.xes.insert(0, r)
            return o
        
        return Xe(SUM, r, o)
        
    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        o = Xe.xe(other).clone()
        r = self.reduced()
        
        if r.type == PROD:
            r.xes.append(o)
            return r.reduce()

        elif o.type == PROD:
            o.xes.append(r)
            return o.reduce()
        
        return Xe(PROD, r, o)
        
    def __truediv__(self, other):
        return self * (other ** (-1))
    
    def __pow__(self, exp):
        
        r = self.clone()
        
        if r.type == VALUE:
            r.v **= exp
            return r
        
        else:
            r.e *= exp
            if r.type == PROD:
                r.reduce()
                
            return r    
        
    # ---------------------------------------------------------------------------
    # Reverse operations
    
    def __radd__(self, other):
        return Xe.xe(other) + self
        
    def __rsub__(self, other):
        return Xe.xe(other) - self
        
    def __rmul__(self, other):
        return Xe.xe(other) * self
        
    def __rtruediv__(self, other):
        return Xe.xe(other) / self
    
    # ---------------------------------------------------------------------------
    # I operations
    
    def __iadd__(self, other):
        return self.change_to(self + other)
        
    def __isub__(self, other):
        return self.change_to(self + (-other))
        
    def __imul__(self, other):
        return self.change_to(self * other)
        
    def __itruediv__(self, other):
        return self.change_to(self * (other**-1))
    
    def __ipow__(self, exp):
        return self.change_to(self ** exp)
    
    # ---------------------------------------------------------------------------
    # To product
    # Can be inverted with compact
        
    def to_product(self):
        
        if self.type == PROD:
            return self.clone()
        else:
            return Xe(PROD, self, reduce=False)
        
    # ---------------------------------------------------------------------------
    # Extract an expression from a product
    # The expression is treated as a product
    # Return None is expr is not found in the product
    #
    # Product           Expression          Result
    # a.b               a                   b
    # a.b^2             a                   b^2
    # a.b^2             b                   None if exact else b
    # a/b               b                   None if exact else a/b^2 
    # a                 b                   None
    # a/b^4             1/b^2               None if exact else a/b^2
    
    def prod_extract(self, expr, exact=True):
        
        prod = self.to_product()
        xp   = expr.to_product()
        
        vprod = 1
        
        for xp_xe in xp.xes:

            if xp_xe.type == VALUE:
                
                vprod /= xp_xe.v
            
            else:
                found = False
                for i, xe in enumerate(prod.xes):
                    if xp_xe.x == xe.x:
                        same_e = xp_xe.e == Xe.e
                        found = same_e or (not exact)
                        if found:
                            if same_e:
                                del prod.xes[i]
                            else:
                                prod.xes[i].e -= xp_xe.e
                            break
                    
                if not found:
                    return None
                
        self.check(prod * vprod * expr)
            
        return prod * vprod
    
    
    # ---------------------------------------------------------------------------
    # Factorization
    #
    # For an argument expr, transform a sum in:
    #
    # sum = R + F * expr
    #
    # Return the couple R, F
    #
    # Return None, None if self is not a sum
    #
    # ----- expr is None
    #
    # If the expression to factorize is None, find the maximum factorizable
    # expression such as:
    #
    # sum = A * B
    #
    # Return A and B
    #
    # ----- expr is compact
    #
    # If expr is compact (product, simple expression or sum^e), extract it from all
    # the possible terms of the sum
    #
    # ----- expr is a sum
    #
    # Compute R and F such as:
    #
    # sum = R + F * (expr)
    #
    
    def factorize(self, expr):
        
        # ---------------------------------------------------------------------------
        # Make sure with work with true sums
        
        if self.type != SUM or len(self.xes) < 2:
            return None, None
        
        # ---------------------------------------------------------------------------
        # Expression is None : we factorize by the common factors
        
        if expr is None:
            
            rem_sum = self.clone()
            factor  = 1
            
            if self.xes[0].type == PROD:
                xes = self.xes[0].xes
            else:
                xes = [self.xes[0]]
                
            for xe in xes:
                r, f = rem_sum.factorize(xe)
                if r == 0:
                    factor = factor * xe
                    for i in range(len(rem_sum.xes)):
                        rem_sum.xes[i] = rem_sum.xes[i] / xe
                        
            if factor == 1:
                return 1, self.clone()
            else:
                factor = factor ** rem_sum.e
                (factor * rem_sum).check(self)
                return factor, rem_sum
        
        # ---------------------------------------------------------------------------
        # Factorizing a sum

        if expr.type == SUM and expr.e == 1:
            
            # ----- Pre work with match str
            
            self_sm = [xe.str_match for xe in self.xes]
            expr_sm = [xe.str_match for xe in expr.xes]

            if len(expr_sm) > len(self_sm):
                return None, None
            
            # ----- Let's take the longest term
            
            i_max = 0
            for i, s in enumerate(expr.xes):
                if len(expr_sm[i]) > len(expr_sm[i_max]):
                    i_max = i
                    
            # ----- Loop on the terms of the sum to find the term matching the
            # longest term in the expression
            
            for i_xe, sum_xe in enumerate(self.xes):
                
                # ----- Can this term be the first one of the expression
                
                if expr_sm[i_max] not in self_sm[i_xe]:
                    continue
                
                # ----- By dividing by the first term of the expression
                # we have a possible common factor
                
                factor = sum_xe / expr.xes[i_max]
                
                # ----- Divide the other terms by the factor
                
                xes = []
                for i, ax in enumerate(self.xes):
                    if i != i_xe:
                        xes.append(ax/factor)

                # ----- Check if the expression contains the last terms of the expression
                
                found = False
                for i_term, term in enumerate(expr.xes):
                    
                    if i_term == i_max:
                        continue
                    
                    v_term, xe_term = term.value_xe()
                    
                    found = False
                    i = 0
                    while i < len(xes):
                        
                        v_i, xe_i = xes[i].value_xe()
                        
                        if xe_term is None:
                            ok = xe_i is None
                        else:
                            if xe_i is None:
                                ok = False
                            else:
                                ok = xe_term == xe_i
                                
                        if ok:
                            found = True
                            if v_term == v_i:
                                del xes[i]
                            else:
                                if xe_term is None:
                                    xes[i] = v_i - v_term
                                else:
                                    xes[i] = (v_i - v_term) * xe_term
                            break
                        else:
                            i += 1
                        
                    if not found:
                        break
                    
                # ----- Found : we have our factorizion
                
                if found:
                    
                    rem_xes = [xe * factor for xe in xes]
                    
                    if len(rem_xes) == 0:
                        rem = 0
                    else:
                        rem = Xe(SUM, *rem_xes)
                        
                    if VERBOSE:
                        print(f"Factorization of '{self}' by '{expr}'")
                        print(f"     xes:    {xes}")
                        print(f"     rem:    {rem}")
                        print(f"     factor: {factor}")
                    
                    if CHECK:
                        test = rem + factor*expr
                        try:
                            self.x.check(test)
                        except:
                            test = (rem + factor*expr)
                            print('-'*80)
                            print("Error in factorization")                    
                            print("Factorization of", self)
                            print("len axes, expr  ", len(self.xes), len(expr.xes))
                            print("By              ", expr)
                            print("Factor          ", factor)
                            print("rem_axes        ", rem_xes)
                            print("Rem             ", rem)
                            print("rem + f*expr    ", test)
                            print("Reduced         ", test.reduced())
                            
                            raise RuntimeError("Error in factorization")
                            
                    # Factorization of remaining :-)
                    
                    r, f = Xe.xe(rem).factorize(expr)
                    
                    if r is not None:
                        return r, factor + f
                    else:
                        return rem, factor
                
            return None, None
                    
        # ---------------------------------------------------------------------------
        # We try to extract a compact expression from a sum
        # Note that the compact expression can be a sum with an exponent
                
        else:
            expr_out = Xe.Zero()
            expr_in  = Xe.Zero()
            
            for xe in self.xes:
                factor = xe.prod_extract(expr, False)
                if factor is None:
                    expr_out = expr_out + xe
                else:
                    expr_in = expr_in + factor
                    
            if len(expr_in.xes) == 0:
                return None, None
            else:
                self.check(expr_out + expr_in*expr)
                return expr_out, expr_in
        
        return None, None
        
    # ----------------------------------------------------------------------------------------------------
    # Return an iterator partially matching a compact template
    
    def compact_matches(self, template, xvars):
        return TCompactIter(template, self, xvars)
    
    
    # ---------------------------------------------------------------------------
    # Apply a template
    
    def apply_template(self, template, target):

        check_ = self.clone()

        if self.type == VALUE:
            return self

        # ---------------------------------------------------------------------------
        # Apply the templates on the sub expressions
        
        for xe in self.xes:
            xe.apply_template(template, target)
        
        # ---------------------------------------------------------------------------
        # Get the template variables

        xvars = template.get_vars()

        # ---------------------------------------------------------------------------
        # The template is a sum. Only applies on sums with a greater number of terms
        
        if template.type == SUM:
            
            if self.type != SUM:
                return self
            
            if len(template.xes) > len(self.xes):
                return self
            
            # ---------------------------------------------------------------------------
            # Try to match the first term FIRST of the template with a term in the expression
            #
            # expr = rem + factor * FIRST
            #
            # To find a full match, the remaining terms must be in rem
            #
            # rem = something + factor * (LAST TERMS)
            
            first_index = -1
            for i in range(len(template.xes)):
                if template.xes[first_index].is_template:
                    first_index = i
                    break
                
            tmpl_first = template.xes[first_index]
            
            # ---------------------------------------------------------------------------
            # Loop on the terms in the expression
            
            xvars = template.get_vars()
            
            for expr_i, expr_xe in enumerate(self.xes):
                
                # ---------------------------------------------------------------------------
                # For the current term, let's loop on all the possible template matches
                # When found, a partial match return the factor and the xvars
                #
                # expr_xe = factor * first(xvars)
                
                for factor, xvs in expr_xe.compact_matches(tmpl_first, xvars):
                    
                    # ---------------------------------------------------------------------------
                    # Now we have an expression matching with the first template with a certain
                    # factor, we must find if the other terms match with the same factor
                    # By diving the remaining terms by the factor, we should have exact matches
                    
                    exacts = [xe / factor for xe in self.xes]
                    indices = [] 
                    
                    for tmpl_i, tmpl_xe in enumerate(template.xes):
                        
                        if tmpl_i == first_index:
                            indices.append(expr_i)
                            continue
                        
                        found = False
                        for i in range(len(exacts)):
                            
                            # ----- Already matched
                            if i == expr_i or i in indices:
                                continue
                            
                            # ----- Exact match
                            
                            if xvs.template_match(tmpl_xe, exacts[i]):
                                indices.append(i)
                                found = True
                                break
                            
                        if not found:
                            break
                        
                    if not found:
                        continue
                    
                    if not xvs.completed:
                        continue
                    
                    # ------------------------------------------------------------------------------------------
                    # We have an exact match, let's build the result
                    #
                    # expression = factor * MATCH + remainder
                    #
                    # TEMPLATE is the terms in exacts indiced by indices
                    # Remainder is composed of the terms of self not indices
                    #
                    # CAUTION: fact can be different from factor when more than
                    # on factorization occur
                    
                    rem, fact = self.factorize(template.set_vars(xvs))
                    
                    res = rem + fact * target.set_vars(xvs)
                    
                    if TPL_VERBOSE:
                        print(f"\nSUM >>>> Template '{template}' -> '{target}' applied to '{self}':")
                        print(f"   xvs:      {xvs}")
                        print(f"   rem:      {rem}")
                        print(f"   factor:   {factor}")
                        print(f"   template: {template.set_vars(xvs)}")
                        print(f"   target:   {target}")
                        print(f"             {target.set_vars(xvs)}")
                        print(f"   res:      {res}")

                    self.change_to(res)
                    self.check(check_)
                    
                    return self.apply_template(template, target)
                    
        # ---------------------------------------------------------------------------
        # The template is a product
        # If self is a sum, already done one axes. Work only on products
        
        elif template.type == PROD:
            
            if self.type != PROD or len(self.xes) < len(template.xes):
                return self.check(check_)
            
            
            # ---------------------------------------------------------------------------
            # We loop on the possible maches product to product
            # and stop to the first one
            #
            # expr_xe = factor * first(xvars)
            
            for factor, xvs in self.compact_matches(template, xvars):
                
                if not xvs.completed:
                    continue
                
                if TPL_VERBOSE:
                    print(f"\nPROD >>>> Template '{template}' -> '{target}' applied to '{self}':")
                    print(f"   xvs:      {xvs}")
                    print(f"   factor:   {factor}")
                    print(f"             {self / template.set_vars(xvs)}")
                    print(f"   template: {template.set_vars(xvs)}")
                    print(f"   target:   {target}")
                    print(f"             {target.set_vars(xvs)}")
                    print(f"   res:      {factor * target.set_vars(xvs)}")
                
                res = factor * target.set_vars(xvs)
                self.change_to(res)
                self.check(check_)
                
                return self.apply_template(template, target)
            
        # ---------------------------------------------------------------------------
        # Return

        return self

# ----------------------------------------------------------------------------------------------------
# Quaternion

class Quat:
    def __init__(self, *args):
        self.q = [Xe.Zero() for i in range(4)] 
        for i, arg in enumerate(args):
            self.q[i] = Xe.xe(arg)
        
    @classmethod
    def Rotation(cls, axis, short=True):
        
        q = cls()
        a = 'abc'[axis]
        
        if short:
            q.w           = Xe.Var(f"c{a}")
            q.q[1 + axis] = Xe.Var(f"s{a}")
        else:
            var = Xe.Var(a)
            q.w           = Xe.Func("cos", var)
            q.q[1 + axis] = Xe.Func("sin", var)
            
        
        return q
        
    @property
    def w(self):
        return self.q[0]
    
    @w.setter
    def w(self, v):
        self.q[0] = Xe.xe(v)
    
    @property
    def x(self):
        return self.q[1]

    @x.setter
    def x(self, v):
        self.q[1] = Xe.xe(v)
    
    
    @property
    def y(self):
        return self.q[2]

    @y.setter
    def y(self, v):
        self.q[2] = Xe.xe(v)
    
    
    @property
    def z(self):
        return self.q[3]
    
    @z.setter
    def z(self, v):
        self.q[3] = Xe.xe(v)
        
    def reduce(self):
        for i in range(4):
            self.q[i].reduce()
        return self
    
    
    def to_str(self, names='wxyz', source_code=False):

        s = ""
        for i in range(4):
            s += f"{names[i]} = " + self.q[i].to_str(source_code) + "\n"
            
        return s
    
    def __repr__(self):
        return self.to_str()
    
    @property
    def python(self):
        names = [f"q.a[..., {i}]" for i in range(4)]
        return self.to_str(names, source_code=True)
    
    def __neg__(self):
        p = Quat()
        for i in range(4):
            p.q[i] = -self.q[i]
    
    def __add__(self, other):
        p = Quat()
        if type(other) == Quat:
            for i in range(4):
                p.q[i] = self.q[i] + other.q[i]
        else:    
            for i in range(4):
                p.q[i] = self.q[i] + other
                
    def __sub__(self, other):
        return self + (-other)
    
    def __matmul__(self, other):
        p = Quat()
        
        a = self.w
        b = self.x
        c = self.y
        d = self.z
        
        e = other.w
        f = other.x
        g = other.y
        h = other.z
        
        p.w = a*e - b*f - c*g - d*h
        p.x = a*f + b*e + c*h - d*g
        p.y = a*g - b*h + c*e + d*f
        p.z = a*h + b*g - c*f + d*e
        
        return p.reduce()
    
    @property
    def conjugate(self):
        c = Quat()
        c.q[0] = self.q[0].clone()
        c.q[1] = -self.q[1]
        c.q[2] = -self.q[2]
        c.q[3] = -self.q[3]

        return c
    
    def rotate(self, q):
        
        c = self.conjugate
        
        
        return self @ q @ c
        
    
    @classmethod
    def FromEulers(cls, order='XYZ', short=True):
        
        qx = Quat.Rotation(0, short)
        qy = Quat.Rotation(1, short)
        qz = Quat.Rotation(2, short)
    
        if order == 'XYZ':
            return qz @ qy @ qx
        elif order == 'XZY':
            return qy @ qz @ qx
        elif order == 'YXZ':
            return qz @ qx @ qy
        elif order == 'YZX':
            return qx @ qz @ qy
        elif order == 'ZXY':
            return qy @ qx @ qz
        elif order == 'ZYX':
            return qx @ qy @ qz
        
    @staticmethod
    def python_euler_to_quat(space="    "*3):
        
        if_token = "if"
        
        for order in ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']:
            print()
            print(f"{space}# Order {order}: q{order[2].lower()} @ q{order[1].lower()} @ q{order[0].lower()}")
            print()
            print(f"{space}{if_token} self.order == '{order}':")
            
            q = Quat.FromEulers(order)
            lines = q.python.split("\n")
            for line in lines:
                print(f"{space}    {line}")
                
            if_token = "elif"
            
            break
        
    @staticmethod
    def python_quat_to_euler_OLD(space="    "*3):
        
        X, Y = Xe.Vars("$x", "$y")
        
        tpl_cos2sin2 = Xe.cos(X)**2 + Xe.sin(X)**2

        tpl_sincos   = Xe.sin(X)*Xe.cos(X)
        tgt_sincos   = (Xe.sin(2*X))/2

        tpl_1sin2    = 1 - 2*Xe.sin(X)**2
        tgt_1sin2    = Xe.cos(2*X)

        tpl_coscos   = -2*Xe.cos(X)**2*Xe.sin(Y)**2 - 2*Xe.cos(Y)**2*Xe.sin(X)**2 + 1
        tgt_coscos   = Xe.cos(2*X)*Xe.cos(2*Y)
        

        for order in ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']:
            
            print()
            print("order", order)
            
            if order == 'XYZ':
                inds0  = [0, 2, 1, 3]
                op0    = -1
                solve0 = 'b'
                
            elif order == 'XZY':
                inds0  = [0, 3, 1, 2]
                op0    = 1
                solve0 = 'c'
            
            elif order == 'YXZ':
                inds0  = [0, 1, 2, 3]
                op0    = 1
                solve0 = 'a'
            
            elif order == 'YZX':
                inds0  = [0, 3, 1, 2]
                op0    = -1
                solve0 = 'c'
            
            elif order == 'ZXY':
                inds0  = [0, 1, 2, 3]
                op0    = -1
                solve0 = 'a'
            
            elif order == 'ZYX':
                inds0  = [0, 2, 1, 3]
                op0    = 1
                solve0 = 'b'

            q = Quat.FromEulers(order, short=False)
            
            eqs = [
                Equality('w', q.w),
                Equality('x', q.x),
                Equality('y', q.y),
                Equality('z', q.z),
                ]
            
            if True:
                for eq in eqs:
                    print(eq)
                print()
            
            # ---------------------------------------------------------------------------
            # Sin
            
            if False:
            
                comb0 = eqs[inds0[0]] * eqs[inds0[1]]
                if op0 > 0:
                    comb0 = comb0 + eqs[inds0[2]] * eqs[inds0[3]]
                else:
                    comb0 = comb0 - eqs[inds0[2]] * eqs[inds0[3]]
                    
                comb0.develop().reduce()
                    
                comb0.right.reduce().apply_template(tpl_cos2sin2, Xe.One())
                comb0.right.apply_template(tpl_sincos, tgt_sincos).reduce()
                
                
                comb0 = ~comb0
                print(comb0)
                
                print(comb0.solve(solve0))
                
            # ---------------------------------------------------------------------------
            # Arctan - Numerators
            
            if False:
                
                prods = []
                for i in range(3):
                    for j in range(i+1, 4):
                        eq = (eqs[i]*eqs[j]).develop().reduce()
                        prods.append(eq)
                        
                if False:
                    for i, eq in enumerate(prods):
                        print(i, ">", eq)
                    print()
    
                nums = []
                for i in range(3):
                    
                    rem = [5, 4, 3]
                    num = prods[i] + prods[rem[i]]
                    num.right.reduce()
                    num.right = num.right.factorize(None)
                    
                    num.right.reduce().apply_template(tpl_cos2sin2, Xe.One())
                    num.right.apply_template(tpl_sincos, tgt_sincos).reduce()
                    num.right.apply_template(tpl_1sin2, tgt_1sin2).reduce()
                    
                    nums.append(num)
                    
                if True:
                    for eq in nums:
                        print(eq, '-->', eq.right.count)
                    print()
                
                
            # ---------------------------------------------------------------------------
            # Arctan - Denominators
            
            if True:

                sqs = []
                for i in range(1, 4):
                    eq = (eqs[i]*eqs[i]).develop().reduce()
                    sqs.append(eq)
                    
                if False:
                    for eq in sqs:
                        print(eq)
                    print()
                    
                dens = []
                for i in range(2):
                    for j in range(i+1, 3):

                        den = -(sqs[i] + sqs[j])*2 + 1
                        den.develop().reduce()
                        den.right = den.right.factorize(None)
                        
                        den.right.apply_template(tpl_cos2sin2.clone(), Xe.One())
                        den.right.apply_template(tpl_coscos,   tgt_coscos)
                        
                        dens.append(den)
                        
                if True:
                    for i, eq in enumerate(dens):
                        print(i, '>', eq, '-->', eq.right.count)
                    print()

                
            
            return
        
    @staticmethod
    def python_quat_to_euler(space="    "*3):
        
        X, Y = Xe.Vars("$x", "$y")
        
        tpl_cos2sin2 = Xe.cos(X)**2 + Xe.sin(X)**2

        tpl_sincos   = Xe.sin(X)*Xe.cos(X)
        tgt_sincos   = (Xe.sin(2*X))/2

        tpl_1sin2    = 1 - 2*Xe.sin(X)**2
        tgt_1sin2    = Xe.cos(2*X)

        tpl_coscos   = -2*Xe.cos(X)**2*Xe.sin(Y)**2 - 2*Xe.cos(Y)**2*Xe.sin(X)**2 + 1
        tgt_coscos   = Xe.cos(2*X)*Xe.cos(2*Y)
        
        tpl_tan      = Xe.sin(X)/Xe.cos(X)
        tgt_tan      = Xe.tan(X)
        
        
        tab   = " "*4

        for order in ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']:
            
            if order != 'ZYX':
                pass
                #continue
                
            if order == 'XYZ':
                inds0  = [0, 2, 1, 3]
                op0    = -1
                solve0 = 'b'

                nums_i = [0, 2]
                dens_i = [0, 2]
                atan0  = 'a'
                atan1  = 'c'
                
                
            elif order == 'XZY':
                inds0  = [0, 3, 1, 2]
                op0    = 1
                solve0 = 'c'

                nums_i = [0, 1]
                dens_i = [1, 2]
                atan0  = 'a'
                atan1  = 'b'
            
            elif order == 'YXZ':
                inds0  = [0, 1, 2, 3]
                op0    = 1
                solve0 = 'a'

                nums_i = [1, 2]
                dens_i = [0, 1]
                atan0  = 'b'
                atan1  = 'c'
            
            elif order == 'YZX':
                inds0  = [0, 3, 1, 2]
                op0    = -1
                solve0 = 'c'

                nums_i = [0, 1]
                dens_i = [1, 2]
                atan0  = 'a'
                atan1  = 'b'
            
            elif order == 'ZXY':
                inds0  = [0, 1, 2, 3]
                op0    = -1
                solve0 = 'a'

                nums_i = [1, 2]
                dens_i = [0, 1]
                atan0  = 'b'
                atan1  = 'c'
            
            elif order == 'ZYX':
                inds0  = [0, 2, 1, 3]
                op0    = 1
                solve0 = 'b'

                nums_i = [0, 2]
                dens_i = [0, 2]
                atan0  = 'a'
                atan1  = 'c'
                

            # ---------------------------------------------------------------------------
            # Left
            
            left_euler = {
                'a': "eulers.a[..., 0]",
                'b': "eulers.a[..., 1]",
                'c': "eulers.a[..., 2]",
                }
            
            right_euler = {
                'a': None,
                'b': None,
                'c': None,
                }
                
            # ---------------------------------------------------------------------------
            # Compute the quaternions from the order

            q = Quat.FromEulers(order, short=False)
            
            eqs = [
                Equality('w', q.w),
                Equality('x', q.x),
                Equality('y', q.y),
                Equality('z', q.z),
                ]
            
            base = []
            for eq in eqs:
                base.append(f"# {eq}")
            
            if False:
                for eq in eqs:
                    print(eq)
                print()
            
            # ---------------------------------------------------------------------------
            # Sin
            
            if True:
            
                comb0 = eqs[inds0[0]] * eqs[inds0[1]]
                
                comb0 = comb0 + (eqs[inds0[2]] * eqs[inds0[3]])*op0
                    
                comb0.develop().reduce()
                    
                comb0.right.reduce().apply_template(tpl_cos2sin2, Xe.One())
                comb0.right.apply_template(tpl_sincos, tgt_sincos).reduce()
                
                
                comb0 = ~comb0
                comb0 = comb0.solve(solve0)
                comb0.right = 2*comb0.right
                
                right_euler[solve0] = comb0.right.python
                
                
            # ---------------------------------------------------------------------------
            # Arctan - Numerators
            
            if True:
                
                prods = []
                for i in range(3):
                    for j in range(i+1, 4):
                        eq = (eqs[i]*eqs[j]).develop().reduce()
                        prods.append(eq)
                        
                if False:
                    for i, eq in enumerate(prods):
                        print(i, ">", eq)
                    print()
    
                nums = []
                num0 = None
                num1 = None
                for i in range(3):
                    
                    if i not in nums_i:
                        continue
                    
                    rem = [5, 4, 3]
                    num = prods[i] - prods[rem[i]]*op0
                    num.right.reduce()
                    a, b = num.right.factorize(None)
                    num.right = a*b
                    
                    num.right.reduce().apply_template(tpl_cos2sin2, Xe.One())
                    num.right.apply_template(tpl_sincos, tgt_sincos).reduce()
                    num.right.apply_template(tpl_1sin2, tgt_1sin2).reduce()
                    
                    if num0 is None:
                        num0 = num
                    else:
                        num1 = num
                    
                    nums.append(num)
                    
                if False:
                    for eq in nums:
                        print(eq)
                    print()
                
                
            # ---------------------------------------------------------------------------
            # Arctan - Denominators
            
            if True:

                sqs = []
                for i in range(1, 4):
                    eq = (eqs[i]*eqs[i]).develop().reduce()
                    sqs.append(eq)
                    
                if False:
                    for eq in sqs:
                        print(eq)
                    print()
                    
                dens = []
                i_den = -1
                den0 = None
                den1 = None
                for i in range(2):
                    for j in range(i+1, 3):
                        
                        i_den += 1
                        if i_den not in dens_i:
                            pass
                            continue

                        den = -(sqs[i] + sqs[j])*2 + 1
                        den.develop().reduce()
                        a, b = den.right.factorize(None)
                        den.right = a * b
                        
                        den.right.apply_template(tpl_cos2sin2.clone(), Xe.One())
                        den.right.apply_template(tpl_coscos,   tgt_coscos)
                        
                        if den0 is None:
                            den0 = den
                        else:
                            den1 = den
                            
                        dens.append(den)
                        
                if False:
                    for i, eq in enumerate(dens):
                        print(i, '>', eq)
                    print()
                    
                    
            # ---------------------------------------------------------------------------
            # The formulas
            
            def atan_str(num, den):
                return f"np.arctan2(2*({num.left.python}), {den.left.python})"
                
                
                d = (1 - den.left)/2
                return f"np.arctan2(2*({num.left.python}), 1 - 2*({d.python}))"
            
            right_euler[atan0] = atan_str(num0, den0)
            right_euler[atan1] = atan_str(num1, den1)

            print()
            space = tab*2
            sif = "if" if order == 'XYZ' else "elif"
            print(f"{space}{sif} order == '{order}':")
            print()
            
            space = tab*3

            for line in base:
                print(f"{space}{line}")
            print()

            for v in ['a', 'b', 'c']:
                print(f"{space}{left_euler[v]} = {right_euler[v]}")
            print()
 
    
            
# ----------------------------------------------------------------------------------------------------
# Equality between a left and a right expression
    
class Equality:
    
    def __init__(self, left, right):
        self.left  = Xe.xe(left).reduced()
        self.right = Xe.xe(right).reduced()
        
    def clone(self):
        return Equality(self.left.clone(), self.right.clone())
        
    def to_str(self, source_code=False):
        return f"{self.left} = {self.right}"
    
    def __repr__(self):
        return self.to_str(False)
    
    def __python__(self):
        return self.to_str(True)
    
    def develop(self):
        self.left.develop().reduce()
        self.right.develop().reduce()
        return self
    
    def reduce(self):
        self.left.reduce()
        self.right.reduce()
        return self
    
    def __invert__(self):
        return Equality(self.right, self.left)
    
    def __neg__(self):
        return Equality(-self.left, -self.right)
    
    def __add__(self, other):
        if type(other) is Equality:
            return Equality(self.left + other.left, self.right + other.right)
        else:
            return Equality(self.left + other, self.right + other)
        
    def __sub__(self, other):
        if type(other) is Equality:
            return Equality(self.left - other.left, self.right - other.right)
        else:
            return Equality(self.left - other, self.right - other)
    
    def __mul__(self, other):
        if type(other) is Equality:
            return Equality(self.left * other.left, self.right * other.right)
        else:
            return Equality(self.left * other, self.right * other)
    
    def __truediv__(self, other):
        if type(other) is Equality:
            return Equality(self.left / other.left, self.right / other.right)
        else:
            return Equality(self.left / other, self.right / other)
        
    def __pow__(self, other):
        return Equality(self.left ** other, self.right ** other)
        
        
        
    def solve(self, var_name):
        count = self.left.contains(var_name)
        if count != 1:
            print(self)
            raise RuntimeError(f"Impossible de solve the equality for '{var_name}': {count} found!")
            
        
        for _ in range(100):
            
            left = self.left

            if left.e != 1:
                self = self ** -left.e
                
            if left.type == VAR and left.v == var_name:
                return self
                
            loop = False
            if left.type == SUM:
                for xe in left.xes:
                    if xe.contains(var_name) == 0:
                        self = self - xe
                        loop = True
                        continue
            if loop:
                continue
                    
            loop = False
            if self.left.type == PROD:
                for xe in self.left.xes:
                    if xe.contains(var_name) == 0:
                        self = self / xe
                        loop = True
                        continue
                    
            if loop:
                continue
                    
            if left.type == FUNC:
                inv = left.func_inv
                self.left = left.xes[0]
                self.right = Xe.Func(inv, self.right)
                continue
                    
        return self
    
#VERBOSE = False
#TPL_VERBOSE = False    

Quat.python_quat_to_euler(space="")

rot = Quat(*['qw', 'qx', 'qy', 'qz'])
q = Quat(*[0, 'x', 'y', 'z'])
x, y, z = Xe.Vars('x', 'y', 'z')

p = rot.rotate(q)
print(p)

for i in range(1, 4):
    a = p.q[i]
    
    a = a.develop()
    #print(i, ">", a)
    rx, fx = a.factorize(x)
    #print(rx, fx)
    ry, fy = rx.factorize(y)
    rz, fz = ry.factorize(z)
    
    a = x*fx + y*fy + z*fz
    
    print(['rw', 'rx', 'ry', 'rz'][i], "=", a.python)
    
print()




