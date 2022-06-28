# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import time
import numpy as np
from fractions import Fraction

verbose     = False
tpl_verbose = False
CHECK       = True

DEBUG = 0


# ----------------------------------------------------------------------------------------------------
# Token

VALUE = 0
VAR   = 1
FUNC  = 2
PROD  = 3
SUM   = 4

ATYPES = ['value', 'var', 'f', 'product', 'sum']

LETTERS = 'abcdefghijklmnopqrstuvwxyz_'
FIGURES = '0123456789'

FUNCTIONS = {
    'sin': 'arcsin',
    'cos': 'arccos',
    'tan': 'arctan',
    }

# ----------------------------------------------------------------------------------------------------
# A token a.x^e


class Axe:
    
    def __init__(self, atype, *axes, a=1, e=1, name=None):
        
        self.axes   = []
        self.a_     = Fraction(0)
        self.e_     = Fraction(1)
        
        if a == 0:
            self.atype = VALUE
            self.a     = 0
            self.e     = 1
            self.name  = None

        elif e == 0:
            self.atype = VALUE
            self.a     = 1
            self.e     = 1
            self.name  = None
            
        else:
            self.atype = atype
            for axe in axes:
                if type(axe) is str:
                    self.axes.append(Axe(VAR, name=axe))
                elif type(axe) in [int, float]:
                    self.axes.append(Axe(VALUE, a=axe))
                else:
                    self.axes.append(axe.clone())
    
            self.a     = Fraction(a) 
            self.e     = Fraction(e)
            self.name  = name
            
            if self.atype in [SUM, PROD]:
                self.reduce()
            
        
    # ---------------------------------------------------------------------------
    # a and e must be fractions
    
    @property
    def a(self):
        return self.a_
    
    @a.setter
    def a(self, v):
        self.a_ = Fraction(v)
        
    @property
    def e(self):
        return self.e_
    
    @e.setter
    def e(self, v):
        self.e_ = Fraction(v)
        

    # ---------------------------------------------------------------------------
    # Clone and change
    
    def clone(self):
        clone = Axe.Zero() # To avoid calling init which calls reduce()

        clone.atype = self.atype
        clone.axes  = [axe.clone() for axe in self.axes]
        clone.a     = self.a
        clone.e     = self.e
        clone.name  = self.name
        
        return clone
    
    def change_to(self, other):
        
        if type(other) is Axe:
            self.atype  = other.atype
            self.axes   = [axe.clone() for axe in other.axes]
            self.a      = other.a
            self.e      = other.e
            self.name   = other.name
            
            return self
            
        elif type(other) is str:
            return self.change_to(Axe.Var(other))
        
        else:
            return self.change_to(Axe.Value(other))
        
    
    
    # ---------------------------------------------------------------------------
    # The x attribute of a.x^e
        
    @property
    def x(self):
        x = self.clone()
        x.a = 1
        x.e = 1
        return x
    
    @x.setter
    def x(self, value):
        atype = self.atype
        a     = self.a
        e     = self.e
        
        self.change_to(value)
        
        if atype != VALUE:
            if self.atype == VALUE:
                self.a = a * (self.a**e)
            else:
                self.a = a
                self.e = e
                
    # ---------------------------------------------------------------------------
    # The x^e  attribute of a.x^e
        
    @property
    def xe(self):
        x = self.clone()
        x.a = 1
        return x
    
    @xe.setter
    def xe(self, value):
        atype = self.atype
        a     = self.a
        
        self.change_to(value)
        
        if atype != VALUE and self.atype != VALUE:
            self.a = a 
            
    # ---------------------------------------------------------------------------
    # Count
    
    @property
    def count(self):
        count = 1
        for axe in self.axes:
            count += axe.count
        return count
    
    def contains(self, var_name):
        if self.atype == VAR and self.name == var_name:
            return 1
        else:
            count = 0
            for axe in self.axes:
                count += axe.contains(var_name)
            return count
        
    # ---------------------------------------------------------------------------
    # Creations
    
    @staticmethod
    def axe(v):
        if v is None:
            raise RuntimeError("Other parameter can't be None")
            
        if type(v) is str:
            return Axe.Var(v)
        
        elif issubclass(type(v), Axe):
            return v
        
        else:
            return Axe.Value(v)
        
    @classmethod
    def Value(cls, value):
        return cls(VALUE, a=value)
    
    @classmethod
    def Var(cls, name):
        return cls(VAR, name=name)
    
    @classmethod
    def Sum(cls, *args):
        return cls(SUM, *args).reduce()

    @classmethod
    def Prod(cls, *args):
        return cls(PROD, *args).reduce()

    @classmethod
    def Func(cls, name, *args):
        return cls(FUNC, *args, name=name)
    
    @classmethod
    def cos(cls, var):
        return cls.Func('cos', var)
    
    @classmethod
    def sin(cls, var):
        return cls.Func('sin', var)
    
    @classmethod
    def tan(cls, var):
        return cls.Func('tan', var)
    
    @classmethod
    def Zero(cls):
        return cls.Value(0)
    
    @classmethod
    def One(cls):
        return cls.Value(1)
    
    @property
    def func_inv(self):
        if self.atype == FUNC:
            for name, inv in FUNCTIONS.items():
                if name == self.name:
                    return inv
                elif inv == self.name:
                    return name
        return None

    # ---------------------------------------------------------------------------
    # Appendable
    
    @property
    def appendable(self):
        if self.atype == SUM:
            return self.a == 1 and self.e == 1
        
        elif self.atype == PROD:
            return self.e == 1
        
        else:
            return False
    
    # ---------------------------------------------------------------------------
    # Dump
    
    def dump(self, space=""):
        name = ATYPES[self.atype]
        if self.name is not None:
            name = "(" + name + " " + self.name + ")"
        
        print(space + f"atype: {self.a}*{name}^{self.e} -> {self}")
        if len(self.axes) > 0:
            for i, axe in enumerate(self.axes):
                print(space + "> a.x^e", i)
                axe.dump(space + "   ")
            print()
            
    # ---------------------------------------------------------------------------
    # Extract the variables
    
    def get_vars(self, xvars=None, values=False):
        
        if xvars is None: xvars = {}
        
        if self.atype == VAR:
            if values:
                xvars[self.name] = Axe.Value(np.random.randint(2, 20))
            else:
                xvars[self.name] = None
        else:
            for axe in self.axes:
                axe.get_vars(xvars, values=values)
                
        return xvars
    
    # ---------------------------------------------------------------------------
    # Variables to template
    
    @property
    def is_template(self):
        if self.atype == VAR:
            return self.name[0] == '$'
        else:
            for axe in self.axes:
                if axe.is_template:
                    return True
            return False
    
    def vars_to_template(self):
        if self.atype == VAR:
            if self.name[0] != '$':
                self.name = '$' + self.name
        else:
            for axe in self.axes:
                axe.vars_to_template()

        return self
    
    # ---------------------------------------------------------------------------
    # Template to axe
    
    def set_vars(self, xvars):
        if self.atype == VAR:
            x = xvars.get(self.name)
            if x is None:
                raise RuntimeError(f"set_vars error: var name '{self.name}' not found in {xvars}")
            
            self.x = x
                
        else:
            for axe in self.axes:
                axe.set_vars(xvars)
        return self
                
    # ---------------------------------------------------------------------------
    # Evaluate
    
    def compute(self, xvars=None):
        if xvars is None:
            xvars = self.get_vars()
            for k in xvars:
                xvars[k] = Axe.Value(np.random.randint(2, 20))

        clone = self.clone()
        clone.set_vars(xvars)
        
        return eval(clone.python, None, xvars)
    
    # ---------------------------------------------------------------------------
    # Compare two computations
    
    def check(self, other, seed=None):
        
        if not CHECK:
            return self
        
        xvars = {}
        xvars = self.get_vars(xvars)
        other.get_vars(xvars)
        
        if seed is not None:
            np.random.seed(seed)
        for v in xvars:
            xvars[v] = Axe.Value(np.random.randint(2, 20))
            
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
            
        if self.a == 0:
            return "0"
        
        if self.e == 0:
            return f"{self.a}"
        
        if self.atype == VALUE:
            return f"{self.a}"

        # ---------------------------------------------------------------------------
        # x to str
        
        if self.atype == VAR:
            s = self.name
            
        elif self.atype == FUNC:
            
            # special cases
            
            asin = self.name == 'arcsin' and source_code
            atan = self.name == 'arctan' and source_code and self.axes[0].atype == PROD
            
            # Arctan in arctan2
            
            if atan:
                arg = self.axes[0].reduced()
                
                s = "np.arctan2("
                num = Axe.Value(arg.a)
                den = Axe.One()
                for axe in arg.axes:
                    if axe.e < 0:
                        den = den/axe
                    else:
                        num = num*axe
                        
                s += num.to_str(True) + ", " + den.to_str(True) + ")" 
                
            # Standard cases
                
            else:
            
                s = f_pref + self.name + "("
                if asin:
                    s += "np.clip("
                
                for i, axe in enumerate(self.axes):
                    if i != 0:
                        s += ", "
                    s += axe.to_str(source_code)
                
                if asin:
                    s += ", -1, 1)"
                
                s += ")"

        elif self.atype == SUM:
            s = ""
            for i, axe in enumerate(self.axes):
                sx = axe.to_str(source_code)
                if axe.atype == SUM and axe.appendable:
                    sx = '(' + sx + ')'
                    
                if i == 0:
                    s = sx
                else:
                    if sx[0] == "-":
                        s += " - " + sx[1:]
                    else:
                        s += " + " + sx

        elif self.atype == PROD:
            
            num = []
            den = []
            for axe in self.axes:
                if axe.e < 0:
                    den.append(axe)
                else:
                    num.append(axe)
            num.extend(den)
            
            s = ""
            for i, axe in enumerate(num):
                
                # ---------------------------------------------------------------------------
                # x to string
                
                if axe.atype == VALUE:

                    # No value expression in a product
                    if len(self.axes) > 1:
                        return self.normalized().to_str(source_code)
                    
                    sx = f"{axe.a}"
                    e  = 1
                    
                else:
                    # No factor within axes of a product
                    if axe.a != 1:
                        return self.normalized().to_str(source_code)
                    
                    sx = axe.x.to_str(source_code)
                    e  = axe.e
                    
                if axe.atype == SUM or (axe.atype == PROD and axe.appendable):
                    sx = '(' + sx + ')'
                    
                # ---------------------------------------------------------------------------
                # Exponent part = x^e
                
                if abs(e) != 1:
                    if e.denominator == 1:
                        sx += EXP + f"{abs(e)}"
                    else:
                        sx += EXP + f"({abs(e)})"
                    
                # ---------------------------------------------------------------------------
                # chain of terms x^e

                oper = '/' if e < 0 else MUL0
                if i == 0:
                    if oper == '/':
                        s = '1/' + sx
                    else:
                        s = sx
                else:
                    if oper != '/':
                        if (s[-1] in LETTERS and sx[0] in LETTERS) or (s[-1] in FIGURES and sx[0] in FIGURES):
                            oper = MUL1
                        else:
                            oper = MUL0
                    
                    s += oper + sx

                            
                            
        # ---------------------------------------------------------------------------
        # Quick
        
        if self.a == 1 and self.e == 1:
            return s
                            
        # ---------------------------------------------------------------------------
        # Exponent
        
        ok_par = False
        if self.e != 1:
            if self.e == -1:
                suff = ""
            else:
                suff = f"{abs(self.e)}"
                if (self.e.denominator != 1):
                    suff = "(" + suff + ")"
                    
            ok_par = True
            if self.atype in [SUM, PROD]:
                s = "(" + s + ")"
                
            if suff != "":
                s = s + EXP + suff
                    
        # ---------------------------------------------------------------------------
        # Factor
        
        muldiv = '/' if self.e < 0 else MUL0
        pref = ""
        if self.a == 1:
            if self.e < 0:
                pref = "1/"
        else:
            if self.a == -1:
                if self.e < 0:
                    pref = "-1/"
                else:
                    pref = "-"
            elif self.a.denominator == 1:
                pref = f"{self.a}" + muldiv
            else:
                pref = f"({self.a})"  + muldiv
                
        if pref != "":
            if self.atype == SUM and not ok_par:
                s = '(' + s + ')'
            s = pref + s
            
        return s
    
    # ---------------------------------------------------------------------------
    # Representation

    def __repr__(self):
        return self.to_str(source_code=False)
    
    @property
    def python(self):
        return self.to_str(source_code=True)
    
    # ---------------------------------------------------------------------------
    # Compare two arrays of axes
    
    @staticmethod
    def comp_axes(axes0, axes1, sort=True):
        
        if len(axes0) > len(axes1):
            return -1
        elif len(axes0) < len(axes1):
            return 1

        # ----- Arrays have the same size
        
        if sort:
            ax0 = [axe.clone() for axe in axes0]
            ax1 = [axe.clone() for axe in axes1]
            ax0.sort()
            ax1.sort()
        else:
            ax0 = axes0
            ax1 = axes1
        
        for x0, x1 in zip(ax0, ax1):
            cmp = Axe.compare(x0, x1)
            if cmp != 0:
                return cmp
            
        return 0   
    
    
    # ---------------------------------------------------------------------------
    # Compare two axes
    
    @staticmethod
    def compare(axe0, axe1):
        
        # ---------------------------------------------------------------------------
        # axe0 is a template
        #
        # Templates are expressions were the variables start with $ sign
        # When comparing a template with a regular expression, we need a variable instancer
        # providing expressions for each variable.
        #
        # 
        # When the instancer gives no expression, it is initialized with the current expression
        #
        # The factor a is not taken into account, just used to compute the target expression
        # The exponent is not used if == 1
        #
        # With expression 2cos(x)^2, let's see what the template gives:
        #
        # Template  --> Instanciation
        # $a            $a = 2.cos(x)^2
        # $a^2          $a = 1/2.cos(x)
        # 2$a^2         $a = cos(x)
        # 
        # If the instancer returns an expression, this last one is used in the template and
        # the result is compared to the other expression
        #
        # Template  --> Instanciation    --> Resulting expression
        # $a            $a = 2.cos(x)^2      2.cos(x)^2
        # $a            $a = 1/2.cos(x)      1/2.cos(x)
        # $a            $a = cos(x)          cos(x)
        # $a^2          $a = 2.cos(x)^2      2.cos(x)^4
        # $a^2          $a = 1/2.cos(x)      1/2.cos(x)^2
        # $a^2          $a = cos(x)          cos(x)^2
        # 2$a^2         $a = 2.cos(x)^2      4.cos(x)^4
        # 2$a^2         $a = 1/2.cos(x)      1/2.cos(x)^2
        # 2$a^2         $a = cos(x)          2.cos(x)^2
        #
        # Note that normally, a should be controllet to be 1
          
        # ---------------------------------------------------------------------------
        # Comparizon with developed expressions
        
        axe0 = axe0.developed()
        axe1 = axe1.developed()
        
        # ---------------------------------------------------------------------------
        # Types
        
        if axe0.atype < axe1.atype:
            return -1
        if axe0.atype > axe1.atype:
            return 1
        
        # ---------------------------------------------------------------------------
        # Values
        
        if axe0.atype == VALUE:
            if axe0.a < axe1.a:
                return -1
            if axe0.a > axe1.a:
                return 1
            return 0
        
        # ---------------------------------------------------------------------------
        # Vars or funcs
        
        if axe0.atype in [VAR, FUNC]:
            
            if axe0.name < axe1.name:
                return -1
            if axe0.name > axe1.name:
                return 1
            
            if axe0.atype == FUNC:
                cmp = Axe.comp_axes(axe0.axes, axe1.axes, sort=False)
                if cmp != 0:
                    return cmp
                
        # ---------------------------------------------------------------------------
        # Sums or products
            
        elif axe0.atype in [SUM, PROD]:
            cmp = Axe.comp_axes(axe0.axes, axe1.axes, sort=True)
            if cmp != 0:
                return cmp
            
        # ---------------------------------------------------------------------------
        # Exponent and factor to compare
        
        if axe0.e < axe1.e:
            return -1
        if axe0.e > axe1.e:
            return 1
        
        if axe0.a < axe1.a:
            return -1
        if axe0.a > axe1.a:
            return 1
        
        # ---------------------------------------------------------------------------
        # Equal !!!!!
        
        return 0
    
    # ---------------------------------------------------------------------------
    # Comparison
    
    def __eq__(self, other):
        return Axe.compare(self, Axe.axe(other)) == 0
    
    def __lt__(self, other):
        return Axe.compare(self, Axe.axe(other)) < 0
    
    def __let__(self, other):
        return Axe.compare(self, Axe.axe(other)) <= 0
    
    def __gt__(self, other):
        return Axe.compare(self, Axe.axe(other)) > 0
    
    def __get__(self, other):
        return Axe.compare(self, Axe.axe(other)) >= 0
    
    # ---------------------------------------------------------------------------
    # Signs
    
    def __pos__(self):
        return self.clone()
    
    def __neg__(self):
        clone = self.clone()
        clone.a *= -1
        return clone
    
    def __abs__(self):
        clone = self.clone()
        clone.a = abs(clone.a)
        return clone
    
    # ---------------------------------------------------------------------------
    # Operations
    
    def __add__(self, other):
        o = Axe.axe(other).clone()
        r = self.clone()
        
        if r.atype == SUM and r.appendable:
            r.axes.append(o)
        
        elif o.atype == SUM and o.appendable:
            o.axes.insert(0, r)
            r = o
        
        else:
            r = Axe.Sum(r, o)
            
        return r.reduce()
        
    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        o = Axe.axe(other).clone()
        r = self.clone()
        
        if r.atype == PROD and r.appendable:
            if o.atype == VALUE:
                r.axes.insert(0, o)
            else:
                r.axes.append(o)
        
        elif o.atype == PROD and o.appendable:
            o.axes.insert(0, r)
            r = o
        
        else:
            if o.atype == VALUE:
                r = Axe.Prod(o, r)
            else:
                r = Axe.Prod(r, o)
                
        return r.reduce()
        
    def __truediv__(self, other):
        return self * (1/other)
    
        o = Axe.axe(other).clone()
        o.e *= -1
        o.a = 1/o.a
        o.reduce()
        return self * o
    
    def __pow__(self, other):
        if issubclass(type(other), Axe):
            if other.atype == VALUE:
                r = self ** other.a
            else:
                return Axe.Func('pow', self, other)
        else:
            r = self.clone()
            r.e *= other
            
        return r.reduce()
        
    # ---------------------------------------------------------------------------
    # Reverse operations
    
    def __radd__(self, other):
        return Axe.axe(other) + self
        
    def __rsub__(self, other):
        return Axe.axe(other) - self
        
    def __rmul__(self, other):
        return Axe.axe(other) * self
        
    def __rtruediv__(self, other):
        clone = self.clone()
        clone.e *= -1
        clone.a = 1/clone.a
        clone.normalize()
        return Axe.axe(other) * clone
    
    def __rpow__(self, other):
        return Axe.axe(other) ** self
    
    # ---------------------------------------------------------------------------
    # To product
    # Can be inverted with compact
        
    def to_product(self):
        clone = self.reduced()
        if self.atype == PROD:
            return clone
        
        else:
            prod       = Axe.Value(self.a)
            clone.a    = 1
            
            prod.atype = PROD
            prod.axes  = [clone]
            
            return prod
        
    def as_product_OLD(self):
        if self.atype == PROD:
            return self
        else:
            prod = Axe(PROD, self, a=self.a)
            if len(prod.axes) > 0:
                prod.axes[0].a = 1
            return prod
    
    # ---------------------------------------------------------------------------
    # Compact sums and products
    
    def compacted(self):
        return self.clone().compact()
    
    def compact(self):
        
        if self.a == 0:
            return self.Zero()
        
        if self.e == 0:
            return self.Value(self.a)
        
        for axe in self.axes:
            axe.compact()
            
        if self.atype == VALUE:
            if self.e != 1:
                self.a **= self.e
                self.e = 1
                
        elif self.atype in [SUM, PROD]:
            if len(self.axes) == 0:
                if self.atype == SUM:
                    return Axe.Zero()
                else:
                    return Axe.Value(self.a)
                
            elif len(self.axes) == 1:
                a = self.a
                e = self.e
                self.change_to(self.axes[0])
                self.a **= e
                self.a  *= a
                self.e  *= e
                
                return self.compact()
        
        return self
    
    # ---------------------------------------------------------------------------
    # Normalization
    
    def normalized(self):
        return self.clone().normalize()
    
    def normalize(self):
        
        self.compact()
        
        if self.atype == SUM:
            
            for axe in self.axes:
                axe.normalize()
            
            # ----- Distribute the factor if no exponent
            if self.e == 1:
                for axe in self.axes:
                    axe.a *= self.a
                    
                self.a = 1
                
            # ----- Merge the values
            v = 0
            axes = []
            for axe in self.axes:
                if axe.atype == VALUE:
                    v += axe.a
                else:
                    axes.append(axe)
                    
            # ----- Compact
            if len(axes) == 0:
                if v == 0:
                    self.change_to(Axe.Zero())
                else:
                    v **= self.e
                    v *= self.a
                    self.change_to(Axe.Value(v))
            else:
                if v != 0:
                    axes.append(Axe.Value(v))
                self.axes = axes
                self.compact()
                
        elif self.atype == PROD:
            
            # ----- Distribute the exponent
            if self.e != 1:
                for axe in self.axes:
                    axe.a **= self.e
                    axe.e  *= self.e
                self.e = 1
                
            # ----- One single factor
            for axe in self.axes:
                self.a *= axe.a
                axe.a = 1
                
            # ----- Remove the values
            axes = []
            for axe in self.axes:
                if axe.atype != VALUE:
                    axes.append(axe)
                    
            for axe in self.axes:
                axe.normalize()
                
            if len(axes) == 0:
                self.change_to(self.a)
                
            elif len(axes) == 1:
                a = self.a
                e = self.e
                self.change_to(axes[0])
                self.a **= e
                self.a *= a
                
            else:
                self.axes = axes
                
                
        return self
    
    # ---------------------------------------------------------------------------
    # Reduction
    
    def reduced(self):
        return self.clone().reduce()
    
    def reduce(self):
        
        check_ = self.clone()
        
        self.normalize()
        
        if self.atype in [VALUE, VAR]:
            return self

        # --------------------------------------------------
        # Reduces the sub axes
        
        
        for axe in self.axes:
            axe.reduce()
            
        if self.atype == FUNC:
            return self.check(check_)

        # --------------------------------------------------
        # Axes with no items
        
        if len(self.axes) == 0:
            
            raise RuntimeError(f"Strange, should have been avoided by compact !!!!")
            
            if self.atype == SUM:
                self.change_to(Axe.Zero())
                
            elif self.atype == PROD:
                raise RuntimeError(f"Product with no axes !!!!")
                self.change_to(Axe.Zero())
                #self.change_to(Axe.One())
                
            return self.compact().check(check_)

        # --------------------------------------------------
        # Axes with only one axe
        
        if len(self.axes) == 1:
            print("HERE", self, ATYPES[self.atype], len(self.axes))
            
            raise RuntimeError(f"Strange, should have been avoided by compact !!!!")
            
            axe = self.axes[0]
            axe.a **= self.e
            axe.a *= self.a
            self.change_to(axe)
            return self.reduce().check(check_)

        # --------------------------------------------------
        # Sum
        
        if self.atype == SUM:
            
            # --------------------------------------------------
            # Sum of sums
            
            axes = []
            for axe in self.axes:
                if axe.atype == SUM and axe.appendable:
                    axes.extend(axe.axes)
                else:
                    axes.append(axe)
            self.axes = axes

            # --------------------------------------------------
            # Merging
            
            axes  = []
            index = 0
            while index < len(self.axes):
                
                axe  = self.axes[index]
                i    = index + 1
                while i < len(self.axes):
                    
                    ax = self.axes[i]
                    
                    if axe.e == ax.e and axe.x == ax.x:
                        if verbose:
                            sverb = f"SUM MERGE> {axe} + {ax} ="

                        axe.a += ax.a
                        del self.axes[i]
                        
                        if verbose:
                            print(sverb, axe)

                    else:
                        i += 1
                        
                if axe.a != 0:
                    axes.append(axe)
                    
                index += 1
                
            self.axes = axes
            if len(self.axes) < 2:
                return self.reduce().check(check_)
            
        # --------------------------------------------------
        # Product
        
        elif self.atype == PROD:
            
            # --------------------------------------------------
            # Prod of prod
            
            axes = []
            for axe in self.axes:
                if axe.a != 1:
                    raise RuntimeError(f"Normally axe.a should be 1:", axe, " in ", self)
                if axe.atype == PROD:
                    axes.extend(axe.axes)
                else:
                    axes.append(
                        axe)
            self.axes = axes
            
            # --------------------------------------------------
            # Merging

            axes = []
            index = 0
            while index < len(self.axes):
                
                axe  = self.axes[index]
                i    = index + 1
                while i < len(self.axes):
                    
                    ax = self.axes[i]
                    
                    if axe.x == ax.x:
                        if verbose:
                            sverb = f"PROD MERGE> {axe} * {ax} ="
                            
                        axe.e += ax.e
                        del self.axes[i]
                        
                        if verbose:
                            print(sverb, axe, "e:", axe.e)

                    else:
                        i += 1
                        
                if axe.e != 0:
                    axes.append(axe)
                    
                index += 1
                
            self.axes = axes
            if len(self.axes) < 2:
                return self.compact().reduce().check(check_)
            
        # --------------------------------------------------
        # Done
        
        return self.check(check_)
        
    # ---------------------------------------------------------------------------
    # Development
    
    def developed(self):
        return self.clone().develop()
    
    def develop(self):
        
        check_ = self.clone()
        
        # ---------------------------------------------------------------------------
        # Develop a list made of sums
        # sums is a list containing one or several Axe of type SUM
        
        def dev_list(sums):
            
            # The shape give the number of items in each sum
            
            shape = tuple([len(axe.axes) for axe in sums])
            
            # Total number of products resulting from the development
            
            total = int(np.product(shape))

            # The developed products will be stored in x_sum
            
            axes = []
            
            # Let's develop the sums
            for index in range(total):
                
                # Unravel to get the indices per sum
                
                indices = np.unravel_index(index, shape)
                
                # Let's multiply the indices sums
                
                for i, ind in enumerate(indices):
                    if i == 0:
                        m = sums[i].axes[ind]
                    else:
                        m = m * sums[i].axes[ind]
                        
                # m is the current product, we store it in x_sum
                
                axes.append(m)
                
            # We have a sum which can be reduced
            
            return Axe.Sum(*axes).reduce()
        
        # ---------------------------------------------------------------------------
        # Start by a proper reduction
        
        self.reduce()
        
        if self.atype in [VALUE, VAR]:
            return self.check(check_)
        
        for axe in self.axes:
            axe.develop()
            
        if self.atype == FUNC:
            return self.check(check_)
        
        # ---------------------------------------------------------------------------
        # Sum
        
        if self.atype == SUM:
            
            # Since (a + b + c)**2 has been developed, need a global reduction
            
            self.reduce()
            
            # Reduce if an int exponent exists
            
            n = abs(self.e.numerator)
            if n != 1:
                sums = [self.clone() for i in range(n)]
                self.axes = dev_list(sums).axes
                self.e = Fraction(1 if self.e > 0 else -1, self.e.denominator)
                self.reduce()
            
        # ---------------------------------------------------------------------------
        # Product
        
        elif self.atype == PROD:
            
            # ---------------------------------------------------------------------------
            # Regroup all the sums with the same exponent
            
            the_sums = {}
            axes = []
            for axe in self.axes:
                if axe.atype == SUM:
                    
                    e = axe.e
                    n = abs(e.numerator)
                    d = e.denominator * (1 if e > 0 else -1)
                    
                    x = axe.x
                    if the_sums.get(d) is None:
                        the_sums[d] = []

                    # NOTE: the exponent is not taken into account by dev_list
                        
                    for i in range(n):
                        the_sums[d].append(x.clone())

                else:
                    axes.append(axe)
                    
            # The sums in ds will be developed in a product of big sums
            # The other items form a product which mulitplies all this
            # NOTE: self.a will taken into account later on
            
            if len(axes) == 0:
                product = Axe.One()
            elif len(axes) == 1:
                product = axes[0]
            else:
                product = Axe(PROD, *axes)
                
            # ---------------------------------------------------------------------------
            # Develop sums with the same exponents
            # Each entry in es will produce a sum of developed products
            # witj a give exponent
            # The sums will be stores in dev_part
            
            dev_part = []
            
            for d, sums in the_sums.items():
                
                dev   = dev_list(sums)
                dev.e = Fraction(1, d)
                dev_part.append(dev)

            # ---------------------------------------------------------------------------
            # All the products are concatened in axes
            # The sums, with their exponent, are in dev_part
            # We concatenate the sums with an exponent != 1 with the axes
            
            last_sum = None
            for axe in dev_part:
                if axe.e == 1:
                    if last_sum is not None:
                        raise RuntimeError("Algorithm error")
                    last_sum = axe
                else:
                    product = product*axe
                    product.normalize()
                    
            # ---------------------------------------------------------------------------
            # If there is as last_sum, we can develop it iwth the product
            
            if last_sum is not None:
                last_sum.a = self.a
                self.a = 1
                last_sum.normalize()
                axes = []
                for axe in last_sum.axes:
                    axes.append(product*axe)
                
                self.atype = SUM
                self.axes = axes
                self.normalize()
                
            else:
                product.a = self.a
                self.change_to(product)

                
        # ---------------------------------------------------------------------------
        # We are done !
        
        self.reduce()
        
        return self.check(check_)
    
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
        
        for xp_axe in xp.axes:
            found = False
            for i, axe in enumerate(prod.axes):
                if xp_axe.x == axe.x:
                    same_e = xp_axe.e == axe.e
                    found = same_e or (not exact)
                    if found:
                        if same_e:
                            del prod.axes[i]
                        else:
                            prod.axes[i].e -= xp_axe.e
                        break
                    
            if not found:
                return None
            
        return prod.reduce()
    
    # ---------------------------------------------------------------------------
    # Extract an expression from a sum
    #
    # Return the termes which are not in the extraction and the factor
    # of the extraction
    #
    # self = remain + factor*extract
    #
    # If the expression to extract is compact (ie != from a sum with e == 1)
    # Performe a prod_extract with exact == False
    #
    # If the expression to extract is a sum with e == 1
    #
    # Use the first term of the extraction and succefully divide all the espression
    # termes with it. Use the result of the division with the previsous algorithm
    # to get a factorized sum. Check if all the terms are exactly inside
    
    def factorize(self, expr):

        # ---------------------------------------------------------------------------
        # Make sure with work with true sums

        the_sum = self.compact().normalized()
        if the_sum.atype != SUM:
            return None, None
        
        # ---------------------------------------------------------------------------
        # Expression is None : we factorize by the common factors
        
        if expr is None:
            rem_sum = self.clone()
            factor  = Axe.One()
            
            if self.axes[0].atype == PROD:
                axes = self.axes[0].axes
            else:
                axes = [self.axes[0]]
                
            for axe in axes:
                r, f = rem_sum.factorize(axe)
                if r == 0:
                    factor = factor * axe
                    for i in range(len(rem_sum.axes)):
                        rem_sum.axes[i] = rem_sum.axes[i] / axe
                        
            if factor == Axe.One():
                return rem_sum
            else:
                factor    = factor ** rem_sum.e
                factor.a  = rem_sum.a
                rem_sum.a = 1
                
                return (factor * rem_sum).check(self)
        
        
        # ---------------------------------------------------------------------------
        # Factorizing a sum

        expr = expr.compact().normalized()
        
        if expr.atype == SUM and expr.e == 1:
            
            first = expr.axes[0]
            lasts = expr.axes[1:]
            
            # ----- Loop on the termes of the sum
            
            for i_axe, axe in enumerate(the_sum.axes):
                
                # ----- By dividing by the first term of the expression
                # we have a possible common factor
                
                factor = (axe / first).reduce()
                
                # ----- Divide by the factor
                
                axes = []
                for i, ax in enumerate(the_sum.axes):
                    if i != i_axe:
                        axes.append((ax/factor).reduce())

                # ----- Check if the axes contains the last terms of the expression
                
                for i_term, term in enumerate(lasts):
                    found = False
                    for i, ax in enumerate(axes):
                        if ax.xe == term.xe:
                            if ax.a == term.a:
                                del axes[i]
                            else:
                                ax.a -= term.a
                            found = True
                            break
                        
                    if not found:
                        break
                
                # ----- Found : we have our factorizion
                
                if found:
                    rem_axes = [(axe * factor).reduce() for axe in axes]
                    rem = Axe(SUM, *rem_axes).compact().reduce()
                    
                    if CHECK:
                        test = rem + factor*expr
                        try:
                            self.x.check(test)
                        except:
                            test = (rem + factor*expr)
                            print('-'*80)
                            print("Error in factorization")                    
                            print("Factorization of", self)
                            print("len axes, expr  ", len(self.axes), len(expr.axes))
                            print("By              ", expr)
                            print("Factor          ", factor)
                            print("rem_axes        ", rem_axes)
                            print("Rem             ", rem)
                            print("rem + f*expr    ", test)
                            print("Reduced         ", test.reduced())
                            
                            raise RuntimeError("Error in factorization")
                    
                    # Factorization of remaining :-)
                    
                    r, f = rem.factorize(expr)
                    if r is not None:
                        
                        # self = r + f.x + factor.x ==> r + (factor + f).x
                        
                        return r, (factor + f).reduce()
                    
                    else:
                    
                        return rem, factor
                
        # ---------------------------------------------------------------------------
        # We try to extract a compact expression from a sum
                
        else:
            expr_out = Axe.Zero()
            expr_in  = Axe.Zero()
            
            for axe in the_sum.axes:
                factor = axe.prod_extract(expr, exact=True)
                if factor is None:
                    expr_out = expr_out + axe
                else:
                    expr_in = expr_in + factor
                    
            if len(expr_in.axes) == 0:
                return None, None
            else:
                return expr_out, expr_in
            
        return None, None
    
    # ---------------------------------------------------------------------------
    # Templates are expressions where the variables start with $ sign
    # When comparing a template with a regular expression, we need a variable instancer
    # providing expressions for each variable.
    # 
    # When the instancer gives no expression, it is initialized with the current expression
    #
    # The factor a is not taken into account, just used to compute the target expression
    # The exponent is not used if == 1
    #
    # With expression 2cos(x)^2, let's see what the template gives:
    #
    # Template  --> Instanciation
    # $a            $a = 2.cos(x)^2
    # $a^2          $a = 1/2.cos(x)
    # 2$a^2         $a = cos(x)
    # 
    # If the instancer returns an expression, this last one is used in the template and
    # the result is compared to the other expression
    #
    # Template  --> Instanciation    --> Resulting expression
    # $a            $a = 2.cos(x)^2      2.cos(x)^2
    # $a            $a = 1/2.cos(x)      1/2.cos(x)
    # $a            $a = cos(x)          cos(x)
    # $a^2          $a = 2.cos(x)^2      2.cos(x)^4
    # $a^2          $a = 1/2.cos(x)      1/2.cos(x)^2
    # $a^2          $a = cos(x)          cos(x)^2
    # 2$a^2         $a = 2.cos(x)^2      4.cos(x)^4
    # 2$a^2         $a = 1/2.cos(x)      1/2.cos(x)^2
    # 2$a^2         $a = cos(x)          2.cos(x)^2
    #
    # Note that normally, a should be controllet to be 1     
    #
    # A template can be either a function, a sum or a product
    #
    # Function
    # sin($a) matches with sin(x^2) --> $a = x^2
    #
    # Product
    # a/b matches with 6cos(x).sin(x) --> $a = cos(x) $b = 1/sin(x) (reverse also works)
    #
    # Sum / Product
    # ^$a^2 - $b^2 matches with (2z.x^4 - 2z.y^2) --> $a = x^2 $b = y in 2z((x^2)^2 - y^2)
    #
    # A template is extractec from an expression by calling extract_template
    #
    # template          expression            extract
    # a^2 - b^2         cos(x)(x^4 - y^2)     cos(x)(template with a=x^2 and b=y^2)
    
    # ----------------------------------------------------------------------------------------------------
    # Build all the combination of small different indices within inbig
    # if permutations, the combinations include all the possible permutations
    
    @staticmethod
    def combinations(small, big, permutations=True):
        
        if big < small:
            return []
        
        if permutations:
            shape = tuple([big - i for i in range(small)])
            total = int(np.product(shape))
            a = np.zeros((total, small), int)
            
            for index in range(total):
                inds = np.unravel_index(index, shape)

                indices = [i for i in range(big)]
                select  = []
            
                for i in range(small):
                    select.append(indices[inds[i]])
                    del indices[inds[i]]
                    
                a[index] = np.array(select)
                
            return a
        
        raise RuntimeError("Not implemented")
        
        
        def f(n):
            return int(np.product(np.arange(1, n)))
        
    
    # ---------------------------------------------------------------------------
    # Template match
    
    @staticmethod
    def xvars_complete(xvars):
        for v in xvars.values():
            if v is None:
                return False
        return True
    
    def match_template(self, tp, xvars):
        
        if tp.atype != PROD or self.atype != PROD:
            raise RuntimeError(f"match_template only applies on products, not {self} and {tp}")
            
        # ---------------------------------------------------------------------------
        # Loop on all the permutations
        
        n         = len(tp.axes)
        matches   = []
        mem_xvars = dict(xvars)
        permuts   = Axe.combinations(len(tp.axes), len(self.axes))

        for i_perm in range(len(permuts)):
            
            xvars  = dict(mem_xvars)
            perm   = permuts[i_perm]
            select = [self.axes[i] for i in perm]
            item   = Axe(PROD, *select).to_product()
            
            # --------------------------------------------------
            # Compare one per one
            # found is False if no match
            
            found = True
            for i in range(n):
                
                # ----- The terms to compare
                
                try:
                    s0 = item.axes[i]
                    t0 = tp.axes[i]
                except:
                    print(self, tp, xvars)
                    print(perm, select, item)
                    print("i, n:", i, n)
                    print("len item, tp axes:", len(item.axes), len(tp.axes))
                    print()
                    self.dump()
                    raise RuntimeError("Snif :-(")
                
                # ----- If the template is a var, compare
                # with its value or iniialize the value
                
                if (t0.atype == VAR) and (t0.name[0] == '$'):
                    comp = xvars[t0.name]
                    if comp is None:
                        xvars[t0.name] = s0
                    else:
                        if comp != s0:
                            found = False
                            break
                        
                # Types must match to continue
                # with its value or iniialize the value
                
                elif (t0.atype != s0.atype) or ((t0.atype == s0.atype) and (t0.e not in [s0.e])):
                    found = False
                    break
                
                elif t0.atype in [VAR, VALUE]:
                    if t0 != s0:
                        found = False
                        break
                
                elif t0.atype == FUNC:
                    if t0.name == s0.name:
                        
                        m = s0.axes[0].to_product().match_template(t0.axes[0].to_product(), xvars)
                        
                        if len(m) == 0:
                            found = False
                            break

                        if len(m) != 1:
                            raise RuntimeError(f"Normally only one match", m)

                        xvars = m[0][1]
                    else:
                        found = False
                        
                else:
                    raise RuntimeError(f"The terms of a template in match_template can't be sums or products: {tp} includes {t0}")

            # --------------------------------------------------
            # Something found
            
            if found:
                r_axes = []
                for i in range(len(self.axes)):
                    if not i in perm:
                        r_axes.append(self.axes[i].clone())
                        
                if len(r_axes) == 0:
                    rem = Axe.Value(self.a/tp.a)
                else:
                    rem = Axe(PROD, *r_axes).reduce()
                    rem.a = self.a/tp.a
                    
                matches.append([rem, dict(xvars)])
                
                # OLD matches.append([(self/item.reduce()).reduce(), dict(xvars)])
        
        # ---------------------------------------------------------------------------
        # Return the founds marches
        
        xvars = mem_xvars
        return matches


    # ---------------------------------------------------------------------------
    # Apply a template
    
    def apply_template(self, template, target):

        check_ = self.clone()

        if self.atype == VALUE:
            return self

        # ---------------------------------------------------------------------------
        # Apply the templates on the arguments
        
        for axe in self.axes:
            axe.apply_template(template, target)
        
        # ---------------------------------------------------------------------------
        # Ensure template is a template
        # Initialize the tempalte variables
        
        template = template.clone().vars_to_template()
        xvars    = template.get_vars()

        # ---------------------------------------------------------------------------
        # The template is a sum. Only applies on sums
        
        if template.atype == SUM:
            if self.atype != SUM:
                return self
            
            if len(template.axes) > len(self.axes):
                return self
            
            # ---------------------------------------------------------------------------
            # Loop on all the combinations
            
            permuts   = Axe.combinations(len(template.axes), len(self.axes))
            mem_xvars = dict(xvars)
            poss      = []

            for i_perm in range(len(permuts)):
                
                perm   = permuts[i_perm]
                xvars  = dict(mem_xvars)
                
                full_match = True
                for i_tpl in range(len(template.axes)):
                    
                    tpl  = template.axes[i_tpl].to_product()
                    item = self.axes[perm[i_tpl]].to_product()
                    
                    if i_tpl == 0:
                        matches = item.match_template(tpl, xvars)
                        if len(matches) == 0:
                            full_match = False
                            break
                        the_vars = [match[1] for match in matches]
                    else:
                        
                        new_vars = []
                        for vrs in the_vars:
                            matches = item.match_template(tpl, vrs)
                            if len(matches) > 0:
                                for match in matches:
                                    new_vars.append(match[1])
                                    
                        if len(new_vars) == 0:
                            full_match = False
                            break
                        
                        the_vars = new_vars
                    
                # ------------------------------------------------------------
                # Full match, we can test
                
                if full_match:
                    
                    for xvars in the_vars:
                        ok = True
                        for name, val in xvars.items():
                            if val is  None:
                                ok = False
                                #break
                                raise RuntimeError(f"Template algorithm error. the variable '{name}' of the template '{template}' is not initialized in {xvars}")
                    
                    if ok:
                        formula = template.clone().set_vars(xvars)
                        rem, factor = self.factorize(formula)
                        
                        if rem is not None:
                            
                            if tpl_verbose:
                                s  = f"\nSUM >>>> Template '{template}' -> '{target}' applied to {self}:\n"
                                s += f"   rem:      {rem}\n"
                                s += f"   factor:   {factor}\n"
                                s += f"   template: {formula}\n"
                                s += f"   xvars:    {xvars}\n"
                                s +=  "   result:   "
                            
                            replace = (rem + factor*target.clone().set_vars(xvars)).reduce()
                            
                            if tpl_verbose:
                                print(s + f"{replace}\n")
                                
                            self.change_to(replace).reduce()
                            
                            self.check(check_)
                            
                            self.apply_template(template, target)
                            
                            return self.reduce()
                        
                    
        # ---------------------------------------------------------------------------
        # The template is a product
        # If self is a sum, already done one axes. Work only on products
        
        elif template.atype == PROD:
            
            if self.atype == SUM:
                return self.check(check_)
            
            prod = self.to_product()
            
            matches = prod.match_template(template, xvars)
            
            if tpl_verbose:
                print("Matches founds for ", template, ' in ', prod)
                print(matches)
            
            if len(matches) != 0:

                for match in matches:
                    xvs = match[1]
                    ok = True
                    for val in xvs.values():
                        if val is None:
                            ok = False
                            break

                    if ok:
                        if tpl_verbose:
                            print(f"\nPROD >>>> Template '{template}' -> {target} applied to {self}:")
                            print(f"   match[0]: {match[0]}")
                            print(f"   xvs:      {xvs}")
                            print(f"   found:    {template.clone().set_vars(xvs)}")
                            print(f"   target:   {target}")
                            print(f"             {target.clone().set_vars(xvs)}")
                            print(f"   rem:      {prod / template.clone().set_vars(xvs)}")
                            
                        res = (match[0] * target.clone().set_vars(xvs)).reduce()
                        self.change_to(res)
                        self.reduce()
                        
                        if tpl_verbose:
                            print(f"      res:   {res}")
                            print()
                        
                        return self.check(check_)
        
        return self

# ----------------------------------------------------------------------------------------------------
# Quaternion

class Quat:
    def __init__(self):
        self.q = [Axe.Zero() for i in range(4)] 
        
    @classmethod
    def Rotation(cls, axis, short=True):
        
        q = cls()
        a = 'abc'[axis]
        
        if short:
            q.w           = Axe.Var(f"c{a}")
            q.q[1 + axis] = Axe.Var(f"s{a}")
        else:
            var = Axe.Var(a)
            q.w           = Axe.Func("cos", var)
            q.q[1 + axis] = Axe.Func("sin", var)
            
        
        return q
        
    @property
    def w(self):
        return self.q[0]
    
    @w.setter
    def w(self, v):
        self.q[0] = Axe.axe(v)
    
    @property
    def x(self):
        return self.q[1]

    @x.setter
    def x(self, v):
        self.q[1] = Axe.axe(v)
    
    
    @property
    def y(self):
        return self.q[2]

    @y.setter
    def y(self, v):
        self.q[2] = Axe.axe(v)
    
    
    @property
    def z(self):
        return self.q[3]
    
    @z.setter
    def z(self, v):
        self.q[3] = Axe.axe(v)
        
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
        
        X = Axe.Var('$X')
        Y = Axe.Var('$Y')
        
        tpl_cos2sin2 = Axe.cos(X)**2 + Axe.sin(X)**2

        tpl_sincos   = Axe.sin(X)*Axe.cos(X)
        tgt_sincos   = (Axe.sin(2*X))/2

        tpl_1sin2    = 1 - 2*Axe.sin(X)**2
        tgt_1sin2    = Axe.cos(2*X)

        tpl_coscos   = -2*Axe.cos(X)**2*Axe.sin(Y)**2 - 2*Axe.cos(Y)**2*Axe.sin(X)**2 + 1
        tgt_coscos   = Axe.cos(2*X)*Axe.cos(2*Y)
        

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
                    
                comb0.right.reduce().apply_template(tpl_cos2sin2, Axe.One())
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
                    
                    num.right.reduce().apply_template(tpl_cos2sin2, Axe.One())
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
                        
                        den.right.apply_template(tpl_cos2sin2.clone(), Axe.One())
                        den.right.apply_template(tpl_coscos,   tgt_coscos)
                        
                        dens.append(den)
                        
                if True:
                    for i, eq in enumerate(dens):
                        print(i, '>', eq, '-->', eq.right.count)
                    print()

                
            
            return
        
    @staticmethod
    def python_quat_to_euler(space="    "*3):
        
        X = Axe.Var('$X')
        Y = Axe.Var('$Y')
        
        tpl_cos2sin2 = Axe.cos(X)**2 + Axe.sin(X)**2

        tpl_sincos   = Axe.sin(X)*Axe.cos(X)
        tgt_sincos   = (Axe.sin(2*X))/2

        tpl_1sin2    = 1 - 2*Axe.sin(X)**2
        tgt_1sin2    = Axe.cos(2*X)

        tpl_coscos   = -2*Axe.cos(X)**2*Axe.sin(Y)**2 - 2*Axe.cos(Y)**2*Axe.sin(X)**2 + 1
        tgt_coscos   = Axe.cos(2*X)*Axe.cos(2*Y)
        
        tpl_tan      = Axe.sin(X)/Axe.cos(X)
        tgt_tan      = Axe.tan(X)
        
        
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
                    
                comb0.right.reduce().apply_template(tpl_cos2sin2, Axe.One())
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
                    num.right = num.right.factorize(None)
                    
                    num.right.reduce().apply_template(tpl_cos2sin2, Axe.One())
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
                        den.right = den.right.factorize(None)
                        
                        den.right.apply_template(tpl_cos2sin2.clone(), Axe.One())
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
            for v in ['a', 'b', 'c']:
                print(f"{space}{left_euler[v]} = {right_euler[v]}")
            print()
 

            
# ----------------------------------------------------------------------------------------------------
# Equality between a left and a right expression
    
class Equality:
    
    def __init__(self, left, right):
        self.left  = Axe.axe(left).reduced()
        self.right = Axe.axe(right).reduced()
        
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

            if left.a != 1:
                self = self / left.a
                
            if left.e != 1:
                self = self ** -left.e
                
            if left.atype == VAR and left.name == var_name:
                return self
                
            loop = False
            if left.atype == SUM:
                for axe in left.axe:
                    if axe.contains(var_name) == 0:
                        self = self - axe
                        loop = True
                        continue
            if loop:
                continue
                    
            loop = False
            if self.left.atype == PROD:
                for axe in self.left.axe:
                    if axe.contains(var_name) == 0:
                        self = self / axe
                        loop = True
                        continue
                    
            if loop:
                continue
                    
            if left.atype == FUNC:
                inv = left.func_inv
                self.left = left.axes[0]
                self.right = Axe.Func(inv, self.right)
                continue
                    
        return self
    
    

Quat.python_quat_to_euler(space="")



