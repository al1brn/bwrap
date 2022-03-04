#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 09:35:16 2021

@author: alain
"""


import struct
import math

class Number_TEST():
    
    UNITS = {}
    
    def __init__(self, v, unit=None):
        if issubclass(type(v), Number):
            self.n = v.n
            self.e = 0
        else:
            self.n = v
            self.e = 0
        
    @property
    def sgn(self):
        return -1 if self.n < 0 else 1
    
    @sgn.setter
    def sgn(self, s):
        self.n = -abs(self.n) if s < 0 else abs(self.n)
            
    # ---------------------------------------------------------------------------
    # Units management
    
    @property
    def unit(self):
        if hasattr(self, 'unit_'):
            return self.unit_
        else:
            return None
        
    @unit.setter
    def unit(self, name):
        u = Number.UNITS.get(name)
        if u is None:
            Number.add_unit(name, self.precision)
        self.unit_ = name
        self.precision_to_unit()
    
    @classmethod
    def add_unit(cls, name, precision, fmt_func=None, fmt_unit=None, disp_prec=None):
        cls.UNITS[name] = {
            'precision': precision, 
            'f'        : fmt_func,
            'unit'     : name if fmt_unit is None else fmt_unit,
            'dprec'    : precision if disp_prec is None else disp_prec
            }
        
    def precision_to_unit(self):
        name = self.unit
        if name is not None:
            self.precision = Number.UNITS[name]['precision']

        return self
            
    # ---------------------------------------------------------------------------
    # Decimal representation
    
    def fmt(self, precision=None, f=None, unit=None):
        return f"{self.n}"

    # ---------------------------------------------------------------------------
    # Units management
            
    def __repr__(self):
        return f"{self.fmt()}"
    
    def copy(self, other):
        self.n = Number(other).n
        return self
    
    # ---------------------------------------------------------------------------
    # Float value  of the number
    
    @property
    def value(self):
        return self.n
    
    # ---------------------------------------------------------------------------
    # Highest bit
    
    @property
    def highest_bit(self):
        return 52
    
    # ---------------------------------------------------------------------------
    # Floor, ceil, trunc 
    
    @property
    def int_frac(self):
        return int(self.n)
        
    @property
    def is_int(self):
        return False
        
    @property
    def frac(self):
        return self.n - math.trunc(self.n)

    @property
    def floor(self):
        return math.floor(self.n)
    
    @property
    def ceil(self):
        return math.ceil(self.n)
    
    @property
    def trunc(self):
        return math.trunc(self.n)
    
    # ---------------------------------------------------------------------------
    # Sign management
    
    @property
    def is_null(self):
        return self.n == 0
    
    def set_sgn(self, sgn):
        self.sgn = sgn
        return self
    
    
    def abs(self):
        return abs(self.n)
        
    def opposite(self):
        return Number(-self.n)
    
    # ---------------------------------------------------------------------------
    # Multiply by power of two
    
    def mul_pow2_eq(self, p2, increase_e=False):
        
        if p2 == 0:
            return self
        
        elif p2 > 0:
            self.n *= (1 << p2)
            return self
        
        else:
            self.n /= (1 << (-p2))
            return self
        
    def mul_pow2(self, p2, increase_e=False):
        return Number(self).mul_pow2_eq(p2, increase_e)
    
    # ---------------------------------------------------------------------------
    # Align the exponent part
    
    def set_e(self, e):
        return self
    
    # ---------------------------------------------------------------------------
    # Precision

    @property
    def precision(self):
        return 15
    
    @precision.setter
    def precision(self, p):
        pass
    
    # ---------------------------------------------------------------------------
    # Return two copies with aligned exponents
    
    def align_e(self, other):
        return Number(self), Number(other)

        s = Number(self)
        if s.n == 0:
            s.sgn = 1

        o = Number(other)
        if o.n == 0:
            o.sgn = 1

        e = max(s.e, o.e)
        return s.set_e(e), o.set_e(e)
    
    # ---------------------------------------------------------------------------
    # Comparison
    
    def compare(self, other):
        s, o = self.align_e(other)
        if s.n == o.n:
            return 0
        
        return -1 if s.n < o.n else 1

    def greater(self, other):
        return self.compare(other) > 0
    
    def less(self, other):
        return self.compare(other) < 0
    
    def greater_equal(self, other):
        return self.compare(other) >= 0
    
    def less_equal(self, other):
        return self.compare(other) <= 0
    
    
    # ---------------------------------------------------------------------------
    # Addition 
    
    def add_eq(self, other):
        self.n += Number(other).n
        return self
    
    # ---------------------------------------------------------------------------
    # Addition 
    
    def add(self, other):
        return Number(self).add_eq(other)
    
    # ---------------------------------------------------------------------------
    # Substraction
    
    def sub_eq(self, other):
        return self.add_eq(Number(other).opposite())
    
    def sub(self, other):
        return Number(self).add_eq(Number(other).opposite())
        
    # ---------------------------------------------------------------------------
    # Multiplication
    
    def mul_eq(self, other):
        self.n *= Number(other).n
        return self
        
    def mul(self, other):
        return Number(self).mul_eq(other)

    # ---------------------------------------------------------------------------
    # Division
    
    def div_eq(self, other):
        self.n /= Number(other).n
        return self

    def div(self, other):
        return Number(self).div_eq(other)
    
    # ---------------------------------------------------------------------------
    # Square
    
    def square_eq(self):
        self.n =self.n*self.n
        return self
    
    def square(self):
        return self.mul(self).set_e(self.e)
    
    # ---------------------------------------------------------------------------
    # Squared root
    
    def sqrt_eq(self):
        self.n = math.sqrt(self.n)
        return self

    def sqrt(self):
        return Number(self).sqrt_eq()
    
    # ---------------------------------------------------------------------------
    # Invert
    
    def inv(self):
        self.n = 1/self.n
        return self
    
    def inv_eq(self):
        return self.copy(self.inv())
    
    # ---------------------------------------------------------------------------
    # Power of an int
    
    def pow_int_eq(self, expn):
        self.n = math.pow(self.n, expn)
        return self

    def pow_int(self, n):
        return Number(self).pow_int_eq(n)
    
    # ---------------------------------------------------------------------------
    # Any power
    #
    # Mulitply power by fractional power
    
    def pow(self, pw):
        self.n = math.pow(self.n, pw)
        return self
    
    def pow_eq(self, pw):
        return self.copy(self.pow(pw))
    
# ---------------------------------------------------------------------------
# Number class

class Number():
    
    UNITS = {}
    
    def __init__(self, v, unit=None):

        if isinstance(v, float):
            n, e, neg = Number.double_parts(v)
            
            self.n   = n + (1 << 52)    # 1 + frac
            self.e   = e - 1023         # exponent of 2
            self.sgn = -1 if neg else 1 # sign
            
            if self.e == 0:
                self.e = 52
                
            elif self.e < 0:
                self.n << -self.e
                self.e = 52 - self.e
                
            elif self.e > 52:
                self.n <<= self.e - 52
                self.e = 0

            else:
                self.n << self.e
                self.e = 52 - self.e
            
        elif isinstance(v, int):
            self.n   = abs(v)
            self.sgn = -1 if v < 0 else 1
            self.e   = 0
            
        elif issubclass(type(v), Number):
            self.copy(v)
            
        else:
            raise RuntimeError(f"Unknwo type '{type(v).__name__}' to initialize a Number: {v}")
            
        if unit is not None:
            self.unit = unit
            
    # ---------------------------------------------------------------------------
    # Units management
    
    @property
    def unit(self):
        if hasattr(self, 'unit_'):
            return self.unit_
        else:
            return None
        
    @unit.setter
    def unit(self, name):
        u = Number.UNITS.get(name)
        if u is None:
            Number.add_unit(name, self.precision)
        self.unit_ = name
        self.precision_to_unit()
    
    @classmethod
    def add_unit(cls, name, precision, fmt_func=None, fmt_unit=None, disp_prec=None):
        cls.UNITS[name] = {
            'precision': precision, 
            'f'        : fmt_func,
            'unit'     : name if fmt_unit is None else fmt_unit,
            'dprec'    : precision if disp_prec is None else disp_prec
            }
        
    def precision_to_unit(self):
        name = self.unit
        if name is not None:
            self.precision = Number.UNITS[name]['precision']

        return self
            
    # ---------------------------------------------------------------------------
    # Decimal representation
    
    def fmt(self, precision=None, f=None, unit_name=None, unit=None):
        
        name = self.unit if unit is None else unit
        if name is None:
            fmt_prec  = self.precision
            fmt_f     = None
            fmt_uname = ""
        else:
            UNIT = Number.UNITS[name]
            
            fmt_prec  = UNIT['dprec']
            fmt_f     = UNIT['f']
            fmt_uname = UNIT['unit']

        if precision is not None:
            fmt_prec = precision
        if f is not None:
            fmt_f = f
        if unit_name is not None:
            fmt_uname = unit_name
            
        # ---------------------------------------------------------------------------
        # The intermediary number
        
        n = Number(self, unit=None)
        if fmt_f is not None:
            n = fmt_f(n)
            
        n.precision = fmt_prec
        
        # ---------------------------------------------------------------------------
        # Format this number
            
        s_sgn = "-" if n.sgn < 0 else ""
        s = f"{s_sgn}{abs(n.trunc)}"
        i_frac = n.int_frac
        
        if i_frac != 0:
            #p10 = round(math.log10(1 << n.e))
            s += "." + format((i_frac*(10**fmt_prec)) // (1 << n.e), f"0{fmt_prec}d")
            
        if fmt_uname != "":
            s += " " + fmt_uname
            
        return s
            
    
    # ---------------------------------------------------------------------------
    # Units management
            
    def __repr__(self):
        return f"{self.fmt()}"
    
    def copy(self, other):
        self.n   = other.n
        self.e   = other.e
        self.sgn = other.sgn
        if hasattr(other, 'unit_'):
            self.unit_ = other.unit_
        
        return self
            
    # ---------------------------------------------------------------------------
    # Decompose the structure of a double
            
    @staticmethod
    def double_parts(v):
        bits = int.from_bytes(struct.pack('!d', v), 'big')
        return bits & ((1 << 52) - 1), (bits >> 52) & ((1 << 11) - 1), bits & (1 << 63) != 0

    # ---------------------------------------------------------------------------
    # Compute a double from its parts
    
    @staticmethod
    def double_parts_to_float(n, e, neg):
        frac = (n/(1 << 52))
        if e > 1023:
            v = (1 + frac)*(1 << (e - 1023))
        else:
            v = (1 + frac)*(1 >> (1023 - e))
            
        return -v if neg else v
    
    # ---------------------------------------------------------------------------
    # Float value  of the number
    
    @property
    def value(self):
        return self.n/(1 << self.e)*self.sgn
    
    # ---------------------------------------------------------------------------
    # Highest bit
    
    @property
    def highest_bit(self):
        if self.n == 0:
            return 0

        i = 0
        while (1 << i) <= self.n:
            i += 1
            
        return i-1
    
    # ---------------------------------------------------------------------------
    # Floor, ceil, trunc 
    
    @property
    def int_frac(self):
        if self.e == 0:
            return 0
        else:
            return self.n & ((1 << self.e) - 1)
        
    @property
    def is_int(self):
        return self.int_frac == 0
        
    @property
    def frac(self):
        
        if self.e == 0:
            return 0.
        
        f = self.int_frac
        if f == 0:
            return 0.
        else:
            return f/(1 << self.e)
            
    @property
    def floor(self):
        
        if self.e == 0:
            return self.n
        
        f = self.n >> self.e
        
        if self.int_frac == 0:
            return f
        
        return - f - 1 if self.sgn < 0 else f
    
    @property
    def ceil(self):
        if self.int_frac == 0:
            return self.floor
        else:
            return self.floor + 1
    
    @property
    def trunc(self):
        return self.ceil if self.sgn < 0 else self.floor
    
    # ---------------------------------------------------------------------------
    # Sign management
    
    @property
    def is_null(self):
        return self.n == 0
    
    def set_sgn(self, sgn):
        self.sgn = sgn
        return self
    
    
    def abs(self):
        return Number(self).set_sgn(1)
        
    def opposite(self):
        return Number(self).set_sgn(-self.sgn)
    
    # ---------------------------------------------------------------------------
    # Multiply by power of two
    
    def mul_pow2_eq(self, p2, increase_e=False):
        
        if p2 == 0:
            return self
        
        elif p2 > 0:
            self.n <<= p2
            return self
        
        else:
            if increase_e:
                self.e += -p2
            else:
                self.n = (self.n >> -p2) + int((self.n & (1 << (-p2-1))) != 0)
            return self
        
    def mul_pow2(self, p2, increase_e=False):
        return Number(self).mul_pow2_eq(p2, increase_e)
    
    # ---------------------------------------------------------------------------
    # Align the exponent part
    
    def set_e(self, e):
        if self.e == e:
            pass
        
        elif self.e > e:
            d = self.e - e
            self.n = (self.n >> d) + int((self.n & (1 << (d-1))) != 0)
        
        else:
            self.n <<= (e - self.e)
            
        self.e = e

        return self
    
    # ---------------------------------------------------------------------------
    # Precision

    @property
    def precision(self):
        return round(math.log(2)/math.log(10)*self.e)
    
    @precision.setter
    def precision(self, p):
        self.set_e(round(math.log(10)/math.log(2)*p))
    
    # ---------------------------------------------------------------------------
    # Return two copies with aligned exponents
    
    def align_e(self, other):

        s = Number(self)
        if s.n == 0:
            s.sgn = 1

        o = Number(other)
        if o.n == 0:
            o.sgn = 1

        e = max(s.e, o.e)
        return s.set_e(e), o.set_e(e)
    
    # ---------------------------------------------------------------------------
    # Comparison
    
    def compare(self, other):
        s, o = self.align_e(other)

        if s.sgn < 0:
            # - - 
            if o.sgn < 0:
                if s.n == o.s:
                    return 0
                elif self.n < o.n:
                    return 1
                else:
                    return -1
            # - +
            else:
                return -1
        else:
            # + -
            if o.sgn < 0:
                return 1
            
            # + +
            else:
                if s.n == o.n:
                    return 0
                elif self.n < o.n:
                    return -1
                else:
                    return 1
                
    def greater(self, other):
        return self.compare(other) > 0
    
    def less(self, other):
        return self.compare(other) < 0
    
    def greater_equal(self, other):
        return self.compare(other) >= 0
    
    def less_equal(self, other):
        return self.compare(other) <= 0
    
    
    # ---------------------------------------------------------------------------
    # Addition 
    
    def add_eq(self, other):
        
        o = Number(other)
        
        me = max(self.e, o.e)
        self.set_e(me)
        o.set_e(me)
        
        if self.sgn < 0:
            
            # - -
            if o.sgn < 0:
                self.n += o.n
                
            # - + 
            else:
                if self.n < o.n:
                    self.n = o.n - self.n
                    self.sgn = 1
                    
                else:
                    self.n -= o.n
                    
        else:
            
            # + -
            if o.sgn < 0:
                if self.n > o.n:
                    self.n -= o.n
                    
                else:
                    self.n = o.n - self.n
                    self.sgn = -1
                    
            # + +
            else:
                self.n += o.n
                
        # Avoid -0
                
        if self.n == 0:
            self.sgn = 1

        return self
    
    # ---------------------------------------------------------------------------
    # Addition 
    
    def add(self, other):
        return Number(self).add_eq(other)
    
    # ---------------------------------------------------------------------------
    # Substraction
    
    def sub_eq(self, other):
        return self.add_eq(Number(other).opposite())
    
    def sub(self, other):
        return Number(self).add_eq(Number(other).opposite())
        
    # ---------------------------------------------------------------------------
    # Multiplication
    
    def mul_eq(self, other):
        o = Number(other)
        self.n   *= o.n
        self.sgn *= o.sgn
        self.e   += o.e
        return self
        
    def mul(self, other):
        return Number(self).mul_eq(other)

    # ---------------------------------------------------------------------------
    # Division a / b
    # with : m = 2^e
    # With can compute (k, r) = divmod(ma, mb)
    # ma = k.mb + r
    # m.a/b = mk + r/b
    # r/b = mr/mb
    
    def div_eq(self, other):
        
        if self.is_null:
            return self
        
        o = Number(other)
        if o.is_null:
            raise RuntimeError("Number: division by zero.")
            
        if True:
            #e = max(self.e, (o.highest_bit-o.e) - (self.highest_bit-self.e)+ 2)
            e = max(self.e, self.highest_bit + 2)
        else:
            e = max(self.e, o.e)
            
        self.set_e(e)
        o.set_e(e)

        k, r = divmod(self.n, o.n)
        self.n = (k << e) + ((r << e)//o.n)
        
        self.sgn *= o.sgn
        
        return self

    def div(self, other):
        return Number(self).div_eq(other)
    
    # ---------------------------------------------------------------------------
    # Square
    
    def square_eq(self):
        e = self.e
        return self.mul_eq(self).set_e(e)
    
    def square(self):
        return self.mul(self).set_e(self.e)
    
    # ---------------------------------------------------------------------------
    # Squared root
    
    def sqrt_eq(self):
        
        if self.is_null:
            return self
        
        if self.sgn < 0:
            raise RuntimeError("Number: sqrt of negative number {self}.")
            
        target = self.n << self.e
        
        n0 = 1 << ((self.highest_bit + self.e) >> 1)
        n1 = n0 << 1
        
        if n0*n0 == target:
            n = n0
            
        elif n1*n1 == target:
            n = n1
            
        else:
            n = (n0 + n1) >> 1
        
            while n1 - n0 > 1:
                n = (n0 + n1) >> 1
                n2 = n * n
                if n2 == target:
                    break
                
                elif n2 > target:
                    n1 = n
                    
                else:
                    n0 = n
                    
        self.n = n
        
        return self
    
    def sqrt(self):
        return Number(self).sqrt_eq()
    
    # ---------------------------------------------------------------------------
    # Invert
    
    def inv(self):
        return Number(1).set_e(self.e).div_eq(self)
    
    def inv_eq(self):
        return self.copy(self.inv())
    
    # ---------------------------------------------------------------------------
    # Power of an int
    # c = (ma)^n = m^n.a^n
    # m.a^n = c/m^(n-1)
    #
    # m = 2^e ==> m^(n-1) = (2^e)^(n-1) = 2^(e(n-1)) = 1 << e.(n-1)
    #
    # m.(a^n) = c/((n-1) << e)
    
    def pow_int_eq(self, expn):
        
        if expn == 0:
            self.n = 1 << self.e
            self.sgn = 1
            return self
        
        if expn == 1:
            return self
        
        if expn < 0:
            return self.pow_int_eq(-expn).inv_eq()
        
        if True:
            
            # ----- Number of bits in the exponent
            
            nbits = Number(expn).highest_bit + 1
            
            # ----- Enhance temporarily the precision
            
            e  = self.e
            ne = e + 2*nbits
            self.set_e(ne)
            
            # Value of the last ignored bit
            half = 1 << (ne - 1)
            
            # ----- Each loop, the value n^(2 << 1) is computed
            # it is multiplied with the result depending on the current bit
            # of exponent
            # The precision is reduced to ne at each step
            
            n_p2 = self.n                # n^(2^i)
            n    = Number(1).set_e(ne).n # The result
            
            for i in range(nbits):
                
                # Current bit is one
                if (expn & (1 << i)) != 0:
                    # Multiply by the current power of n
                    n *= n_p2
                    
                    # Reduce the precision from 2*ne to ne
                    n = (n >> ne) + int((n & half) != 0)
                    
                # Next multiplication with reduction
                if i < nbits-1:
                    n_p2 *= n_p2
                    n_p2 = (n_p2 >> ne) + int((n_p2 & half) != 0)
            
            # ----- Return by setting the initial precision
            
            self.n = n
            return self.set_e(e)
            
            
        else:
        
            # ----- (ma) ^n
            
            hb = Number(expn).highest_bit
            pn = self.n
            r = 1
            
            for i in range(hb+1):
                if (expn & (1 << i)) != 0:
                    r *= pn
                pn *= pn                
                    
            # ----- / m^(n-1)
            
            self.n = r // (1 << self.e*(expn - 1))
        
            return self
    
    def pow_int(self, n):
        return Number(self).pow_int_eq(n)
    
    # ---------------------------------------------------------------------------
    # Any power
    #
    # Mulitply power by fractional power
    
    def pow(self, pw):
        
        npw = Number(pw)
        if npw.sgn < 0:
            return self.pow(npw.abs()).inv_eq()
        
        # ---------------------------------------------------------------------------
        # Int part
        
        i_pow = self.pow_int(npw.trunc)
        
        # ---------------------------------------------------------------------------
        # Fraction part of the exponent
        
        i_frac = npw.int_frac
        
        # ----- Null : this is an int power
        
        if i_frac == 0:
            return i_pow
        
        # ----- Error
        
        if self.sgn < 0:
            raise RuntimeError(f"Power {pw} of a negative value {self}")
        
        
        # ----- Not null : power algorithm
        
        # Number of bits of the fraction
        
        nbits = Number(i_frac).highest_bit + 1
        
        # ----- Use a higher precision
        
        ne  = self.e + 2*nbits
        
        # ----- Each loop, the value n^(2 << 1) is computed
        # it is multiplied with the result depending on the current bit
        # of exponent
        # The precision is reduced to ne at each step
        
        n_p2  = Number(self).set_e(ne).sqrt() # n^(1/2^i)
        f_pow = Number(1).set_e(ne)           # The result
        
        for i in reversed(range(npw.e)):

            # Current bit is one
            if (i_frac & (1 << i)) != 0:
                # Multiply by the current power of n
                f_pow.mul_eq(n_p2).set_e(ne)
                
            # Next sqrt with reduction
            if i > 0:
                n_p2.sqrt_eq()
        
        # ----- Return by setting the requested precision
        
        return i_pow.mul_eq(f_pow).set_e(self.e)
    
    def pow_eq(self, pw):
        return self.copy(self.pow(pw))

    
# ---------------------------------------------------------------------------
# Vector class

class Vector():
    def __init__(self, xy=(0, 0), unit=None):
        
        if issubclass(type(xy), Vector):
            self.x = Number(xy.x)
            self.y = Number(xy.y)
            
        elif hasattr(xy, '__len__'):
            self.x, self.y = Number(xy[0]).align_e(Number(xy[1]))
            
        elif hasattr(xy, 'x') and hasattr(xy, 'y'):
            self.x, self.y = Number(xy.x).align_e(Number(xy.y))
            
        else:
            self.x = Number(xy)
            self.y = Number(xy)
            
        if unit is not None:
            self.x.unit = unit
            self.y.unit = unit
            
    def fmt(self, **kwargs):
        return f"[{self.x.fmt(**kwargs)} {self.y.fmt(**kwargs)}]"
            
    def __repr__(self):
        return f"[{self.x}, {self.y}]"
        
    def clone(self):
        v = Vector()
        return v.copy(self)
        
    def copy(self, other):
        self.x = Number(other.x)
        self.y = Number(other.y)
        self.unit = other.unit
        return self
    
    @property
    def value(self):
        return [self.x.value, self.y.value]
    
    @property
    def unit(self):
        return self.x.unit
    
    @unit.setter
    def unit(self, name):
        self.x.unit = name
        self.y.unit = name
        
    def precision_to_unit(self):
        self.x.precision_to_unit()
        self.y.precision_to_unit()
        return self
    
    @property
    def e(self):
        return self.x.e
        
    def set_e(self, e):
        self.x.set_e(e)
        self.y.set_e(e)
        
        return self
    
    def align_e(self, other):
        o = Vector(other)
        e = max(self.e, o.e)
        return Vector(self).set_e(e), o.set_e(e)
        
    @property
    def precision(self):
        return self.x.precision

    @precision.setter
    def precision(self, p):
        self.x.precision = p
        self.y.precision = p
        
    def add_eq(self, other):
        self.x.add_eq(Vector(other).x)
        self.y.add_eq(Vector(other).y)
        
        return self
    
    def add(self, other):
        return self.clone().add_eq(other)

    def sub_eq(self, other):
        self.x.sub_eq(Vector(other).x)
        self.y.sub_eq(Vector(other).y)
        
        return self
    
    def sub(self, other):
        return self.clone().sub_eq(other)
    
    def mul_eq(self, value):
        self.x.mul_eq(value)
        self.y.mul_eq(value)
        
        return self
        
    def mul(self, value):
        return self.clone().mul_eq(value)
    
    def div_eq(self, value):
        self.x.div_eq(value)
        self.y.div_eq(value)
        
        return self
        
    def div(self, value):
        return self.clone().div_eq(value)
    
    def vmul_eq(self, other):
        self.x.mul_eq(Vector(other).x)
        self.y.mul_eq(Vector(other).y)
        
        return self
        
    def vmul(self, other):
        return self.clone().vmul_eq(other)
    
    def vdiv_eq(self, other):
        self.x.div_eq(Vector(other).x)
        self.y.div_eq(Vector(other).y)
        
        return self
        
    def vdiv(self, other):
        return self.clone().vdiv_eq(other)
    
    # ---------------------------------------------------------------------------
    # Squared norm
    
    @property
    def squared_norm(self):
        return self.x.square().add_eq(self.y.square())
    
    @property
    def norm(self):
        return self.squared_norm.sqrt_eq().set_e(self.e)
    
    @norm.setter
    def norm(self, n):
        cur_n, new_n = self.norm.align_e(Number(n))
        self.mul_eq(new_n.div_eq(cur_n))
    
    def normalize(self):
        return self.div_eq(self.norm)
    
    def normalized(self):
        return Vector(self).normalize()
    
    # ---------------------------------------------------------------------------
    # Dot product
    
    def dot(self, other):
        s, o = self.align_e(other)
        return s.x.mul(o.x).add(s.y.mul(o.y))
    
    def cross(self, other):
        s, o = self.align_e(other)
        return s.x.mul(o.y).sub(s.y.mul(o.x))
    


GMdt = Number(-16593982104374999040).set_e(0)
r2   = Number(4874399525610000000000).set_e(27)
r    = Number(69816900000).set_e(27)
v    = Vector([0, 38860]).set_e(17)
p    = Vector([69816900000, 0]).set_e(27)

a = GMdt.div(r2).div(r)
print(a, a.e)

print(p)
print(p.mul(a))


