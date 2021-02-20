#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:24:15 2021

@author: alain
"""

# -----------------------------------------------------------------------------------------------------------------------------
# Curved spaces
#
# Créé en août 2017
# Module le 30/09/2017
# Modified february 2021

CURVES_DEBUG=None

from math import cos, sin, tan, asin, acos, atan, atan2, radians, degrees, sqrt, pi
import numpy as np

from .bezier import  from_points


#from mathutils import (Matrix, Vector)
#import fourD
#import automator as atm

# -----------------------------------------------------------------------------------------------------------------------------
# Derivative function

def derivative_f(f, s, ds = 0.01):
    ds2 = ds/2
    return (f(s + ds2) - f(s - ds2))/ds

# -----------------------------------------------------------------------------------------------------------------------------
# Partial derivative

def partial_f(f, X, i, dX):

    XL = np.array(X, np.float)

    XL[i] = XL[i] - dX[i]/2.
    v0 = f(XL)

    XL[i] = XL[i] + dX[i]
    v1 = f(XL)

    return (v1 - v0)/dX[i]

# -----------------------------------------------------------------------------------------------------------------------------
# Second derivative

def partial2_f(f, X, i, j, dX):
    return partial_f(lambda XL: partial_f(f, XL, i, dX), X, j, dX)

# -----------------------------------------------------------------------------------------------------------------------------
# Some formating

def a2s(a):
    s = "["
    sep = ""
    for v in a:
        a += sep + f"{v:6.2f}"
        sep = ", "
    return s + "]"

# -----------------------------------------------------------------------------------------------------------------------------
# Curved space
#
# The following functions must/can be overloaded:
#
# Mandatory:
# - f(self, X): returns a vector
#
# Facultative:
# - partial_calc(self, X, i)      : Partial derivative
# - partial2_calc(self, X, i, j)  : Second partial derivative
# - partial_metric_tensor_calc(X) : Derivative of the metric tensor

class Space():

    # ---------------------------------------------------------------------------
    # Initialize with the space dimension
    # Create the dXs with their default values

    def __init__(self, dim, by_metric=False, dX=None):
        self.dim = dim
        self.by_metric = by_metric
        if dX is None:
            self.dX = np.ones(dim)*0.01
        else:
            self.dX = np.array(dX, np.float)
        self.basis_is_inversible = False # True only when space and amp dimes are equal

    # ---------------------------------------------------------------------------
    # Return an empty Vector

    def empty_X(self):
        return np.zeros(self.dim, np.float)

    # ---------------------------------------------------------------------------
    # Retourne un vecteur 3D pour visualisation

    def to3D(self, X):
        if len(X) == 1:
            return np.array((X[0], 0, 0))
        elif len(X) == 2:
            return np.array((X[0], X[1], 0))
        elif len(X) == 3:
            return np.array(X)
        else:
            return np.array(X)[:3]

    # ---------------------------------------------------------------------------
    # The function return a vector for given map coordinates

    def f(self, X):
        raise RuntimeError(f"Function f of Space class must be overloaded in subclass {type(self)}")

    # ---------------------------------------------------------------------------
    # Partial derivative
    # Numerically computed if the partial_calc method doesn't exist

    def partial(self, X, i):
        if hasattr(self, 'partial_calc'):
            return self.partial_calc(X, i)

        return partial_f(lambda Y: self.f(Y), X,  i, self.dX)

    # ---------------------------------------------------------------------------
    # Second partial derivative

    def partial2(self, X, i, j):
        if hasattr(self, 'partial2_calc'):
            return self.partial2_calc(X, i, j)

        return partial2_f(lambda Y: self.f(Y), X, i, j, self.dX)

    # ---------------------------------------------------------------------------
    # Covariant basis

    def covariant_basis(self, X):

        # If defined by metric : returns simply an orthonormal basis

        if self.by_metric:
            return np.identity(self.dim)

        # Otherwise, covariant basis derivative

        return np.array([self.partial(X, i) for i in range(self.dim)])

    # ---------------------------------------------------------------------------
    # Metric tensor

    def metric_tensor(self, X, basis=None):

        # If defined by metric: the method must be overloaded!

        if self.by_metric:
            return np.identity(self.dim)

        # Defined by covariant basis: we compute the dot products

        if basis is None:
            basis = self.covariant_basis(X)

        g = np.zeros((self.dim,self.dim), np.float)

        for i in range(self.dim):
            for j in range(self.dim):
                g[i,j] = np.dot(basis[i], basis[j])

        return g

    # ---------------------------------------------------------------------------
    # Partial derivative of the metric tensor
    # Numerically computed if partial_metric_tensor_calc method doesn't exist
    # Returns a tensor of order 3

    def derivated_metric_tensor(self, X):

        if hasattr(self, 'partial_metric_tensor_calc'):
            return self.partial_metric_tensor_calc(X)

        Dg = np.zeros(self.dim**3).reshape((self.dim, self.dim, self.dim))

        for i in range(self.dim):
            for j in range(self.dim):
                for k in range(self.dim):

                    def f(s):
                        Y = X.copy()
                        Y[k] = s
                        return self.metric_tensor(Y)[i,j]

                    Dg[i,j, k] = derivative_f(f, X[k], self.dX[k])

        return Dg

    # ---------------------------------------------------------------------------
    # Basis matrix inversion

    def invert_basis(self, X, basis=None):

        if basis is None:
            basis = self.covariant_basis(X)

        # Must be invertible !

        if not self.basis_is_inversible:
            raise RuntimeError(f"Curved space {type(self)}: invert_basis" +
            f"The space dimension is {self.dim}, the shape of the basis is {basis.shape}. It can't be inverted!"
            )

        # Inversion

        try:
            res = np.linalg.inv(basis)
        except Exception as inst:
            raise RuntimeError(f"Curved space {type(self)}: invert_basis" +
            f"The space dimension is {self.dim}, the shape of the basis is {basis.shape}." +
            f"The basis can't be inverted at X={a2s(X)}" +
            f"Basis: {basis}")

        return res

    # ---------------------------------------------------------------------------
    # Conversion: space point P to map coordinates X

    def space_to_map(self, X, V, invert=None):

        if invert is None:
            invert = self.invert_basis(X)

        try:
            R = Vector(np.einsum('ij,i', invert, V))
        except Exception as inst:
            raise RuntimeError(f"Curved space {type(self)}: space_to_map" +
            f"The space dimension is {self.dim}, the shape of the basis is {basis.shape}." +
            f"ERREUR space_to_map: X={a2s(X)} V={a2s(V)}" +
            f"Invert: {invert}")

        return R

    # ---------------------------------------------------------------------------
    # Conversion: map coordinated X to space point P

    def map_to_space(self, X, C, basis=None):
        if basis is None:
            basis = self.covariant_basis(X)

        try:
            R = Vector(np.einsum('ij,i', basis, C))
        except Exception as inst:
            raise RuntimeError(f"Curved space {type(self)}: space_to_map" +
            f"The space dimension is {self.dim}, the shape of the basis is {basis.shape}." +
            f"ERREUR space_to_map: X={a2s(X)} C={a2s(C)}" +
            f"Basis: {basis}")

        return R

    # -----------------------------------------------------------------------------------------------------------------------------
    # Christoffel symbols

    def christoffel_symbols(self, X, basis=None):

        # Prepare the array with dim^3 entries for the Christoffel symbols

        G = np.zeros((self.dim, self.dim, self.dim))

        # ---------------------------------------------------------------------------
        # From the metric tensor

        if self.by_metric:

            # Partial derivatives are computed once

            D = self.derivated_metric_tensor(X)

            # Metric tensor and its inverse

            g = self.metric_tensor(X)
            g_ = np.linalg.inv(g)

            # Compute the symbols

            E = np.zeros(self.dim)
            for k in range(self.dim):
                for l in range(self.dim):

                    # Partial derivatives sum

                    for m in range(self.dim):
                        E[m] = D[m,k, l] + D[m,l, k] - D[k,l, m]

                    # Einstein sum

                    es_i = np.einsum('im,m', g_, E)

                    for i in range(self.dim):
                        G[k,l, i] =  es_i[i]/2.


        # ---------------------------------------------------------------------------
        # From the covariant basis

        else:
            if basis is None:
                basis = self.covariant_basis(X)

            #M = np.array([basis[i]/Vector(basis[i]).length_squared for i in range(self.dim)])
            M = np.array([basis[i]/np.dot(basis[i], basis[i]) for i in range(self.dim)])

            # M = self.to_local_matrix(X, basis=basis)

            for i in range(self.dim):
                for j in range(self.dim):

                    # Partial derivative of vector i along the j direction

                    de = self.partial2(X, i, j)

                    # In the array of symbols

                    G[i,j] = M.dot(de)


        return G

    # -----------------------------------------------------------------------------------------------------------------------------
    # Parallel transport
    # X    : The map coordinates of the point
    # dX   : the map displacement of the point
    # V    : the vector to transport with this displacement in map coordinates
    # Returns the new composants of the V vector

    def parallel_transport(self, X, dX, V, basis=None):

        # If the curvature is not defined by the metric, we can try to invert the basis

        invert = None
        if self.basis_is_inversible:
            Y  = np.array(X) + np.array(dX)  # Coordonnées MAP du point d'arrivée
            if not self.by_metric:
                try:
                    invert = self.invert_basis(Y)
                except:
                    invert = None

        # ----- If not inverted matrix, we use the Christoffel symbols

        if invert is None:

            G  = self.christoffel_symbols(X, basis=basis)   # The symbols
            V  = np.array(V)                                # The vector to move
            dV = np.einsum('ijk,i,j', G, V, np.array(dX))   # Vector derivative
            W  = V - dV                                     # The new vector

        # ----- The inverted matrix exists: we can use the inversion algorithm

        else:
            VS = self.map_to_space(X, V, basis=basis)    # Space direction at point X
            W  = self.space_to_map(Y, VS, invert=invert) # Local direction at point Y

        return W

    # -----------------------------------------------------------------------------------------------------------------------------
    # Fully parameterized geodesic
    # - X           : starting map coordinates
    # - V           : direction
    # - length      : geodesic length
    # - map_length  : length is computed on map coordinates, not on surface coordinaates
    # - move_first  : move in the direction of the vector before parallel transport
    # - count       : number of points to return
    # - ds          : increment

    def geodesic(self, X, V, length, count=None, map_length=False, ds=0.01):

        X = np.array(X)
        V = np.array(V)
        V = V / np.linalg.norm(V)

        s = 0.              # Length
        G = [X]             # The geodesic
        T = [np.array(V)]   # Tangent
        P = self.f(X)

        # Loop
        for i in range(10000):

            V = V * ds
            V = self.parallel_transport(X, V, V)

            X = X + V
            G.append(X)

            Q = self.f(X)
            if map_length:
                s += np.linalg.norm(V)
            else:
                s += np.linalg.norm(Q - P)
            P = Q

            V = V / np.linalg.norm(V)
            T.append(V)

            if s >= length:
                break

        if count is None:
            return np.array(G), np.array(T)
        else:
            return from_points(count, G)[0], from_points(count, T)[0]


    # -----------------------------------------------------------------------------------------------------------------------------
    # Geodesic

    def geodesic_OLD(self, X, V, length, count=None, ds=0.01):

        X = np.array(X)
        V = np.array(V)
        V = V / np.linalg.norm(V)

        s = 0.              # Length
        G = [X]             # The geodesic
        T = [np.array(V)]   # Tangent
        P = self.f(X)

        move_before = False

        # Loop
        for i in range(10000):
            V = V * ds

            # Option 1 : X incremented after
            if move_before:
                G.append(X+V)

            V = self.parallel_transport(X, V, V)
            X = X + V

            # Option 2 : X incremented after
            if not move_before:
                G.append(X)

            Q = self.f(X)
            s += np.linalg.norm(Q - P)
            P = Q

            V = V / np.linalg.norm(V)
            T.append(V)

            if s >= length:
                break

        if count is None:
            return np.array(G), np.array(T)
        else:
            return from_points(count, G)[0], from_points(count, T)[0]

    # -----------------------------------------------------------------------------------------------------------------------------
    # Curved vector

    def curved_vector(self, X, axis, length=1., count=None, map_length=True, ds=0.01):
        V = np.zeros(self.dim, np.float)
        V[axis] = 1

        #G, T = self.geodesic(X, V, length, count)
        G, T = self.geodesic(X, V, length, map_length=map_length, count=count, ds=ds)
        return G

        #return self.f([G[:, 0], G[:, 1]])

    # -----------------------------------------------------------------------------------------------------------------------------
    # Tensor envelop

    def tensor_envelop(self, X, theta, phi_, r=0.1):

        # ----- Local metric tensor

        basis = self.covariant_basis(X)
        tm = self.metric_tensor(X, basis=basis)

        # ----- Spherical coordinates of the point on the envelop
        # Space coordinates

        if self.dim == 2:
            V = r*np.array((np.cos(theta), np.sin(theta)))
        else:
            V = r*np.array((np.sin(phi_)*np.cos(theta), np.sin(phi_)*np.sin(theta), np.cos(phi_)))

        # ----- Apply the metric tensor

        T = tm.dot(V)
        E = np.array(X) + np.array(T)

        # Additional dimension for 3D representation

        if self.dim == 2:
            E = np.array((E[0], E[1], 0.1 if phi_> pi/2 else -0.1))

        return E


# -----------------------------------------------------------------------------------------------------------------------------
# Random Surface

class RandSurface(Space):

    def __init__(self, count=10, size=400, omega=0.3, amplitude=20., damping=50, seed=0):
        Space.__init__(self, dim=2)

        rng = np.random.default_rng(seed)

        self.Os  = rng.uniform(-size/2, size/2, (count, 2))
        self.Ws  = rng.uniform(omega/10, omega, count)
        self.As  = rng.uniform(amplitude/10, amplitude, count)
        self.damping = damping

    def f(self, X):

        Xs  = np.array(X)
        zs  = np.zeros(Xs[0].shape, np.float)

        for i in range(len(self.Ws)):
            dxs = Xs[0] - self.Os[i, 0]
            dys = Xs[1] - self.Os[i, 1]
            ds  = np.sqrt(dxs*dxs + dys*dys)

            zs += self.As[i]*np.sin(self.Ws[i]*ds)*np.exp(-ds/self.damping)

        return np.array((Xs[0], Xs[1], zs)).transpose()


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Surface 3D sphère

class Sphere3(Space):

    # ---------------------------------------------------------------------------
    # Initialise en dimension 3

    def __init__(self):
        Space.__init__(self, dim=3)
        self.dX[0] = 0.01
        self.dX[1] = radians(0.5)
        self.dX[2] = radians(1.)

    # ---------------------------------------------------------------------------
    # Retourne un vecteur en coordonnées sphériques

    def f(self, X):

        Xs = np.array(X)
        if np.size(Xs) == 3:
            r = Xs[0]*sin(Xs[1])
            return np.array( (r*np.cos(Xs[2]), r*np.sin(Xs[2]), Xs[0]*np.cos(Xs[1])))

        pts = np.empty((len(Xs), 3), np.float)

        rs  = Xs[:, 0] * np.sin(Xs[:, 1])
        pts[:, 0] = rs * np.cos(Xs[:, 2])
        pts[:, 1] = rs * np.sin(Xs[:, 2])
        pts[:, 2] = Xs[:, 0]*np.cos(Xs[:, 1])

        return pts

    # ---------------------------------------------------------------------------
    # Dérivée partielle calculée

    def partial_calc(self, X, i):

        R       = X[0]
        theta   = X[1]
        phi     = X[2]

        if i == 0:
            sint = sin(theta)
            return  np.array((sint*cos(phi), sint*sin(phi), cos(theta)))

        if i == 1:
            r = R*cos(theta)
            return  np.array((r*cos(phi), r*sin(phi), -R*sin(theta)))

        if i == 2:
            r = R*sin(theta)
            return  np.array((-r*sin(phi), r*cos(phi), 0.))

        return None

    # ---------------------------------------------------------------------------
    # Dérivée partielle seconde calculée

    def partial2_calc(self, X, i, j):

        R       = X[0]
        theta   = X[1]
        phi     = X[2]

        if i == 0:
            if j == 0:
                return np.zeros(3)
            if j == 1:
                r = cos(theta)
                return  np.array((r*cos(phi), r*sin(phi), -sin(theta)))
            if j == 2:
                r = sin(theta)
                return  np.array((-r*sin(phi), r*cos(phi), 0.))

        if i == 1:
            if j == 0:
                r = cos(theta)
                return  np.array((r*cos(phi), r*sin(phi), -sin(theta)))
            if j == 1:
                r = -R*sin(theta)
                return  np.array((r*cos(phi), r*sin(phi), -R*cos(theta)))
            if j == 2:
                r = R*cos(theta)
                return  np.array((-r*sin(phi), r*cos(phi), 0.))

        if i == 2:
            if j == 0:
                r = sin(theta)
                return  np.array((-r*sin(phi), r*cos(phi), 0.))
            if j == 1:
                r = R*cos(theta)
                return  np.array((-r*sin(phi), r*cos(phi), 0.))
            if j == 2:
                r = R*sin(theta)
                return  np.array((-r*cos(phi), -r*sin(phi), 0.))

        return None






# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Surface 2D à la surface d'une sphère

class Sphere(Space):

    def __init__(self, R=1.):
        Space.__init__(self, dim=2)
        self.R = R
        self.dX[0] = radians(0.5)
        self.dX[1] = radians(1.)

    def f(self, X):
        Xs = np.array(X)
        r = self.R*np.sin(Xs[0])
        Y = np.array( (r*np.cos(Xs[1]), r*np.sin(Xs[1]), self.R*np.cos(Xs[0])))
        return np.transpose(Y)

    # ---------------------------------------------------------------------------
    # Dérivées partielles calculées
    # Utilise les formules de l'espace sphérique en supprimant l'axe 0

    def partial_calc(self, X, i):
        return Sphere3.partial_calc(self, np.array((self.R, X[0], X[1])), i+1)

    def partial2_calc(self, X, i, j):
        return Sphere3.partial2_calc(self, np.array((self.R, X[0], X[1])), i+1, j+1)


"""


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Cache d'un espace

SPACE = []
class CurvedSpace():
    def set(space):
        del SPACE[:]
        SPACE.append(space)
    def get():
        if len(SPACE) == 0:
            return None
        return SPACE[0]

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Une courbe sur une surface
# La courbe gère une liste de GeoPoints
# Class racine d'une géodésique

class Curve():

    def __init__(self, space=None):

        if space is None:
            self.space = CurvedSpace.get()
        else:
            self.space = space

        # La liste des points

        self.points = []
        self.s_min = 0.
        self.s_max = 0.

    # -----------------------------------------------------------------------------------------------------------------------------
    # Dump

    def dump(self):
        n = len(self.points)
        print("-"*100)

        print("Dump Curve avec %i points" % n)
        for i in range(n):
            if (i <= 10) or (i >= n-11):
                print("%3i> %s" % (i, self.points[i].to_string()))
            else:
                if i == 11:
                    print("...")

    # -----------------------------------------------------------------------------------------------------------------------------
    # Nombre de points

    def count(self):
        return len(self.points)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Reset

    def clear(self):
        del self.points[:]
        self.s_min = 0.
        self.s_max = 0.

    # -----------------------------------------------------------------------------------------------------------------------------
    # Ajoute un point par ses coordonnées

    def add(self, X):
        P =  self.space.f(X)
        ds = 0.

        if len(self.points) > 0:
            GP = self.points[len(self.points)-1]
            dX = Vector(X)-Vector(GP.X)

            g  = CurvedSpace.get().metric_tensor(X)
            ds = math.sqrt(abs(np.einsum('ij,i,j', g, dX, dX)))

            #print("add(%3i): s: %.3f ds:%.3f" % (len(self.points), s, ds), atm.vstr(X))

        self.s_max += ds
        self.points.append(Geo_Point(X, P, s=self.s_max))

    # -----------------------------------------------------------------------------------------------------------------------------
    # Interpole les coordonnées X d'une abscisse curviligne

    def curvi_s(self, s):

        # Restons dans les limites

        if s <= self.s_min:
            P = self.points[0].copy()
            return P
        if s >= self.s_max:
            P = self.points[-1].copy()
            return P

        # On recherche par dicchotomie

        i0 = 0
        i1 = len(self.points)-1

        while i1-i0 > 1:
            i = (i0 + i1) // 2
            ds = (s - self.points[i].s)
            if abs(ds) < 0.0000001:
                return self.points[i].copy()
            if ds < 0:
                i1 = i
            else:
                i0 = i

        P0 = self.points[i0]
        P1 = self.points[i1]

        p = (s - P0.s) / (P1.s - P0.s)
        X = (P0.X * (1. - p)) + (P1.X * p)

        GP = Geo_Point(X, self.space.f(X), s=s)
        GP.X3D = self.space.to3D(GP.X)

        return GP

    # -----------------------------------------------------------------------------------------------------------------------------
    # Réduit le nombre de points
    # Permet de consruire une géodésique précisément mais de ne garder qu'un nombre réduit de points

    def decimate(self, count):

        if count == len(self.points):
            return

        ds = (self.s_max-self.s_min) / (count-1)

        pts = []
        for i in range(count):
            pts.append(self.curvi_s(self.s_min + i*ds))

        self.points = pts

    # -----------------------------------------------------------------------------------------------------------------------------
    # Construction map : f(x) --> X
    # Construit avec une fonction f de signature f(x, param=None) --> X

    def map_build(self, f, x0=0., x1=1., param=None, steps=2000, s_max=10., decimate=100):

        del self.points[:]

        dx = (x1-x0)/(steps-1)
        for i in range(steps):
            x = x0+i*dx
            if param is None:
                X = f(x)
            else:
                X = f(x, param=param)
            self.add(X)
            if self.s_max > s_max:
                break
        self.decimate(decimate)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Construction space : f(x) --> P
    # ATTENTION: Seulement possible si la méthode inverse inv_f a été définie
    # Construit avec une fonction f de signature f(x, param=None) --> P

    def space_build(self, f, x0=0., x1=1., param=None, steps=2000, s_max=10., decimate=100):

        del self.points[:]

        if not hasattr(self.space, "inv_f"):
            print("Curve.space_build: Erreur il faut que l'espace dispose de la fonction inverse inv_f")
            return

        dx = (x1-x0)/(steps-1)
        for i in range(steps):
            x = x0+i*dx
            if param is None:
                P = f(x)
            else:
                P = f(x, param=param)
            X = self.space.inv_f(P)
            self.add(X)
            if self.s_max > s_max:
                break
        self.decimate(decimate)



# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Geodesique : une courbe initialisée avec un point de départ et une direction

class Geodesic(Curve):

    def __init__(self, X0, V, steps=1000, sMax=math.inf, decimate=100, space=None):

        # Initialisation en tant que courbe

        Curve.__init__(self, space=space)

        # ----- Controle la longueur de l'incrément géodésique

        def length_control(dX):
            dX = Vector(dX)
            return dX/(dX.length)*0.01

        # ----- Premier point

        self.add(X0)

        # ----- Il existe une fonction inverse : ce sera plus précis

        if hasattr(self.space, "inv_f"):

            P = self.space.f(X0)
            W = self.space.map_to_space(X0, length_control(V))

            for i in range(steps):
                P = P + W
                X = self.space.inv_f(P)
                self.add(X)
                if self.s_max >= sMax:
                    break

        # ----- Algorithme avec transport parallèle

        else:
            X  = Vector(X0)
            dX = length_control(V)

            for i in range(steps):
                X = X + dX
                self.add(X)
                dX = self.space.parallel_transport(X, dX, dX)
                if self.s_max >= sMax:
                    break
                dX = length_control(dX)

        # ----- Maîtrise du nombre de points

        self.decimate(decimate)



# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Cache qui gère une liste de géodésiques

GEODESICS = []
DELTA_MAP = Vector((0., 0., 0.))

class Geodesics():

    def delta_map():
        return DELTA_MAP

    def set_delta_map(v):
        for i in range(3):
            DELTA_MAP[i] = v[i]

    def clear():
        del GEODESICS[:]

    def count():
        return len(GEODESICS)

    def add(geod, decimate=None):
        if decimate is not None:
            geod.decimate(decimate)
        GEODESICS.append(geod)
        return len(GEODESICS)-1

    def append(geod, decimate=None):
        return Geodesics.add(geod, decimate=decimate)

    def get(index):
        idx = round(index)
        if idx < 0:
            idx = 0
        if idx >= Geodesics.count():
            idx = Geodesics.count()-1

        if idx < 0:
            return None

        return GEODESICS[idx]

    # -----------------------------------------------------------------------------------------------------------------------------
    # Construit une série de géodésiques entre deux coordonnées et/ou deux directions
    # - X0, X1 : l'intervalle des coordonnées à suivre
    # - V0, V1 : l'intervalle des directions à suivre

    def populate(X0, V0, X1=None, V1=None, count=11, steps=1000, sMax=math.inf, decimate=100, space=None):

        if space is None:
            space = CurvedSpace.get()

        X0 = Vector(X0)
        if X1 is None:
            X1 = Vector(X0)
        else:
            X1 = Vector(X1)

        V0 = Vector(V0)
        if V1 is None:
            V1 = Vector(V0)
        else:
            V1 = Vector(V1)

        print("----- Geodesics : population")
        for i in range(count):
            p = 1.*i/(count-1)
            X = (1.-p)*X0 + p*X1
            V = (1.-p)*V0 + p*V1
            print("   %3i>X: %s -->  V:%s" % (i, atm.vstr(X), atm.vstr(V)))
            geod = Geodesic(X, V, steps=steps, sMax=sMax, decimate=decimate, space=space)
            Geodesics.add(geod, decimate=decimate)



    # -----------------------------------------------------------------------------------------------------------------------------
    # Construit une série de géodésiques partant d'un axe et dans la direction d'un autre
    # - along           : index de l'axe source
    # - towards         : index de la direction à suivre
    # - x0, x1, count   : interval à couvrir

    def plane(along=0, towards=1, x0=0., x1=1., count=11,  steps=10000, sMax=10., decimate=100, space=None):

        if space is None:
            space = CurvedSpace.get()

        X0 = space.empty_X()
        X1 = space.empty_X()

        X0[along] = x0
        X1[along] = x1

        V = space.empty_X()
        V[towards] = 1.

        Geodesics.populate(X0, V, X1=X1, V1=V, count=count, steps=steps, sMax=sMax, decimate=decimate, space=space)




# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Une surface sphérique
# Convention phyisique : phi est la longitude (0 à 2pi) et theta la colatitude (0 à pi)

class SpaceSphere(Space):

    # ---------------------------------------------------------------------------
    # Initialise en dimension 3

    def __init__(self):
        Space.__init__(self, dim=3)
        self.dX[0] = 0.01
        self.dX[1] = radians(0.5)
        self.dX[2] = radians(1.)

    # ---------------------------------------------------------------------------
    # Retourne un vecteur en coordonnées sphériques

    def f(self, X):
        R       = X[0]
        theta   = X[1]
        phi     = X[2]

        r = R*sin(theta)
        return np.array( (r*cos(phi), r*sin(phi), R*cos(theta)))

    # ---------------------------------------------------------------------------
    # Dérivée partielle calculée

    def partial_calc(self, X, i):

        R       = X[0]
        theta   = X[1]
        phi     = X[2]

        if i == 0:
            sint = sin(theta)
            return  np.array((sint*cos(phi), sint*sin(phi), cos(theta)))

        if i == 1:
            r = R*cos(theta)
            return  np.array((r*cos(phi), r*sin(phi), -R*sin(theta)))

        if i == 2:
            r = R*sin(theta)
            return  np.array((-r*sin(phi), r*cos(phi), 0.))

        return None

    # ---------------------------------------------------------------------------
    # Dérivée partielle seconde calculée

    def partial2_calc(self, X, i, j):

        R       = X[0]
        theta   = X[1]
        phi     = X[2]

        if i == 0:
            if j == 0:
                return np.zeros(3)
            if j == 1:
                r = cos(theta)
                return  np.array((r*cos(phi), r*sin(phi), -sin(theta)))
            if j == 2:
                r = sin(theta)
                return  np.array((-r*sin(phi), r*cos(phi), 0.))

        if i == 1:
            if j == 0:
                r = cos(theta)
                return  np.array((r*cos(phi), r*sin(phi), -sin(theta)))
            if j == 1:
                r = -R*sin(theta)
                return  np.array((r*cos(phi), r*sin(phi), -R*cos(theta)))
            if j == 2:
                r = R*cos(theta)
                return  np.array((-r*sin(phi), r*cos(phi), 0.))

        if i == 2:
            if j == 0:
                r = sin(theta)
                return  np.array((-r*sin(phi), r*cos(phi), 0.))
            if j == 1:
                r = R*cos(theta)
                return  np.array((-r*sin(phi), r*cos(phi), 0.))
            if j == 2:
                r = R*sin(theta)
                return  np.array((-r*cos(phi), -r*sin(phi), 0.))

        return None

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Surface 2D à la surface d'une sphère

class SpaceSphereSurface(Space):

    def __init__(self, R=1.):
        Space.__init__(self, dim=2)
        self.R = R
        self.dX[0] = radians(0.5)
        self.dX[1] = radians(1.)

    def f(self, X):

        R       = self.R
        theta   = X[0]
        phi     = X[1]

        r = R*sin(theta)
        return np.array( (r*cos(phi), r*sin(phi), R*cos(theta)))

    # ---------------------------------------------------------------------------
    # Dérivées partielles calculées
    # Utilise les formules de l'espace sphérique en supprimant l'axe 0

    def partial_calc(self, X, i):
        return SpaceSphere.partial_calc(self, np.array((self.R, X[0], X[1])), i+1)

    def partial2_calc(self, X, i, j):
        return SpaceSphere.partial2_calc(self, np.array((self.R, X[0], X[1])), i+1, j+1)

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Surface 2D sphérique métrique

class MetricSphere2D(Space):

    def __init__(self, R=1.):
        Space.__init__(self, dim=2, by_metric=True)
        self.R = R
        self.R2 = R*R
        self.dX[0] = radians(0.5)
        self.dX[1] = radians(1.)

    # ---------------------------------------------------------------------------
    # Le tenseur métrique

    def metric_tensor(self, X, basis=None):
        cu = sin(X[0])
        return np.array([[self.R2, 0.], [0., self.R2*cu*cu]])


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Surface 2D à la surface d'un hyperboloïde

class SpaceHyperboloid1(Space):

    def __init__(self, R):
        Space.__init__(self, dim=2)
        self.R = R
        self.dX[0] = radians(0.01)
        self.dX[1] = radians(0.01)

    def f(self, X):

        R       = self.R
        z       = X[0]
        phi     = X[1]

        r = math.sqrt(R**2 + z**2)
        return np.array( (r*cos(phi), r*sin(phi), z))


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Surface 3D à la surface d'un hyperboloïde

class SpaceHyperboloid1_3D(Space):

    def __init__(self):
        Space.__init__(self, dim=3)
        self.dX[0] = radians(0.01)
        self.dX[1] = radians(0.01)

    def f(self, X):

        R       = X[0]
        z       = X[1]
        phi     = X[2]

        r = math.sqrt(R**2 + z**2)
        return np.array( (r*cos(phi), r*sin(phi), z))


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Espace sphèrique 3D à l'intérieur d'une hypersphère
# Les coordonnées sont theta1, theta2, theta3
# Le rayon de la sphere

class Space3DHypersphere(Space):

    def __init__(self, R=1.):

        Space.__init__(self, dim=3)

        self.R = R
        self.dX[0] = radians(1.0)
        self.dX[1] = radians(0.5)
        self.dX[2] = radians(0.5)

    def f(self, X):

        S = fourD.Spherical4D(self.R, X[0], X[1], X[2])
        return fourD.toCartesian4D(S)

        # ----- ANCIEN

        R     = self.R
        th1   = X[0]
        th2   = X[1]
        th3   = X[2]

        sth1 = sin(th1)
        sth2 = sin(th2)
        sth3 = sin(th3)

        return R * np.array( (cos(th1), sth1*cos(th2), sth1*sth2*cos(th3), sth1*sth2*sth3) )

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Espace cylindrique 3D généré par un espace en rotation géométrique
# On utilise les coordonnées cartésiennes

class SpaceCentrifuge(Space):

    def __init__(self, omega=1.):

        Space.__init__(self, dim=3)

        self.omega = omega

        self.dX[0] = 0.01
        self.dX[1] = 0.01
        self.dX[2] = 0.01

    # ---------------------------------------------------------------------------
    # Mouvement du référentiel dans l'espace
    #
    # rho   = sqrt(x^2 + y^2)
    # theta = atan(y/x)
    #
    # x = rho*cos(theta + wt)
    # y = rho*sin(theta + wt)

    def f(self, X):

        Vxy = Vector((X[0], X[1]))

        t     = X[2]
        rho   = Vxy.length
        theta = atan2(Vxy.y, Vxy.x)

        x = rho*cos(theta + self.omega*t)
        y = rho*sin(theta + self.omega*t)

        return np.array( (x, y, t) )

    def inv_f(self, P):
        Vxy = Vector((P[0], P[1]))

        t     = P[2]
        rho   = Vxy.length
        theta = atan2(Vxy.y, Vxy.x)

        x = rho*cos(theta - self.omega*t)
        y = rho*sin(theta - self.omega*t)

        return np.array( (x, y, t) )

    # ---------------------------------------------------------------------------
    # Dérivées partielles
    #
    # X = rho*cos(atan(y/x) + self.omega*t)
    # dX/dx = x/rho*cos() - (-y/x^2 * rho/(1+(y/x)^2)*sin() = x/rho*cos() + y/rho*sin() = (x.cos + y.sin)/rho
    # dX/dy = y/rho*cos() - (1/x * rho/(1+(y/x)^2)*sin() = y/rho*cos() - x/rho*sin()    = (y.cos - x.sin)/rho
    # dX/dt = -omega*rho*sin
    #
    # Y = rho*sin(atan(y/x) + self.omega*t)
    # dY/dx = (x.sin - y.cos)/rho
    # dY/dx = (x.cos + y.sin)/rho
    # dY/dt = omega*rho*cos
    #
    # T = t
    # dT/dx = 0
    # dT/dy = 0
    # dT/dt = 1
    #
    # Aide -------------------------
    # https://www.solumaths.com/fr/calculatrice-en-ligne/calculer/deriver
    #
    # deriver(sqrt(x^2+y^2)*cos(atan(y/x)+w*t);x)

    def partial_calc(self, X, i):

        x = X[0]
        y = X[1]
        t = X[2]

        rho   = sqrt(x*x + y*y)
        theta = atan2(y, x)

        cs = cos(theta + self.omega*t)
        sn = sin(theta + self.omega*t)

        if i == 0:
            if rho < 0.00001:
                return Vector((1000., 0., 0.))
            return Vector(((x*cs + y*sn)/rho, (x*sn - y*cs)/rho, 0.))

        if i == 1:
            if rho < 0.00001:
                return Vector((0., 1000., 0.))
            return Vector(((y*cs - x*sn)/rho, (x*cs + y*sn)/rho, 0.))

        if i == 2:
            sor = self.omega*rho
            return Vector((-sor*sn, sor*cs, 1.))

        return None

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Métrique de Newton (?)

class NewtonGravity(Space):

    def __init__(self, G=1., M=1., c=1.):
        Space.__init__(self, dim=3, by_metric=False)

        self.G = G
        self.M = M
        self.c = c

        self.dX[0] = 0.1
        self.dX[1] = 0.1
        self.dX[2] = radians(0.5)

    def f(self, X):
        r     = X[0]
        theta = X[1]
        t     = X[2]

        return Vector((t, r - self.G*self.M/r**2*(self.c*t)**2, theta))

    def metric_tensor_OLD(self, X, basis=None):

        r     = X[0]
        theta = X[1]
        t     = self.c*X[2]


        g = np.zeros((3, 3))
        g[0, 0] = r*r
        g[1, 1] = 1.
        g[2, 2] = 1.

        return g




# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Métrique de Schwarzschild

class MetricSwartzschild(Space):

    def __init__(self, G=1., M=1., c=1.):
        Space.__init__(self, dim=4, by_metric=True)

        self.G = G
        self.M = M
        self.c = c

        # Rayon de Swarzschild
        self.GM_c2 = 2.*self.G*self.M/(self.c*self.c)
        self.SR = self.GM_c2

        self.dX[0] = 0.1
        self.dX[1] = 0.1
        self.dX[2] = radians(0.5)
        self.dX[3] = radians(1.)


    def to3D(self, X):
        t     = X[0]
        R     = X[1]
        theta = X[2]
        phi   = X[3]
        return Vector((R*cos(theta), R*sin(theta), t))


    # ---------------------------------------------------------------------------
    # Points

    def f(self, X):
        t     = X[0]
        R     = X[1]
        theta = X[2]
        phi   = X[3]

        r = R*sin(theta)
        return np.array( (t, r*cos(phi), r*sin(phi), R*cos(theta)))

    # ---------------------------------------------------------------------------
    # Le tenseur métrique

    def metric_tensor(self, X, basis=None):

        t     = X[0]
        r     = X[1]
        theta = X[2]
        phi   = X[3]

        rsin = r*sin(theta)
        K = (1. - self.GM_c2/r)

        g = np.zeros((4, 4))
        g[0, 0] = -K
        g[1, 1] = 1/K
        g[2, 2] = r*r
        g[3, 3] = rsin*rsin

        return g
"""
