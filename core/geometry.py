#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Matrix, vectors, quaternions and eulers managed in arrays

Implement computations and transformations with arrays of vectors, eulers...

Created: Jul 2020
"""

__author__     = "Alain Bernard"
__copyright__  = "Copyright 2020, Alain Bernard"
__credits__    = ["Alain Bernard"]

__license__    = "GPL"
__version__    = "1.0"
__maintainer__ = "Alain Bernard"
__email__      = "wrapanime@ligloo.net"
__status__     = "Production"


from math import pi, degrees
import numpy as np

try:
    from .commons import base_error_title
    error_title = base_error_title % "geometry.%s"
except:
    error_title = "ERROR geometry.%s: "

# Default ndarray float type
ftype = np.float

# Zero
zero = 1e-6

# -----------------------------------------------------------------------------------------------------------------------------
# Get an axis

def get_axis(axis):
    """Axis can be defined aither by a letter or a vector.

    Parameters
    ----------
    axis: array
        array of vector specs, ie triplets or letters: [(1, 2, 3), 'Z', (1, 2, 3), '-X']

    Returns
    -------
    array of normalized vectors
    """
    
    # ---------------------------------------------------------------------------
    # Axis is a str
    
    if type(axis) is str:
        
        upper = axis.upper()
        
        if upper in ['X', '+X', 'POS_X']:
            return np.array((1., 0., 0.))
        elif upper in ['Y', '+Y', 'POS_Y']:
            return np.array((0., 1., 0.))
        elif upper in ['Z', '+Z', 'POS_Z']:
            return np.array((0., 0., 1.))

        elif upper in ['-X', 'NEG_X']:
            return np.array((-1., 0., 0.))
        elif upper in ['-Y', 'NEG_Y']:
            return np.array((0., -1., 0.))
        elif upper in ['-Z', 'NEG_Z']:
            return np.array((0., 0., -1.))
        else:
            raise RuntimeError((error_title % "get_axis") +
                f"Unknwon axis spec: '{axis}'")
            
    # ---------------------------------------------------------------------------
    # Axis is an array
    
    # Number of axis
    n = max(1, np.size(axis) // 3)
    
    # An array of vectors
    axiss = np.resize(axis, (n, 3))
    
    # The length
    nrms = np.linalg.norm(axiss, axis=1)
    
    # Remove zeros
    axiss[nrms<zero] = (0, 0, 1)
    nrms[nrms<zero] = 1
    
    # Normalization
    axiss = axiss / np.expand_dims(nrms, 1)
    
    if n == 1 and len(np.shape(axis)) < 2:
        return axiss[0]
    else:
        return axiss
    
    
# -----------------------------------------------------------------------------------------------------------------------------
# A utility given the index of an axis and its sign

def signed_axis_index(axis):
    axs = get_axis(axis)
    i = np.argmax(abs(axs))
    return i, -1 if axs[i] < 0 else 1


# -----------------------------------------------------------------------------------------------------------------------------
# Dump the content of an array

def _str(array, dim=1, vtype='scalar'):

    if array is None:
        return "[None array]"

    if dim is None:
        return f"{array}"

    array = np.array(array)

    if array.size == 0:
        return f"[Empty array of shape {np.array(array).shape}]"

    def scalar(val):
        if vtype.lower() in ['euler', 'degrees']:
            return f"{degrees(val):6.1f}°"
        else:
            return f"{val:7.2f}"

    def vector(vec):
        s = ""

        if vtype.lower() in ['quat', 'quaternion']:
            if len(vec) != 4:
                s = "!quat: "
            else:
                ax, ag = axis_angle(vec)
                return f"<{vector(ax)} {degrees(ag):6.1f}°>"

        lmax = 5
        if len(vec) <= 2*lmax:
            lmax = len(vec)

        for i in range(lmax):
            s += " " + scalar(vec[i])

        if len(vec) > lmax:
            s += " ..."
            for i in reversed(range(lmax)):
                s += " " + scalar(vec[-1-i])

        return "[" + s[1:] + "]"

    def matrix(mat, prof=""):
        s = ""
        sep = "["
        for vec in mat:
            s += sep + vector(vec)
            sep = "\n " + prof
        return s + "]"


    def arrayof():
        lmax = 10
        if len(array) <= 2*lmax:
            lmax = len(array)

        s = ""
        sep = "\n["
        for i in range(lmax):
            if dim == 1:
                sep = "\n[" if i == 0 else "\n "
                s += sep + vector(array[i])
            else:
                sep = "\n[" if i == 0 else "\n "
                s += sep + matrix(array[i], prof=" ")

        if len(array) > lmax:
            s += f"\n ... total={len(array)}"

            for i in reversed(range(lmax)):
                if dim == 1:
                    s += sep + vector(array[-1-i])
                else:
                    s += sep + matrix(array[-1-i], prof=" ")
        return s + "]"


    if dim == 0:
        if len(array.shape == 0):
            return scalar(array)
        elif len(array.shape) == 1:
            return vector(array)
        else:
            return f"<Not an array of scalars>\n{array}"

    elif dim == 1:
        if len(array.shape) == 0:
            return f"[Scalar {scalar(array)}, not a vector]"
        elif len(array.shape) == 1:
            return vector(array)
        elif len(array.shape) == 2:
            return arrayof()
        else:
            return f"<Not an array of vectors>\n{array}"

    elif dim == 2:
        if len(array.shape) < 2:
            return f"<Not a matrix>\n{array}"
        elif len(array.shape) == 2:
            return matrix(array)
        elif len(array.shape) == 3:
            return arrayof()
        else:
            return f"<Not an array of matrices>\n{array}"

    else:
        return f"[array of shape {array.shape} for object of dim {dim}]\n{array}"


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Vectors geometry

# -----------------------------------------------------------------------------------------------------------------------------
# Norm of vectors

def vect_norm(v, null_to_one=False):
    """Compute the norm of an array of vectors.
    
    Can force null values to one.

    Parameters
    ----------
    v : array of vectors
        The vectors to normalized.
    null_to_one : bool, optional
        Set to 1 the null values. The default is False.

    Returns
    -------
    array of vectors
        The normalized vectors.
    """
    
    dim = np.shape(v)[-1]
    count = max(1, np.size(v) // dim)
    
    # ---------------------------------------------------------------------------
    # Only one vector
    
    if count == 1 and len(np.shape(v)) <= 1:
        n = np.linalg.norm(v)
        if null_to_one and n < zero:
            return 1.
        else:
            return n

    # ---------------------------------------------------------------------------
    # Several ones

    vs = np.resize(v, (count, dim))
    if null_to_one:
        nrms = np.linalg.norm(vs, axis=1)
        nrms[nrms < zero] = 1
        return nrms
    else:
        return np.linalg.norm(vs, axis=1)
    
# -----------------------------------------------------------------------------------------------------------------------------
# Noramlizd vectors

def vect_normalize(v, null_replace=None):
    """Normalize and array of vector.
    
    Null vectors are replaced by null_replace if not None.

    Parameters
    ----------
    v : array of vectors
        DESCRIPTION.
    nulls : vector, optional
        The replacement of null vectors if not None. The default is None.

    Returns
    -------
    Array of vectors.
        The normalized vectors.
    """
    
    dim = np.shape(v)[-1]
    count = max(1, np.size(v) // dim)
    
    # ---------------------------------------------------------------------------
    # Only one vector
    
    if count == 1 and len(np.shape(v)) <= 1:
        n = np.linalg.norm(v)
        if n <= zero:
            return np.array(v, np.float) if null_replace is None else np.array(null_replace, np.float)
        else:
            return np.array(v) / n
        
    # ---------------------------------------------------------------------------
    # Several ones

    vs = np.resize(v, (count, dim))
    nrms = np.linalg.norm(vs, axis=1)
    null = nrms < zero
    
    nrms[null] = 1
    vs = vs / np.expand_dims(nrms, axis=1)
    
    if null_replace is not None:
        vs[null] = np.array(null_replace, np.float)
        
    return vs



# -----------------------------------------------------------------------------------------------------------------------------
# Dot product between arrays of vectors

def vect_dot(v, w):
    """Dot products between vectors.
    
    if v (w) is a single vector, dot all along w (v).
    if both v and w are array of vectors, they must have the same length. The dot is computed all
    along the two arrays

    Parameters
    ----------
    v : array of vectors
        The first array of vectors.
    w : array of vectors
        The second array of vectors.

    Raises
    ------
    RuntimeError
        If shapes are not correct.

    Returns
    -------
    array of vectors
        The dot products.
    """
    
    # ----- Make sure we have arrays
    vs = np.array(v, ftype)
    ws = np.array(w, ftype)
    
    # ----- One is a scalar
    if vs.shape == () or ws.shape == ():
        return np.dot(vs, ws)

    # ----- First is a single vector
    if len(vs.shape) == 1:
        return np.dot(ws, vs)

    # ----- Second is a single vector
    if len(ws.shape) == 1:
        return np.dot(vs, ws)
    
    # ----- No more that two dimensions
    if len(vs.shape) > 2 or len(ws.shape) > 2:
        raise RuntimeError(
            error_title % "vect_dot" +
            f"The function only applies on two arrays of vectors, not on {vs.shape} dot {ws.shape}"
            )

    # ----- The number of vectors to dot
    v_count = vs.shape[0]
    w_count = ws.shape[0]

    # ----- v is array with only one vector
    if v_count == 1:
        return np.dot(ws, vs[0])

    # ----- w is array with only one vector
    if w_count == 1:
        return np.dot(vs, ws[0])

    # ----- The array must have the same size
    if v_count != w_count:
        raise RuntimeError(
            error_title % "vect_dot" +
            f"The two arrays of vectors don't have the same length: {v_count} ≠ {w_count}\n"
            )

    return np.einsum('...i,...i', vs, ws)

# -----------------------------------------------------------------------------------------------------------------------------
# Cross product between arrays of vectors

def vect_cross(v, w):
    """Cross product between vectors.

    Parameters
    ----------
    v: vector or array of vectors
    w: vector or array of vectors

    Returns
    -------
    vector or array of vectors
    """
    
    v_count = np.size(v) // 3
    w_count = np.size(w) // 3
    
    # ----- Make sure we have arrays
    if not ((v_count == 1) or (w_count == 1) or (v_count == w_count)):
        raise RuntimeError(
            error_title % "vect_cross" +
            f"cross product needs two arrays of vectors of the same length, not: {np.shape(v)} x {np.shape(w)}\n"
            )
        
    if v_count == 1 and w_count == 1 and len(np.shape(v)) == 1 and len(np.shape(w)) == 1:
        return np.cross(v, w)
    
    return np.cross(np.reshape(v, (v_count, 3)), np.reshape(w, (w_count, 3)))

# -----------------------------------------------------------------------------------------------------------------------------
# Angles between vectors

def vect_angle(v, w):
    """Angles between vectors.

    Parameters
    ----------
    v: vector or array of vectors
    w: vector or array of vectors

    Returns
    -------
    float or array of float
        The angle between the vectors
    """
    
    return np.arccos(vect_dot(vect_normalize(v), vect_normalize((w))))

# -----------------------------------------------------------------------------------------------------------------------------
# Plane (ie vector perppendicular to the two vectors)

def vect_perpendicular(v, w, null_replace=(0, 0, 1)):
    """Compute a normalized vector perpendicular to a couple of vectors.

    Parameters
    ----------
    v : array of vectors
        First vectors.
    w : array of vectors
        Second vectors.
    null_replace : vector, optional
        The value to use when the tw o vectors are colinear. The default is (0, 0, 1).

    Returns
    -------
    array of vectors
        The normalized vectors perpendicular to the given vectors.
    """
    
    return vect_normalize(vect_cross(v, w), null_replace=null_replace)

# -----------------------------------------------------------------------------------------------------------------------------
# Plane (ie vector perpendicular to the two vectors)

def vect_plane_projection(v, perps):
    """Projection of a vector on to a plane.
    
    The plane is either specified by its perpendicular vector or by two vectors

    Parameters
    ----------
    v : array of vectors
        The vectors to project.
    perps : array of normalized vectors, optional
        Planes definition by normalized peprpendiculars. The default is None.

    Returns
    -------
    Array of vectors.
        The projections of vectors in the planes
    """
    
    perps = get_axis(perps)
    count = max(np.size(v) // 3, len(perps))

    vs = np.resize(v, (count, 3))
    vs = vs - perps*np.expand_dims(vect_dot(vs, perps), axis=1)
    
    if count == 1 and len(np.shape(v)) == 1:
        return vs[0]
    else:
        return vs

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Matrix geometry

# -----------------------------------------------------------------------------------------------------------------------------
# Rotation matrix

def matrix(axis, angle):
    """Create matrices from direction and angle.

    Parmeters
    ---------
    axis: array of axis specifications. triplets or letters X, Y, Z or -X, -Y, -Z
        The axis around which to turn

    angle: float or array of floats
        The angle to rotate around the axis

    Return
    ------
    array (3x3) or array of array(3x3)
    """

    return q_to_matrix(quaternion(axis, angle))


# -----------------------------------------------------------------------------------------------------------------------------
# Dot product between matrices and vectors

def m_rotate(m, v):
    """Vector rotation by a matrix.

    Parameters
    ----------
    m: array (n x n) or array of array(n x n)
        The rotation matrices
    v: array(n) or array of array (n)

    Returns
    -------
    array(n) or array of array(n)
    """
    
    nm = np.size(m) // 9
    nv = np.size(v) // 3
    
    if nm == 1 and nv > 1:
        return m_rotate(np.resize(m, (nv, 3, 3)), v)
    
    if nm != nv:
        raise RuntimeError(
            error_title % "m_rotate" +
            f"The number of matrices must match the number of vectors to rotate: matrices =  {np.shape(m)} vectors = {np.shape(v)}"
            )
    
    ms = np.resize(m, (nv, 3, 3))
    vs = np.resize(v, (nv, 3))
    
    return np.einsum('...jk,...j', ms, vs).reshape(np.shape(v))

# -----------------------------------------------------------------------------------------------------------------------------
# Convert a matrix to euler
# The conversion depends upon the order

def m_to_euler(m, order='XYZ'):
    """Transform matrices to euler triplets.

    Parameters
    ----------
    m: array(3 x 3) or array or array(3 x 3)
        The matrices
    order: str
        A valid order in euler_orders

    Returns
    -------
    array(3) or array of array(3)
        The euler triplets
    """

    ms = np.array(m, ftype)

    if not(
        ((len(ms.shape) > 1) and (ms.shape[-2] == 3) and (ms.shape[-2] == 3) ) and \
        (len(ms.shape) <= 3) \
        ):
        raise RuntimeError(
            error_title % "m_to_euler" +
            f"m_to_euler error: argument must be a matrix(3x3) or an array of matrices. Impossible to convert shape {ms.shape}.\n" +
            _str(ms, 2)
            )

    single = len(ms.shape) == 2
    if single:
        ms = np.reshape(ms, (1, 3, 3))

    # ---------------------------------------------------------------------------
    # Indices in the array to compute the angles

    if order == 'XYZ':

        # cz.cy              | cz.sy.sx - sz.cx   | cz.sy.cx + sz.sx
        # sz.cy              | sz.sy.sx + cz.cx   | sz.sy.cx - cz.sx
        # -sy                | cy.sx              | cy.cx

        xyz = [1, 0, 2]

        ls0, cs0, sgn = (2, 0, -1)
        ls1, cs1, lc1, cc1 = (2, 1, 2, 2)
        ls2, cs2, lc2, cc2 = (1, 0, 0, 0)

        ls3, cs3, lc3, cc3 = (0, 1, 1, 1)

    elif order == 'XZY':

        # cy.cz              | -cy.sz.cx + sy.sx  | cy.sz.sx + sy.cx
        # sz                 | cz.cx              | -cz.sx
        # -sy.cz             | sy.sz.cx + cy.sx   | -sy.sz.sx + cy.cx

        xyz = [1, 2, 0]

        ls0, cs0, sgn = (1, 0, +1)
        ls1, cs1, lc1, cc1 = (1, 2, 1, 1)
        ls2, cs2, lc2, cc2 = (2, 0, 0, 0)

        ls3, cs3, lc3, cc3 = (0, 2, 2, 2)

    elif order == 'YXZ':

        # cz.cy - sz.sx.sy   | -sz.cx             | cz.sy + sz.sx.cy
        # sz.cy + cz.sx.sy   | cz.cx              | sz.sy - cz.sx.cy
        # -cx.sy             | sx                 | cx.cy

        xyz = [0, 1, 2]

        ls0, cs0, sgn = (2, 1, +1)
        ls1, cs1, lc1, cc1 = (2, 0, 2, 2)
        ls2, cs2, lc2, cc2 = (0, 1, 1, 1)

        ls3, cs3, lc3, cc3 = (1, 0, 0, 0)

    elif order == 'YZX':

        # cz.cy              | -sz                | cz.sy
        # cx.sz.cy + sx.sy   | cx.cz              | cx.sz.sy - sx.cy
        # sx.sz.cy - cx.sy   | sx.cz              | sx.sz.sy + cx.cy

        xyz = [2, 1, 0]

        ls0, cs0, sgn = (0, 1, -1)
        ls1, cs1, lc1, cc1 = (0, 2, 0, 0)
        ls2, cs2, lc2, cc2 = (2, 1, 1, 1)

        ls3, cs3, lc3, cc3 = (1, 2, 2, 2)

    elif order == 'ZXY':

        # cy.cz + sy.sx.sz   | -cy.sz + sy.sx.cz  | sy.cx
        # cx.sz              | cx.cz              | -sx
        # -sy.cz + cy.sx.sz  | sy.sz + cy.sx.cz   | cy.cx

        xyz = [0, 2, 1]

        ls0, cs0, sgn = (1, 2, -1)
        ls1, cs1, lc1, cc1 = (1, 0, 1, 1)
        ls2, cs2, lc2, cc2 = (0, 2, 2, 2)

        ls3, cs3, lc3, cc3 = (2, 0, 0, 0)

    elif order == 'ZYX':

        # cy.cz              | -cy.sz             | sy
        # cx.sz + sx.sy.cz   | cx.cz - sx.sy.sz   | -sx.cy
        # sx.sz - cx.sy.cz   | sx.cz + cx.sy.sz   | cx.cy

        xyz = [2, 0, 1]

        ls0, cs0, sgn = (0, 2, +1)
        ls1, cs1, lc1, cc1 = (0, 1, 0, 0)
        ls2, cs2, lc2, cc2 = (1, 2, 2, 2)

        ls3, cs3, lc3, cc3 = (2, 1, 1, 1)

    else:
        raise RuntimeError(
            error_title % "m_to_euler" +
            f"m_to_euler error: '{order}' is not a valid euler order")

    # ---------------------------------------------------------------------------
    # Compute the euler angles

    angles = np.zeros((len(ms), 3), ftype)   # Place holder for the angles in the order of their computation
    
    # Computation depends upoin sin(angle 0) == ±1

    neg_1  = np.where(np.abs(ms[:, cs0, ls0] + 1) < zero)[0] # sin(angle 0) = -1
    pos_1  = np.where(np.abs(ms[:, cs0, ls0] - 1) < zero)[0] # sin(angle 0) = +1
    rem    = np.delete(np.arange(len(ms)), np.concatenate((neg_1, pos_1)))


    if len(neg_1) > 0:
        angles[neg_1, 0] = -pi/2 * sgn
        angles[neg_1, 1] = 0
        angles[neg_1, 2] = np.arctan2(sgn * ms[neg_1, cs3, ls3], ms[neg_1, cc3, lc3])

    if len(pos_1) > 0:
        angles[pos_1, 0] = pi/2 * sgn
        angles[pos_1, 1] = 0
        angles[pos_1, 2] = np.arctan2(sgn * ms[pos_1, cs3, ls3], ms[pos_1, cc3, lc3])

    if len(rem) > 0:
        angles[rem, 0] = sgn * np.arcsin(ms[rem, cs0, ls0])
        angles[rem, 1] = np.arctan2(-sgn * ms[rem, cs1, ls1], ms[rem, cc1, lc1])
        angles[rem, 2] = np.arctan2(-sgn * ms[rem, cs2, ls2], ms[rem, cc2, lc2])

    # ---------------------------------------------------------------------------
    # Returns the result

    if single:
        return angles[0, xyz]
    else:
        return angles[:, xyz]

# -----------------------------------------------------------------------------------------------------------------------------
# Conversion matrix to quaternion

def m_to_quat(m):
    """Convert a matrix to a quaternion.

    Parameters
    ----------
    m : array of matrices
        The matrices to convert.

    Returns
    -------
    array of quaternions
        The converted quaternions.

    """
    order = 'XYZ'
    return e_to_quat(m_to_euler(m, order), order)

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Quaternion geometry

def quaternion(axis, angle):
    """Initialize quaternions from axis and angles

    Parameters
    ----------
    axis: array of axis, 3-vectors or letters
        The axis compenent of the quaternions
    angle: float or array of floats
        The angle component of the quaternions

    Returns
    -------
    array(4) of array or array(4)
        The requested quaternions
    """

    axs = get_axis(axis)
    count = max(np.size(axis)//3, np.size(angle))
    
    ags = np.resize(angle, count)/2
    axs = np.insert(
        np.resize(axs,(count, 3))*np.expand_dims(np.sin(ags), 1), 0, np.cos(ags), axis=1)
    
    if count == 1 and len(np.shape(axis)) == 1:
        return axs[0]
    else:
        return axs

# -----------------------------------------------------------------------------------------------------------------------------
# Quaternions to axis and angles

def axis_angle(q, combine=False):
    """Return the axis and angles components of a quaternion.

    Parameters
    ----------
    q: array(4) or array of array(4)
        The quaternions

    Returns
    -------
    array of array(3), array of float
        The axis and angles of the quaternions
    """
    
    count = np.size(q)//4
    qs = np.resize(q, (count, 4))
    
    sn = np.linalg.norm(qs[:, 1:], axis=1)
    ags = 2*np.arctan2(sn, qs[:, 0])
    sn[sn<zero] = 1
    
    axs = qs[:, 1:] / np.expand_dims(sn, 1)
    
    if len(np.shape(q)) == 1:
        return axs[0], ags[0]
    else:
        return axs, ags
    

# -----------------------------------------------------------------------------------------------------------------------------
# Quaternion conjugate

def q_conjugate(q):
    """Compute the conjugate of a quaternion.

    Parameters
    ----------
    q: array(4) or array of array(4)
        The quaternion to conjugate

    Returns
    -------
    array(4) or array of array(4)
        The quaternions conjugates
    """
    
    count = np.size(q)//4
    qs = np.resize(q, (count, 4))
    
    qs[:, 1:] *= -1
    
    return qs.reshape(np.shape(q))

# -----------------------------------------------------------------------------------------------------------------------------
# Two quaternions multiplication
# Used to check the numpy algorithms

def _q_mul(qa, qb):
    """Utility: one, one quaternion multiplication."""

    a = qa[0]
    b = qa[1]
    c = qa[2]
    d = qa[3]

    e = qb[0]
    f = qb[1]
    g = qb[2]
    h = qb[3]

    coeff_1 = a*e - b*f - c*g - d*h
    coeff_i = a*f + b*e + c*h - d*g
    coeff_j = a*g - b*h + c*e + d*f
    coeff_k = a*h + b*g - c*f + d*e

    return np.array([coeff_1, coeff_i, coeff_j, coeff_k])


# -----------------------------------------------------------------------------------------------------------------------------
# Two quaternions multiplication
# numpy algorithm

def _np_q_mul(qa, qb):
    """Utility: one, one quaternion multiplication, numpy version."""

    s = qa[0]
    p = qa[1:4]
    t = qb[0]
    q = qb[1:4]

    a = s*t - sum(p*q)
    v = s*q + t*p + np.cross(p,q)

    return np.array((a, v[0], v[1], v[2]))

# -----------------------------------------------------------------------------------------------------------------------------
# Quaternions multiplications

def q_mul(qa, qb):
    """Quaternions multiplication.

    Parameters
    ----------
    qa: array(4) or array of array(4)
        The first quaternion to multiply
    qb: array(4) or array of array(4)
        The second quaternion to multiply

    Returns
    -------
    array(4) or array of array(4)
        The results of the multiplications: qa x qb
    """
    
    count = max(np.size(qa)//4, np.size(qb)//4)
    qas = np.resize(qa, (count, 4))
    qbs = np.resize(qb, (count, 4))
    
    # ---------------------------------------------------------------------------
    # a = s*t - sum(p*q)

    w = qas[:, 0] * qbs[:, 0] - np.sum(qas[:, 1:] * qbs[:, 1:], axis=1)

    # v = s*q + t*p + np.cross(p,q)
    v  = qbs[:, 1:] * np.expand_dims(qas[:, 0], 1) + \
         qas[:, 1:] * np.expand_dims(qbs[:, 0], 1) + \
         np.cross(qas[:, 1:], qbs[:, 1:])

    qs = np.insert(v, 0, w, axis=1)
    
    # Insert w before v
    if np.shape(qa) == np.shape(qb):
        return qs.reshape(np.shape(qa))
    else:
        if count == 1 and len(np.shape(qa)) == 1:
            return qs[0]
        else:
            return qs


# -----------------------------------------------------------------------------------------------------------------------------
# Quaternion rotation

def q_rotate(q, v):
    """Rotate a vector with a quaternion.

    Parameters
    ----------
    q: array(4) or array of array(4)
        The rotation quaternions
    v: array(3) or array of array(3)
        The vectors to rotate

    Returns
    -------
    array(3) or array of array(3)
    """
    
    nq = np.size(q) // 4
    nv = np.size(v) // 3
    if nq == 1 and nv > 1:
        return q_rotate(np.resize(q, (nv, 4)), v)
    
    if nq != nv:
        raise RuntimeError(
            error_title % "q_rotate" +
            f"The number of quaternions must be either 1 or match the number fo vertices: quaternions = {np.shape(q)} and vectors : {np.shape(v)}"
            )
    
    qs = np.resize(q, (nv, 4))
 
    # Vector --> quaternion by inserting a 0 at position 0
    vs = np.insert(np.resize(v, (nv, 3)), 0, 0, axis=1)
    
    # Vector rotation
    return q_mul(qs, q_mul(vs, q_conjugate(qs)))[:, 1:].reshape(np.shape(v))
    

# -----------------------------------------------------------------------------------------------------------------------------
# Quaternion to matrix

def q_to_matrix(q):
    """Transform quaternions to matrices.

    Parameters
    ----------
    q: array(4) or array of array(4)
        The quaternions to transform

    Returns
    -------
    array(3 x 3) or array of array(3 x 3)
    """

    qs = np.array(q, ftype)

    if not ( ( (len(qs.shape) == 1) and (len(qs)==4) ) or ( (len(qs.shape) == 2) and (qs.shape[-1]==4)) ):
        raise RuntimeError(
            error_title % "q_to_matrix" +
            f"q_to_matrix error: argument must be quaternions, a vector(4) or and array of vectors(4), not shape {qs.shape}\n" +
            _str(qs, 1, True)
            )
    # m1
    # +w	 +z -y +x
    # -z +w +x +y
    # +y	 -x +w +z
    # -x -y -z +w

    # m2
    # +w	 +z -y -x
    # -z +w +x -y
    # +y	 -x +w -z
    # +x +y +z +w

    # ---------------------------------------------------------------------------
    # Only one quaternion

    if len(qs.shape)==1:
        m1 = np.stack((
            qs[[0, 3, 2, 1]]*(+1, +1, -1, +1),
            qs[[3, 0, 1, 2]]*(-1, +1, +1, +1),
            qs[[2, 1, 0, 3]]*(+1, -1, +1, +1),
            qs[[1, 2, 3, 0]]*(-1, -1, -1, +1)
            ))

        m2 = np.stack((
            qs[[0, 3, 2, 1]]*(+1, +1, -1, -1),
            qs[[3, 0, 1, 2]]*(-1, +1, +1, -1),
            qs[[2, 1, 0, 3]]*(+1, -1, +1, -1),
            qs[[1, 2, 3, 0]]*(+1, +1, +1, +1)
            ))

        m = np.matmul(m1, m2).transpose()
        return m[0:3, 0:3]

    # ---------------------------------------------------------------------------
    # The same with an array of quaternions

    m1 = np.stack((
        qs[:, [0, 3, 2, 1]]*(+1, +1, -1, +1),
        qs[:, [3, 0, 1, 2]]*(-1, +1, +1, +1),
        qs[:, [2, 1, 0, 3]]*(+1, -1, +1, +1),
        qs[:, [1, 2, 3, 0]]*(-1, -1, -1, +1)
        )).transpose((1, 0, 2))

    m2 = np.stack((
        qs[:, [0, 3, 2, 1]]*(+1, +1, -1, -1),
        qs[:, [3, 0, 1, 2]]*(-1, +1, +1, -1),
        qs[:, [2, 1, 0, 3]]*(+1, -1, +1, -1),
        qs[:, [1, 2, 3, 0]]*(+1, +1, +1, +1)
        )).transpose((1, 0, 2))
    
    return np.matmul(m1, m2)[:, :3, :3]

# -----------------------------------------------------------------------------------------------------------------------------
# Conversion quaternion --> euler

def q_to_euler(q, order='XYZ'):
    """Convert a quaternion to an euler, with the specified order.

    Parameters
    ----------
    q : array of quaternions
        The quaternions to convert.
    order : str, optional
        The order of the resulting eulers. The default is 'XYZ'.

    Returns
    -------
    array of eulers
        The converted eulers.
    """
    
    return m_to_euler(q_to_matrix(q), order)

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Euler

euler_orders = ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX']
euler_i = {
        'XYZ': [0, 1, 2],
        'XZY': [0, 2, 1],
        'YXZ': [1, 0, 2],
        'YZX': [1, 2, 0],
        'ZXY': [2, 0, 1],
        'ZYX': [2, 1, 0],
    }

# -----------------------------------------------------------------------------------------------------------------------------
# Convert euler to a rotation matrix

def e_to_matrix(e, order='XYZ'):
    """Transform euler triplets to matrices

    Parameters
    ----------
    e: array(3) or array or array(3)
        The eulers triplets
    order: str
        A valid order in euler_orders

    Returns
    -------
    array(3 x 3) or array of array(3 x 3)
    """

    es = np.array(e, ftype)

    if not ( ( (len(es.shape) == 1) and (len(es)==3) ) or ( (len(es.shape) == 2) and (es.shape[-1]==3)) ):
        raise RuntimeError(
            error_title % "e_to_matrix" +
            f"e_to_mat error: argument must be euler triplets, a vector(3) or and array of vectors(3), not shape {es.shape}\n" +
            _str(es, 1, 'euler')
            )

    if not order in euler_orders:
        raise RuntimeError(
            error_title % "e_to_matrix" +
            f"e_to_mat error: '{order}' is not a valid code for euler order, must be in {euler_orders}")

    single = len(es.shape) == 1
    if single:
        es = np.reshape(es, (1, 3))

    m = np.zeros((len(es), 3, 3), ftype)

    cx = np.cos(es[:, 0])
    sx = np.sin(es[:, 0])
    cy = np.cos(es[:, 1])
    sy = np.sin(es[:, 1])
    cz = np.cos(es[:, 2])
    sz = np.sin(es[:, 2])
    
    if order == 'XYZ':
        m[:, 0, 0] = cz*cy
        m[:, 1, 0] = cz*sy*sx - sz*cx
        m[:, 2, 0] = cz*sy*cx + sz*sx
        m[:, 0, 1] = sz*cy
        m[:, 1, 1] = sz*sy*sx + cz*cx
        m[:, 2, 1] = sz*sy*cx - cz*sx
        m[:, 0, 2] = -sy
        m[:, 1, 2] = cy*sx
        m[:, 2, 2] = cy*cx

    elif order == 'XZY':
        m[:, 0, 0] = cy*cz
        m[:, 1, 0] = -cy*sz*cx + sy*sx
        m[:, 2, 0] = cy*sz*sx + sy*cx
        m[:, 0, 1] = sz
        m[:, 1, 1] = cz*cx
        m[:, 2, 1] = -cz*sx
        m[:, 0, 2] = -sy*cz
        m[:, 1, 2] = sy*sz*cx + cy*sx
        m[:, 2, 2] = -sy*sz*sx + cy*cx

    elif order == 'YXZ':
        m[:, 0, 0] = cz*cy - sz*sx*sy
        m[:, 1, 0] = -sz*cx
        m[:, 2, 0] = cz*sy + sz*sx*cy
        m[:, 0, 1] = sz*cy + cz*sx*sy
        m[:, 1, 1] = cz*cx
        m[:, 2, 1] = sz*sy - cz*sx*cy
        m[:, 0, 2] = -cx*sy
        m[:, 1, 2] = sx
        m[:, 2, 2] = cx*cy

    elif order == 'YZX':
        m[:, 0, 0] = cz*cy
        m[:, 1, 0] = -sz
        m[:, 2, 0] = cz*sy
        m[:, 0, 1] = cx*sz*cy + sx*sy
        m[:, 1, 1] = cx*cz
        m[:, 2, 1] = cx*sz*sy - sx*cy
        m[:, 0, 2] = sx*sz*cy - cx*sy
        m[:, 1, 2] = sx*cz
        m[:, 2, 2] = sx*sz*sy + cx*cy

    elif order == 'ZXY':
        m[:, 0, 0] = cy*cz + sy*sx*sz
        m[:, 1, 0] = -cy*sz + sy*sx*cz
        m[:, 2, 0] = sy*cx
        m[:, 0, 1] = cx*sz
        m[:, 1, 1] = cx*cz
        m[:, 2, 1] = -sx
        m[:, 0, 2] = -sy*cz + cy*sx*sz
        m[:, 1, 2] = sy*sz + cy*sx*cz
        m[:, 2, 2] = cy*cx

    elif order == 'ZYX':
        m[:, 0, 0] = cy*cz
        m[:, 1, 0] = -cy*sz
        m[:, 2, 0] = sy
        m[:, 0, 1] = cx*sz + sx*sy*cz
        m[:, 1, 1] = cx*cz - sx*sy*sz
        m[:, 2, 1] = -sx*cy
        m[:, 0, 2] = sx*sz - cx*sy*cz
        m[:, 1, 2] = sx*cz + cx*sy*sz
        m[:, 2, 2] = cx*cy

    if single:
        return m[0]
    else:
        return m


# -----------------------------------------------------------------------------------------------------------------------------
# Rotate a vector with an euler

def e_rotate(e, v, order='XYZ'):
    """Rotate a vector with an euler according the specified order.

    Parameters
    ----------
    e : array of eulers
        The rotation eulers.
    v : array of vectors
        The vectors to rotate.
    order : str, optional
        The order of the euler rotation. The default is 'XYZ'.

    Returns
    -------
    array of vectors
        The rotated vectors.
    """
    
    return m_rotate(e_to_matrix(e, order), v)

# -----------------------------------------------------------------------------------------------------------------------------
# Convert euler to a quaternion

def e_to_quat(e, order='XYZ'):
    """Transform euler triplets to quaternions.

    Parameters
    ----------
    e: array(3) or array or array(3)
        The eulers triplets
    order: str
        A valid order in euler_orders

    Returns
    -------
    array(4) or array of array(4)
        The quaternions
    """
    
    # ----- The order must be valid !
    if not order in euler_orders:
        raise RuntimeError(
            error_title % "e_to_quat" +
            f"e_to_mat error: '{order}' is not a valid code for euler order, must be in {euler_orders}")
    

    # ----- Ensure an array of triplets    
    count = np.size(e)//3
    es = np.resize(e, (count, 3))
    
    qs = [quaternion((1, 0, 0), es[:, 0]),
          quaternion((0, 1, 0), es[:, 1]),
          quaternion((0, 0, 1), es[:, 2])]

    i, j, k = euler_i[order]
    return q_mul(qs[k], q_mul(qs[j], qs[i]))

# -----------------------------------------------------------------------------------------------------------------------------
# Get a quaternion which orients a given axis towards a target direction.
# Another contraint is to have the up axis oriented towards the sky
# The sky direction is the normally the Z
#
# - axis   : The axis to rotate toward the target axis
# - target : Thetarget direction for the axis
# - up     : The up direction wich must remain oriented towards the sky
# - sky    : The up direction must be rotated in the plane (target, sky)

def q_tracker(axis, target, up='Y', sky='Z', no_up=False):
    """Compute a quaternion which rotates an axis towards a target.
    
    The rotation is computed using a complementary axis named 'up' which
    must be oriented upwards.
    The upwards direction is Z by default and can be overriden by the argument 'sky'.
    
    After rotation:
        - 'axis' points towards 'target'.
        - 'up' points such as 'up' cross 'target' is perpendicular to vertical axis.
        - 'sky' is used to replace the 'Z' direction.
    
    This algorithm is more general than m_tracker since axis, up can be any vectors.
    The vertical is also a parameter, nor necessarily the vertical axis.

    Parameters
    ----------
    axis : vector
        The axis to orient.
    target : vector
        The direction the axis must be oriented towards.
    up : vector, optional
        The axis which must be oriented upwards. The default is 'Y'.
    sky : vector, optional
        The direction of the sky, i.e. the upwards direction. The default is 'Z'.
    no_up : bool, optional
        Don't rotate around the target axis. The default is True.

    Raises
    ------
    RuntimeError
        If array lengths are not compatible.

    Returns
    -------
    array of quaternions
        The quaternions that can be used to rotate the axis according the arguments.
    """
    
    axs = get_axis(axis)       # Vectors to rotate
    txs = get_axis(target)     # The target direction after rotation
    
    # ---------------------------------------------------------------------------
    # Make sure we have the proper arrays to work on
    
    count = max(np.size(axs)//3, np.size(txs)//3)
    single = False
    if count == 1:
        single = (len(axs.shape) == 1) and (len(txs.shape) == 1)
    
    axs   = np.resize(axs, (count, 3))
    txs   = np.resize(txs, (count, 3))
    
    # ===========================================================================
    # First rotation
    
    # ---------------------------------------------------------------------------
    # First rotation will be made around a vector perp to  (axs, txs)
    
    perps   = np.cross(axs, txs)  # Perp vector with norm == sine
    rot_sin = np.linalg.norm(perps, axis=1)    # Sine
    rot_cos = np.einsum('...i,...i', axs, txs) # Cosine

    qrot = quaternion(perps, np.arctan2(rot_sin, rot_cos))
    
    # ---------------------------------------------------------------------------
    # When the axis is already in the proper direction, it can point in the wrong way
    # Make final adjustment
    
    qrot[np.logical_and(rot_sin < zero, rot_cos < 0)] = quaternion(up, np.pi)
    
    # ---------------------------------------------------------------------------
    # No up management (for cylinders for instance)

    if no_up:
        if single:
            return qrot[0]
        else:
            return qrot
        
    # ===========================================================================
    # Second rotation around the target axis
        
    # ---------------------------------------------------------------------------
    # The first rotation places the up axis in a certain direction
    # An additional rotation around the target is required
    # to put the up axis in the plane (target, up_direction)

    # The "sky" is normally the Z direction. Let's name it Z for clarity
    
    Z = np.resize(get_axis(sky), (count, 3))
    
    # Since with must rotate 'up vector' in the plane (Z, target),
    # let's compute a normalized vector perpendicular to this plane
    
    N = np.cross(Z, txs)
    nrm = np.linalg.norm(N, axis=1)
    nrm[nrm < zero] = 1.
    N = N / np.expand_dims(nrm, axis=1)
    
    # Let's compute where is now the up axis
    # Note that 'up axis' is supposed to be perpendicular to the axis.
    # Hence, the rotated 'up' direction is perpendicular to the plane (Z, target)
    
    rotated_up = q_rotate(qrot, np.resize(get_axis(up), (count, 3)))
    
    # Let's compute the angle betweent the 'rotated up' and the plane (Z, target)
    # The sine is the projection of the vector on the normal axis,
    # ie the scalar product
    
    r_sin = np.einsum('...i,...i', rotated_up, N)
    
    # The cosine is the projection on the plane, ie the vector minus the 
    # component along the normal.
    # Let's compute the vector first
    
    v_proj = rotated_up - N*np.expand_dims(r_sin, 1)
    
    # And the absolute value of the cosine now
    
    r_cos = np.linalg.norm(v_proj, axis=1)
    
    # If the projected vector in the plane doesn't point in the Z direction
    # we must invert the cosine
    
    r_cos[np.einsum('...i,...i', v_proj, Z) < 0] *= -1
    
    # We have the second rotation now that we can combine with the previous one
    
    qrot = q_mul(quaternion(txs, np.arctan2(r_sin, r_cos)), qrot)
    
    # Result

    if single:
        return qrot[0]
    else:
        return qrot
    
# -----------------------------------------------------------------------------------------------------------------------------
# Get a matrix which orients a given axis towards a target direction.
# Another contraint is to have the up axis oriented towards positive Z.
#
# - axis   : The axis to rotate toward the target axis
# - target : Thetarget direction for the axis
# - up     : The up direction wich must remain oriented towards the sky
   
def m_tracker(axis, target, up='Y'):
    """Compute a matrice which rotates an axis towards a target.
    
    The rotation is computed using a complementary axis named 'up' which
    must be oriented upwards.
    
    After rotation:
        - 'axis' points towards 'target'.
        - 'up' points such as 'up' cross 'target' is perpendicular to vertical axis.
        - 'sky' is used to replace the 'Z' direction.
    
    This algorithm is less general than q_tracker since axis, up must be one of the
    three axis, or the opposite.

    Parameters
    ----------
    axis : vector
        The axis to orient.
    target : vector
        The direction the axis must be oriented towards.
    up : vector, optional
        The axis which must be oriented upwards. The default is 'Y'.

    Returns
    -------
    array of matrices
        The matrices that can be used to rotate the axis according the arguments.
    """
    
    # ---------------------------------------------------------------------------
    # Make sure we manage an array of vectors
    
    count = max(1, np.size(target)//3)
    txs = np.resize(get_axis(target), (count, 3))
    
    # ---------------------------------------------------------------------------
    # For the sake of clarity, X is the name of the axis to point towards target,
    # Y is the name of the up axis and Z is the third axis
    
    # Each axis is taken in the positive direction plus an additional var sx, sy, sz
    # to keep the sign
    
    # ----- X : the axis to orient towards target
    
    X = get_axis(axis)
    ix = np.argmax(np.abs(X))
    sx = 1 if X[ix] > 0 else -1
    X = np.zeros(3, np.float)
    X[ix] = 1
    
    # ----- Y : the axis to up axis
    
    Y = get_axis(up)
    if abs(np.dot(X, Y)) > zero:
        for y in 'XYZ':
            Y = get_axis(y)
            if abs(np.dot(X, Y)) < zero:
                break
    iy = np.argmax(np.abs(Y))
    sy = 1 if Y[iy] > 0 else -1
    Y = np.zeros(3, np.float)
    Y[iy] = 1
            
    # ----- Z : the third axis
    
    # Get the third index in (1, 2, 3) with possible negative value
    sz = 1
    iz = np.array(((0, 3, -2), (-3, 0, 1), (2, -1, 0)))[ix, iy]
    if iz < 0:
        iz = -iz-1
        sz = -1
    else:
        iz -= 1
    Z = np.zeros(3, np.float)
    Z[iz] = 1
    
    # ---------------------------------------------------------------------------
    # The resulting matrix is composed of three normalized vectors:
    # ix --> target
    # iz --> perpendicular to the vertical plane containing the target
    # iy --> perpendicular to the two previous ones
    
    # Let's compute the perpendicular to the vertical plane
    # This vector is the rotated direction of Z. Hence the cross order is:
    # X' (x) Y' --> Z'
    
    N = np.cross(txs, np.resize((0., 0., 11), (count, 3)))
    
    # Theses vectors can be null is the target is Z
    
    N_norm = np.linalg.norm(N, axis=1)
    not_null = np.where(N_norm > zero)
    N[not_null] = N[not_null] / np.expand_dims(N_norm[not_null], axis=1)
    
    # Let's choose Y when the the target is vertical
    N[N_norm <= zero] = (0, 1, 0)
    
    # Now, let's compute the third vector
    up_target = np.cross(N, txs)
    
    # ----- We can build the resulting matrix
    
    M = np.zeros((count, 3, 3), np.float)
    M[:, ix] = txs * sx
    M[:, iy] = up_target * sy
    M[:, iz] = N * sz
    
    if count == 1 and len(np.shape(target)) <= 1:
        return M[0]
    else:
        return M

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Transformation Matrices 4x4

def tmatrix(translation=(0., 0., 0.), mat=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), scale=(1., 1., 1.), count=None):
    """Build an array of transformation matrices from translations, matrices and scales.
    
    A transformation matrix is a (4x4) matrix used to apply the three transformation
    on 4-vectors (3-vectors plus 1 fourth component equal to 1).
    
    mat argument must be an array of valid rotation matrices. No check is made.
    If the matrices are not normal, a unwanted scale factor will be applied.
    
    Arguments are broadcasted to fit with the length of the resulting array.
    If the argument is not specified, the max length of the orther arguments is used.

    Parameters
    ----------
    translation : array of vectors, optional
        The translation to apply. The default is (0., 0., 0.).
    mat : array of matrices(3x3), optional
        The rotation matrices. The default is ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)).
    scale : array of vectors, optional
        The scale to apply. The default is (1., 1., 1.).
    count : int, optional
        The length of the resulting array. The default is None.

    Returns
    -------
    array of (4x4) matrices
        The transformation matrices.
    """

    if count is None:    
        count = max(np.size(translation), np.size(mat) // 3, np.size(scale)) // 3

    # Make sure the arguments are at the correct shape
    ntrans = np.resize(translation, (count, 3))
    nmat   = np.resize(mat,         (count, 3, 3))

    if len(np.shape(scale)) == 0:
        scale = np.resize(scale, 3)
    elif len(np.shape(scale)) == 1 and len(scale) == count:
        scale = np.resize(scale, (3, count)).transpose()
        
    nscale = np.resize(scale, (count, 3))
    
    nmat[:, 0, :3] *= np.expand_dims(nscale[:, 0], 1)
    nmat[:, 1, :3] *= np.expand_dims(nscale[:, 1], 1)
    nmat[:, 2, :3] *= np.expand_dims(nscale[:, 2], 1)
    
    """

    # Rotation mutiplied by scale
    scale_mat = np.resize(np.identity(3), (count, 3, 3))
    scale_mat[:, 0, 0] = nscale[:, 0]
    scale_mat[:, 1, 1] = nscale[:, 1]
    scale_mat[:, 2, 2] = nscale[:, 2]

    nmat = np.matmul(scale_mat, nmat)
    """

    # Resulting array of matrices

    mats = np.resize(np.identity(4), (count, 4, 4))

    # set rotation and scale
    mats[:, :3, :3] = nmat

    # set translation
    mats[:, 3, :3] = ntrans

    # Result
    return mats

    
# -----------------------------------------------------------------------------------------------------------------------------
# Mat scale decomposition

def mat_scale_from_tmat(tmat):
    """Compute the rotation and scale transformations from the transformation matrices.
    
    The transformations are returned back in the UI Blender order:
        - Rotation
        - Scale

    Parameters
    ----------
    tmat : array of (4x4) matrices
        The transformations matrices.

    Returns
    -------
    array of (3x3) matrices
        The rotation part.
    scale : array of 3-vectors
        The scale part.
    """
    
    scale = np.stack((
        np.linalg.norm(tmat[:, 0, :3], axis=1),
        np.linalg.norm(tmat[:, 1, :3], axis=1),
        np.linalg.norm(tmat[:, 2, :3], axis=1))).transpose()
    
    mat = np.array(tmat[:, :3, :3])
    
    mat[:, 0, :3] = mat[:, 0, :3] / np.expand_dims(scale[:, 0], 1)
    mat[:, 1, :3] = mat[:, 1, :3] / np.expand_dims(scale[:, 1], 1)
    mat[:, 2, :3] = mat[:, 2, :3] / np.expand_dims(scale[:, 2], 1)
    
    return mat, scale

# -----------------------------------------------------------------------------------------------------------------------------
# Full decomposition

def decompose_tmat(tmat):
    """Compute the three unitary transformations from the transformation matrices.
    
    The transformations are returned back in the UI Blender order:
        - Translation
        - Rotation
        - Scale

    Parameters
    ----------
    tmat : array of (4x4) matrices
        The transformations matrices.

    Returns
    -------
    array of 3-vectors
        The translation part.
    array of (3x3) matrices
        The rotation part.
    scale : array of 3-vectors
        The scale part.
    """

    m, s = mat_scale_from_tmat(tmat)

    return np.array(tmat[:, 3, :3]), m, s
    

# -----------------------------------------------------------------------------------------------------------------------------
# Individual decompositions

def translation_from_tmat(tmat):
    """Get the translation part of transformation matrices.

    Parameters
    ----------
    tmat : array of (4x4) matrices
        The transformations matrices.

    Returns
    -------
    array of 3-vectors
        The rotation part.
    """
    
    return np.array(tmat[:, 3, :3])
    
    
def mat_from_tmat(tmat):
    """Get the rotation part of transformation matrices.

    Parameters
    ----------
    tmat : array of (4x4) matrices
        The transformations matrices.

    Returns
    -------
    array of (3x3) matrices
        The rotations.
    """
    
    return mat_scale_from_tmat(tmat)[0]

def scale_from_tmat(tmat):
    """Get the scale part of transformation matrices.

    Parameters
    ----------
    tmat : array of (4x4) matrices
        The transformations matrices.

    Returns
    -------
    array of 3-vectors
        The scales.
    """
    
    return mat_scale_from_tmat(tmat)[1]













# -----------------------------------------------------------------------------------------------------------------------------
# A test

def test_tmat(test=0, count=2):
    if test==0:
        t0 = np.random.uniform(0, 10, (count, 3))
        m0 = e_to_matrix(np.random.uniform(0, 7, (count, 3)))
        s0 = np.random.uniform(2, 3, (count, 3))
    else:
        t0 = np.random.uniform(0, 10, (count, 3))
        m0 = e_to_matrix(np.radians(np.resize((10, 70, -30), (count, 3))))
        s0 = (0.3, 0.3, 1)
        
    tmat = tmatrix(t0, m0, s0)
    
    t1, m1, s1 = decompose_tmat(tmat)
    tmat1 = tmatrix(t1, m1, s1)
    
    def sn(n):
        return f"{n:.10f}"
    
    print("----- tmat1")
    print("tmat: ", sn(np.linalg.norm(tmat1-tmat)))
    print("trans:", sn(np.linalg.norm(t1-t0)))
    print("mat:  ", sn(np.linalg.norm(m1-m0)))
    print("scale:", sn(np.linalg.norm(s1-s0)))
    print('.'*10)

    t2, m2, s2 = decompose_tmat(tmat1)
    tmat2 = tmatrix(t2, m2, s2)
    
    print("----- tmat2")
    print("tmat: ", sn(np.linalg.norm(tmat2-tmat)))
    print("trans:", sn(np.linalg.norm(t2-t0)))
    print("mat:  ", sn(np.linalg.norm(m2-m0)))
    print("scale:", sn(np.linalg.norm(s2-s0)))
    print('.'*10)

    print(np.degrees(m_to_euler(m2)))



# ///////////////////////////////////////////////////////////////////////////
# WORK IN PROGRESS


def tmatrix_euler(translation=(0., 0., 0.), euler=(0., 0., 0.), order='XYZ', scale=(1., 1., 1.)):
    return tmatrix(translation, e_to_matrix(euler, order), scale)

def tmatrix_quat(translation=(0., 0., 0.),quat=(1., 0., 0.,0. ), scale=(1., 1., 1.)):
    return tmatrix(translation, q_to_matrix(quat), scale)

def mul_tmatrices(tmat1, tmat2):
    count = max(np.size(tmat1), np.size(tmat2)) // 16

    m1 = np.resize(tmat1, (count, 4, 4))
    m2 = np.resize(tmat2, (count, 4, 4))
    return np.matmul(m1, m2)


def tmat_translation(translation):
    return tmatrix(translation=translation)

def tmat_scale(scale):
    return tmatrix(scale=scale)

def tmat_rotation(mat):
    return tmatrix(mat=mat)

def tmat_rotation_euler(euler, order='XYZ'):
    return tmatrix_euler(euler=euler, order=order)

def tmat_rotation_quat(quat):
    return tmatrix_quat(quat=quat)

def transform(tmat, vectors, falloff=1.):

    # Number fo vectors to transform
    count = np.size(vectors) // 3

    # tmat to the right number of matrices
    mats = np.resize(tmat, (count, 4, 4))

    # To 4-vectors
    v4 = np.ones((count, 4), np.float)
    v4[:, :3] = vectors

    # Apply the transformation matrices
    v4t = np.einsum('ijk,ik->ij', mats, v4)

    # Apply the falloff between the two arrays
    if np.size(falloff) == 1:
        vs = v4t*falloff + v4*(1. - falloff)
    else:
        fo = np.array(falloff)[:, np.newaxis]
        vs = v4t*fo + v4*(1. - fo)


    # Return 3-vectprs
    return vs[:, :3]

def translate(translation, vectors, falloff=1.):
    return transform(tmat_translation(translation), vectors, falloff)

def scale(scale, vectors, falloff=1.):
    return transform(tmat_scale(scale), vectors, falloff)

def rotate_euler(euler, order, vectors, falloff=1.):
    return transform(tmat_rotation_euler(euler, order), vectors, falloff)

def rotate_quat(quat, vectors, falloff=1.):
    return transform(tmat_rotation_quat(quat), vectors, falloff)

def rotate(mat, vectors, falloff=1.):
    return transform(tmat_rotation(mat), vectors, falloff)

