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
    error_title = "ERROR %"

# Default ndarray float type
ftype = np.float

# Zero
zero = 1e-6


# -----------------------------------------------------------------------------------------------------------------------------
# Get an axis

def get_axis(straxis, default=(0, 0, 1)):
    """Axis can be defined aither by a letter or a vector.

    Parameters
    ----------
    staraxis: array
        array of vector specs, ie triplets or letters: [(1, 2, 3), 'Z', (1, 2, 3), '-X']

    Returns
    -------
    array of normalized vectors

    """

    axis = np.array(straxis)
    hasstr = str(axis.dtype)[0] in ['o', '<']

    if hasstr:
        single = len(axis.shape) == 0
    else:
        single = len(axis.shape) == 1

    if single:
        axis = np.array([axis])

    rem = np.arange(len(axis))
    As  = np.zeros((len(axis), 3), ftype)

    # Axis by str
    if hasstr:

        pxs = np.concatenate((np.where(axis ==  'X')[0], np.where(axis == 'POS_X')[0]))
        pys = np.concatenate((np.where(axis ==  'Y')[0], np.where(axis == 'POS_Y')[0]))
        pzs = np.concatenate((np.where(axis ==  'Z')[0], np.where(axis == 'POS_Z')[0]))
        nxs = np.concatenate((np.where(axis == '-X')[0], np.where(axis == 'NEG_X')[0]))
        nys = np.concatenate((np.where(axis == '-Y')[0], np.where(axis == 'NEG_Y')[0]))
        nzs = np.concatenate((np.where(axis == '-Z')[0], np.where(axis == 'NEG_Z')[0]))

        As[pxs] = [ 1,  0,  0]
        As[pys] = [ 0,  1,  0]
        As[pzs] = [ 0,  0,  1]

        As[nxs] = [-1,  0,  0]
        As[nys] = [ 0, -1,  0]
        As[nzs] = [ 0,  0, -1]

        rem = np.delete(rem, np.concatenate((pxs, pys, pzs, nxs, nys, nzs)))

        with_chars = True

    else:
        # The axis can be a single vector
        # In that case the length of axis is 3 which is not
        # the number of expected vectors

        if axis.size == 3:
            As  = np.zeros(3, ftype).reshape(1, 3)
            rem = np.array([0])

        with_chars = False


    # Axis by vectors
    if len(rem > 0):

        if with_chars:
            # Didn't find better to convert np.object to np.ndarray :-()
            # Mixing letters and vectors should be rare
            for i in rem:
                As[i] = axis[i]
        else:
            As[rem] = axis

        V = As[rem]

        # Norm
        n = len(rem)
        norm = np.linalg.norm(V, axis=1)

        # zeros
        zs = np.where(norm < zero)[0]
        if len(zs) > 0:
            norm[zs] = 1.
            V[zs] = (0, 0, 1)

        # Normalize the vectors
        norm = np.resize(norm, n*3).reshape(3, n).transpose()

        As[rem] = V / norm

    # nan replaced by default
    inans = np.where(np.isnan(As))[0]
    if len(inans) > 0:
        As[inans] = np.array(default, ftype)

    # Returns a single value or an array
    if single:
        return As[0]
    else:
        return As

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

def norm(v):
    """Norm of vectors.

    Parameters
    ----------
    v: vector or array of vectors

    Return
    ------
    float or array of float
        The vectors norms
    """

    vs = np.array(v, ftype)
    return np.linalg.norm(vs, axis=len(vs.shape)-1)

# -----------------------------------------------------------------------------------------------------------------------------
# Noramlizd vectors

def normalized(v):
    """Normalize vectors.

    Parameters
    ----------
    v: vector or array of vectors

    Returns
    -------
    vector or array of vectors
        The normalized vectors
    """

    vs = np.array(v, ftype)

    if vs.shape == ():
        return 1.

    elif len(vs.shape) == 1:
        return vs / norm(v)

    elif len(vs.shape) == 2:
        count = vs.shape[0]
        size  = vs.shape[1]
        n = np.resize(norm(v), (size, count)).transpose()
        return vs/n

    raise RuntimeError(
            error_title % "normalized" +
            f"normalized error: invalid array shape {vs.shape} for vector or array of vectors.\n" +
            _str(vs, 1)
        )

# -----------------------------------------------------------------------------------------------------------------------------
# Dot product between arrays of vectors

def dot(v, w):
    """Dot product between vectors.

    Parameters
    ----------
    v: vector or array of vectors
    w: vector or array of vectors

    Returns
    -------
    float or array of floats
    """

    vs = np.array(v, ftype)
    ws = np.array(w, ftype)

    # One is a scalar
    if vs.shape == () or ws.shape == ():
        return np.dot(vs, ws)

    # First is a single vector
    if len(vs.shape) == 1:
        return np.dot(ws, vs)

    # Second is a single vector
    if len(ws.shape) == 1:
        return np.dot(vs, ws)

    # Two arrays
    v_count = vs.shape[0]
    w_count = ws.shape[0]

    # v is array with only one vector
    if v_count == 1:
        return np.dot(ws, vs[0])

    # w is array with only one vector
    if w_count == 1:
        return np.dot(vs, ws[0])

    # Error
    if v_count != w_count:
        raise RuntimeError(
            error_title % dot +
            f"Dot error: the two arrays of vectors don't have the same length: {v_count} ≠ {w_count}\n" +
            _str(vs, 1), _str(ws, 1)
            )

    return np.einsum('...i,...i', vs, ws)

# -----------------------------------------------------------------------------------------------------------------------------
# Cross product between arrays of vectors

def cross(v, w):
    """Cross product between vectors.

    Parameters
    ----------
    v: vector or array of vectors
    w: vector or array of vectors

    Returns
    -------
    vector or array of vectors
    """

    vs = np.array(v, ftype)
    ws = np.array(w, ftype)

    if (len(vs.shape) == 0) or (len(ws.shape) == 0) or \
            (len(ws.shape) > 2) or (len(ws.shape) > 2) or \
            (vs.shape[-1] != 3) or (ws.shape[-1] != 3):
        raise RuntimeError(
            error_title % "cross" +
            f"Cross error: cross product need two vectors or arrays of vectors: {vs.shape} x {vs.shape}\n" +
            _str(vs, 1), _str(ws, 1)
            )

    return np.cross(vs, ws)

# -----------------------------------------------------------------------------------------------------------------------------
# Angles between vectors

def v_angle(v, w):
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

    return np.arccos(np.maximum(-1, np.minimum(1, dot(normalized(v), normalized(w)))))


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
# Multiplication between two arrays of matrices

def m_mul(ma, mb):
    """Matrices multiplication.

    This function is intented for square matrices only.
    To multiply matrices by vectors, use m_rotate instead

    Parameters
    ----------
    ma: array(n x n) or array of array(n x n)
        The first matrix to multiply

    mb: array(n x n) or array of array(n x n)
        The second matrix to multiply

    Returns
    -------
    array(n x n) or array of array(n x n)
        The multiplication ma.mb
    """

    mas = np.array(ma, ftype)
    mbs = np.array(mb, ftype)
    if not(
        ((len(mas.shape) > 1) and (mas.shape[-2] == mas.shape[-1])) and \
        ((len(mbs.shape) > 1) and (mbs.shape[-2] == mbs.shape[-1])) and \
        (len(mas.shape) <= 3) and (len(mbs.shape) <= 3) \
        ):
        raise RuntimeError(
            error_title % "m_mul" +
            f"m_mul errors: arguments must be matrices or arrays of matrices, {mas.shape} x {mbs.shape} is not possible.\n" +
            _str(mas, 2), _str(mbs, 2)
            )

    return np.matmul(mas, mbs)

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

    ms = np.array(m, ftype)
    vs = np.array(v, ftype)

    if not(
        ((len(ms.shape) > 1) and (ms.shape[-2] == ms.shape[-1])) and \
        ((len(vs.shape) > 0) and (vs.shape[-1] == ms.shape[-1])) and \
        (len(ms.shape) <= 3) and (len(vs.shape) <= 2) \
        ):
        raise RuntimeError(
            error_title % "m_rotate" +
            f"m_rotate error: arguments must be matrices and vectors, {ms.shape} . {vs.shape} is not possible.\n" +
            _str(ms, 2), _str(vs, 1)
            )

    # ---------------------------------------------------------------------------
    # A single vector
    if len(vs.shape) == 1:
        return np.dot(ms, vs)

    if vs.shape[0] == 1:
        return np.dot(ms, vs[0])

    # ---------------------------------------------------------------------------
    # A single matrix
    if len(ms.shape) == 2:
        return np.dot(vs, ms.transpose())

    if ms.shape[0] == 1:
        return np.dot(vs, ms[0].transpose())

    # ---------------------------------------------------------------------------
    # Several matrices and severals vectors
    if len(ms) != len(vs):
        raise RuntimeError(
            error_title % "m_rotate" +
            f"m_rotate error: the length of arrays of matrices and vectors must be equal: {len(ms)} ≠ {len(vs)}\n" +
            _str(ms, 2), _str(vs, 1)
            )

    return np.einsum('...ij,...j', ms, vs)

# -----------------------------------------------------------------------------------------------------------------------------
# Transpose matrices

def m_transpose(m):
    """Transpose a matrix.

    Parameters
    ----------
    m: array(n x n) or array of array(n x n)
        The matrices to transpose

    Returns
    -------
    array(n x n) or array of array(n x n)
    """

    ms = np.array(m, ftype)

    if not(
        ((len(ms.shape) > 1) and (ms.shape[-2] == ms.shape[-1])) and \
        (len(ms.shape) <= 3) \
        ):
        raise RuntimeError(
            error_title % "m_transpose" +
            f"transpose error: argument must be a matrix or an array of matrices. Impossible to transpose shape {ms.shape}.\n" +
            _str(ms, 2)
            )

    # A single matrix
    if len(ms.shape) == 2:
        return np.transpose(ms)

    # Array of matrices
    return np.transpose(ms, (0, 2, 1))

# -----------------------------------------------------------------------------------------------------------------------------
# Invert matrices

def m_invert(m):
    """Invert a matrix.

    Parameters
    ----------
    m: array(n x n) or array of array(n x n)
        The matrices to invert

    Returns
    -------
    array(n x n) or array of array(n x n)
    """

    ms = np.array(m, ftype)

    if not(
        ((len(ms.shape) > 1) and (ms.shape[-2] == ms.shape[-1])) and \
        (len(ms.shape) <= 3) \
        ):
        raise RuntimeError(
            error_title % "m_invert" +
            f"invert error: argument must be a matrix or an array of matrices. Impossible to invert shape {ms.shape}.\n" +
            _str(ms, 2)
            )

    return np.linalg.inv(ms)

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

    #rem    = np.arange(len(ms))                           # sin(angle 0) ≠ ±1

    neg_1  = np.where(np.abs(ms[:, ls0, cs0] + 1) < zero)[0] # sin(angle 0) = -1
    pos_1  = np.where(np.abs(ms[:, ls0, cs0] - 1) < zero)[0] # sin(angle 0) = +1
    rem    = np.delete(np.arange(len(ms)), np.concatenate((neg_1, pos_1)))


    if len(neg_1) > 0:
        angles[neg_1, 0] = -pi/2 * sgn
        angles[neg_1, 1] = 0
        angles[neg_1, 2] = np.arctan2(sgn * ms[neg_1, ls3, cs3], ms[neg_1, lc3, cc3])

    if len(pos_1) > 0:
        angles[pos_1, 0] = pi/2 * sgn
        angles[pos_1, 1] = 0
        angles[pos_1, 2] = np.arctan2(sgn * ms[pos_1, ls3, cs3], ms[pos_1, lc3, cc3])

    if len(rem) > 0:
        angles[rem, 0] = sgn * np.arcsin(ms[rem, ls0, cs0])
        angles[rem, 1] = np.arctan2(-sgn * ms[rem, ls1, cs1], ms[rem, lc1, cc1])
        angles[rem, 2] = np.arctan2(-sgn * ms[rem, ls2, cs2], ms[rem, lc2, cc2])

    # ---------------------------------------------------------------------------
    # Returns the result

    if single:
        return angles[0, xyz]
    else:
        return angles[:, xyz]



# -----------------------------------------------------------------------------------------------------------------------------
# Conversion matrix to quaternion

def m_to_quat(m):
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
    ags = np.array(angle, ftype)

    if not ( ( (len(axs.shape) == 1) and (len(axs)==3) ) or ( (len(axs.shape) == 2) and (axs.shape[-1]==3)) ):
        raise RuntimeError(
            error_title % "quaternion" +
            "quaternion error: argument must be vectors(3) and angles.\n" +
            f"{_str(axs, 1)}, {_str(ags, 0)}"
            )

    # ---------------------------------------------------------------------------
    # Only one axis
    if len(axs.shape) == 1:

        # Only one angle: a single quaternion
        if len(ags.shape) == 0:
            axs *= np.sin(angle/2)
            return np.insert(axs, 0, np.cos(angle/2))

        # Several angles: create an array of axis
        axs = np.resize(axs, (len(ags), 3))

    # ---------------------------------------------------------------------------
    # Several axis but on single angle

    elif len(ags.shape) == 0:
        axs *= np.sin(angle/2)
        return np.insert(axs, 0, np.cos(angle/2), axis=1)

    # ---------------------------------------------------------------------------
    # Several axis and angles

    x_count = axs.shape[0]
    a_count = ags.shape[0]
    count = max(x_count, a_count)

    if not( (x_count in [1, count]) and (a_count in [1, count]) ):
        raise RuntimeError(
            error_title % "quaternion" +
            f"quaternion error: The length of the arrays of axis and angles are not the same: {x_count} ≠ {a_count}" +
            f"{_str(axs, 1)}, {_str(ags, 0)}"
            )

    # Adjust the lengths of the arrays
    if a_count < count:
        ags = np.resize(ags, (count, 1))

    if x_count < count:
        axs = np.resize(axs, (count, 3))

    # We can proceed

    ags /= 2
    axs *= np.resize(np.sin(ags), (3, count)).transpose()

    return np.insert(axs, 0, np.cos(ags), axis=1)


# -----------------------------------------------------------------------------------------------------------------------------
# Quaterions to axis and angles

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

    qs = np.array(q, ftype)

    if not ( ( (len(qs.shape) == 1) and (len(qs)==4) ) or ( (len(qs.shape) == 2) and (qs.shape[-1]==4)) ):
        raise RuntimeError(
            error_title % "axis_angle" +
            f"axis_angle error: argument must be quaternions, a vector(4) or and array of vectors(4), not shape {qs.shape}\n" +
            _str(qs, 1, 'quat')
            )

    if len(qs.shape) == 1:
        count = 1
        sn  = norm(qs[1:4])
        if sn < zero:
            axs = np.array((0, 0, 1), ftype)
            ags = 0.
        else:
            axs = qs[1:4] / sn
            ags = 2*np.arccos(np.maximum(-1, np.minimum(1, qs[0])))
    else:
        count = len(qs)
        sn  = norm(qs[:, 1:4])
        zs  = np.where(sn < 0)[0]
        nzs = np.delete(np.arange(len(sn)), zs)
        axs = np.empty((len(sn),3), ftype)
        ags = np.empty(len(sn), ftype)
        if len(zs) > 0:
            axs[zs] = np.array((0, 0, 1), ftype)
            ags[zs] = 0.
        if len(nzs) > 0:
            axs[nzs] = qs[nzs, 1:4] / np.resize(sn[nzs], (3, len(sn))).transpose()
            ags[nzs] = 2*np.arccos(np.maximum(-1, np.minimum(1, qs[nzs, 0])))

    if combine:
        r = np.resize(axs, (4, count)).transpose()
        r[:, 3] = ags
        return r
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

    qs = np.array(q, ftype)

    if not ( ( (len(qs.shape) == 1) and (len(qs)==4) ) or ( (len(qs.shape) == 2) and (qs.shape[-1]==4)) ):
        raise RuntimeError(
            error_title % "q_conjugate" +
            f"conjugate error: argument must be quaternions, a vector(4) or and array of vectors(4), not shape {qs.shape}\n" +
            _str(qs, 1, 'quat')
            )

    if len(qs.shape) == 1:
        qs[1:4] *= -1
    else:
        qs[:, 1:4] *= -1

    return qs

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

    qas = np.array(qa, ftype)
    qbs = np.array(qb, ftype)

    if not (
        ( (len(qas.shape) == 1) and (len(qas)==4) ) or ( (len(qas.shape) == 2) and (qas.shape[-1]==4)) and \
        ( (len(qbs.shape) == 1) and (len(qbs)==4) ) or ( (len(qbs.shape) == 2) and (qbs.shape[-1]==4))
        ):
        raise RuntimeError(
            error_title % "q_mul" +
            f"q_mul error: arguments must be quaternions or array of quaternions, impossible to compute shapes {qas.shape} x {qbs.shape}.\n" +
            f"{_str(qas, 1, 'quat')}, {_str(qbs, 1, 'quat')}"
            )

    a_count = 1 if len(qas.shape) == 1 else qas.shape[0]
    b_count = 1 if len(qbs.shape) == 1 else qbs.shape[0]

    count = max(a_count, b_count)

    if not((a_count in [1, count]) and (b_count in [1, count])):
        raise RuntimeError(
            error_title % "s_mul" +
            f"q_mul errors: the arrays of quaternions must have the same length: {a_count} ≠ {b_count}\n" +
            f"{_str(qas, 1, 'quat')}, {_str(qbs, 1, 'quat')}"
            )

    # ---------------------------------------------------------------------------
    # Resize the arrays to the same size

    if len(qas.shape) == 1:
        qas = np.resize(qa, (count, 4))
    if len(qbs.shape) == 1:
        qbs = np.resize(qb, (count, 4))

    # ---------------------------------------------------------------------------
    # No array at all, let's return a single quaternion, not an array of quaternions

    if count == 1:
        q = _q_mul(qas[0], qbs[0])
        if (len(np.array(qa).shape) == 1) and (len(np.array(qb).shape) == 1):
            return q
        else:
            return np.array([q])

    # ---------------------------------------------------------------------------
    # a = s*t - sum(p*q)
    w = qas[:, 0] * qbs[:, 0] - np.sum(qas[:, 1:4] * qbs[:, 1:4], axis=1)

    # v = s*q + t*p + np.cross(p,q)
    v  = qbs[:, 1:4] * np.resize(qas[:, 0], (3, count)).transpose() + \
         qas[:, 1:4] * np.resize(qbs[:, 0], (3, count)).transpose() + \
         np.cross(qas[:, 1:4], qbs[:, 1:4])

    # Insert w before v
    return np.insert(v, 0, w, axis=1)

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

    vs = np.array(v, ftype)
    if not ( ( (len(vs.shape) == 1) and (len(vs)==3) ) or ( (len(vs.shape) == 2) and (vs.shape[-1]==3)) ):
        raise RuntimeError(
            error_title % "q_rotate" +
            f"q_rotate error: second argument must be a vector(3) or and array of vectors(3), not shape {vs.shape}\n" +
            _str(vs, 1)
            )

    # Vector --> quaternion by inserting a 0 at position 0
    if len(vs.shape) == 1:
        vs = np.insert(vs, 0, 0)
    else:
        vs = np.insert(vs, 0, 0, axis=1)

    # Rotation by quaternion multiplication
    w = q_mul(q, q_mul(vs, q_conjugate(q)))

    # Returns quaternion or array of quaternions
    if len(w.shape)== 1:
        return np.delete(w, 0)
    else:
        return np.delete(w, 0, axis=1)

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

    m = np.matmul(m1, m2).transpose((0, 2, 1))
    return m[:, 0:3, 0:3]

# -----------------------------------------------------------------------------------------------------------------------------
# Conversion quaternion --> euler

def q_to_euler(q, order='XYZ'):
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
        m[:, 0, 1] = cz*sy*sx - sz*cx
        m[:, 0, 2] = cz*sy*cx + sz*sx
        m[:, 1, 0] = sz*cy
        m[:, 1, 1] = sz*sy*sx + cz*cx
        m[:, 1, 2] = sz*sy*cx - cz*sx
        m[:, 2, 0] = -sy
        m[:, 2, 1] = cy*sx
        m[:, 2, 2] = cy*cx

    elif order == 'XZY':
        m[:, 0, 0] = cy*cz
        m[:, 0, 1] = -cy*sz*cx + sy*sx
        m[:, 0, 2] = cy*sz*sx + sy*cx
        m[:, 1, 0] = sz
        m[:, 1, 1] = cz*cx
        m[:, 1, 2] = -cz*sx
        m[:, 2, 0] = -sy*cz
        m[:, 2, 1] = sy*sz*cx + cy*sx
        m[:, 2, 2] = -sy*sz*sx + cy*cx

    elif order == 'YXZ':
        m[:, 0, 0] = cz*cy - sz*sx*sy
        m[:, 0, 1] = -sz*cx
        m[:, 0, 2] = cz*sy + sz*sx*cy
        m[:, 1, 0] = sz*cy + cz*sx*sy
        m[:, 1, 1] = cz*cx
        m[:, 1, 2] = sz*sy - cz*sx*cy
        m[:, 2, 0] = -cx*sy
        m[:, 2, 1] = sx
        m[:, 2, 2] = cx*cy

    elif order == 'YZX':
        m[:, 0, 0] = cz*cy
        m[:, 0, 1] = -sz
        m[:, 0, 2] = cz*sy
        m[:, 1, 0] = cx*sz*cy + sx*sy
        m[:, 1, 1] = cx*cz
        m[:, 1, 2] = cx*sz*sy - sx*cy
        m[:, 2, 0] = sx*sz*cy - cx*sy
        m[:, 2, 1] = sx*cz
        m[:, 2, 2] = sx*sz*sy + cx*cy

    elif order == 'ZXY':
        m[:, 0, 0] = cy*cz + sy*sx*sz
        m[:, 0, 1] = -cy*sz + sy*sx*cz
        m[:, 0, 2] = sy*cx
        m[:, 1, 0] = cx*sz
        m[:, 1, 1] = cx*cz
        m[:, 1, 2] = -sx
        m[:, 2, 0] = -sy*cz + cy*sx*sz
        m[:, 2, 1] = sy*sz + cy*sx*cz
        m[:, 2, 2] = cy*cx

    elif order == 'ZYX':
        m[:, 0, 0] = cy*cz
        m[:, 0, 1] = -cy*sz
        m[:, 0, 2] = sy
        m[:, 1, 0] = cx*sz + sx*sy*cz
        m[:, 1, 1] = cx*cz - sx*sy*sz
        m[:, 1, 2] = -sx*cy
        m[:, 2, 0] = sx*sz - cx*sy*cz
        m[:, 2, 1] = sx*cz + cx*sy*sz
        m[:, 2, 2] = cx*cy

    if single:
        return m[0]
    else:
        return m

# -----------------------------------------------------------------------------------------------------------------------------
# Rotate a vector with an euler

def e_rotate(e, v, order='XYZ'):
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

    es = np.array(e, ftype)

    if not ( ( (len(es.shape) == 1) and (len(es)==3) ) or ( (len(es.shape) == 2) and (es.shape[-1]==3)) ):
        raise RuntimeError(
            error_title % "e_to_quat" +
            f"e_to_quat error: argument must be euler triplets, a vector(3) or and array of vectors(3), not shape {es.shape}\n" +
            _str(es, 1, 'euler')
            )

    if not order in euler_orders:
        raise RuntimeError(
            error_title % "e_to_quat" +
            f"e_to_mat error: '{order}' is not a valid code for euler order, must be in {euler_orders}")


    if len(es.shape) == 1:
        qs = [quaternion((1, 0, 0), es[0]),
              quaternion((0, 1, 0), es[1]),
              quaternion((0, 0, 1), es[2])]
    else:
        qs = [quaternion((1, 0, 0), es[:, 0]),
              quaternion((0, 1, 0), es[:, 1]),
              quaternion((0, 0, 1), es[:, 2])]

    i, j, k = euler_i[order]
    return q_mul(qs[k], q_mul(qs[j], qs[i]))


# -----------------------------------------------------------------------------------------------------------------------------
# Get a quaternion which orient a given axis towards a target direction
# Another contraint is to have the up axis oriented towards the sky
# The sky direction is the normally the Z
#
# - axis   : The axis to rotate toward the target axis
# - target : Thetarget direction for the axis
# - up     : The up direction wich must remain oriented towards the sky
# - sky    : The up direction must be rotated in the plane (target, sky)

def q_tracker(axis, target, up='Y', sky='Z', no_up = True):
    """Work in progress"""

    axs = get_axis(axis)       # Vectors to rotate
    txs = get_axis(target)     # The target direction after rotation

    # ---------------------------------------------------------------------------
    # Let's align the array lengths
    # We work on (n, 3)

    single_axis   = len(axs.shape) == 1
    single_target = len(txs.shape) == 1
    a_count       = 1 if single_axis   else len(axs)
    t_count       = 1 if single_target else len(txs)
    count         = max(a_count, t_count)

    if not ( (a_count in [1, count]) and (t_count in [1, count]) ):
        raise RuntimeError(
            error_title % "q_tracker" +
            f"q_tracker error: the arrays of axis and targets must have the same size: {a_count} ≠ {t_count}\n" +
            _str(axs, 3), _str(txs, 3)
            )

    if a_count < count:
        axs = np.resize(axs, (count, 3))
    if t_count < count:
        txs = np.resize(txs, (count, 3))

    if len(axs.shape) == 1:
        axs = np.array([axs])
    if len(txs.shape) == 1:
        txs = np.array([txs])

    # ---------------------------------------------------------------------------
    # First rotation will be made around a vector perp to  (axs, txs)

    vrot = cross(axs, txs)  # Perp vector with norm == sine
    crot = dot(axs, txs)    # Dot products = cosine
    qrot = quaternion(vrot, np.arccos(np.maximum(-1, np.minimum(1, crot))))

    # Particular cases = axis and target are aligned
    sames = np.where(abs(crot - 1) < zero)[0]
    opps  = np.where(abs(crot + 1) < zero)[0]

    # Where they are the same, null quaternion
    if len(sames) > 0:
        qrot[sames] = quaternion((0, 0, 1), 0)

    # Where they are opposite, we must rotate 180° around a perp vector
    if len(opps) > 0:
        # Let's try a rotation around the X axis
        vx = cross(axs[opps], (1, 0, 0))

        # Doesnt' work where the cross product is null
        xzs = np.where(norm(vx) < zero)[0]
        rem = np.arange(len(vx))

        # If cross product with X is null, it's where vrot == X
        # we can rotate 180° around Y
        if len(xzs) > 0:
            idx = np.arange(count)[opps][xzs]
            qrot[idx] = quaternion((0, 1, 0), pi)
            rem = np.delete(rem, xzs)

        # We can use this vector to rotate 180°
        if len(rem) > 0:
            idx = np.arange(count)[opps][rem]
            qrot[idx] = quaternion(vx, pi)

    # No up management
    if no_up:
        if single_axis and single_target:
            return qrot[0]
        else:
            return qrot


    # ---------------------------------------------------------------------------
    # This rotation places the up axis in a certain direction
    # An additional rotation around the target is required
    # to put the up axis in the plane (target, up_direction)

    upr = q_rotate(qrot, get_axis(up))

    # Projection in the plane perpendicular to the target
    J = upr - dot(upr, txs)*txs

    # We need the normalized version of this vector
    Jn = norm(J)

    # Norm can be null (when the up direction is // to the target)
    # In that case, nothing to do
    nzs = np.where(abs(Jn) > zero)[0]

    if len(nzs) > 0:

        # Normalized version of the vector to rotate
        J[nzs] /= Jn[nzs]

        # Target axis and J are two perpendicular normal vectors
        # They are considered to form the two first vector of a base
        # I = txs
        # J = normalized projection of up perpendicular to I
        # We want to rotate the J vector around I to align it along the sky axis

        # Let's compute K
        K = cross(txs[nzs], J[nzs])

        # We are interested by the components of the sky vector on J and K
        sks = get_axis(sky)
        q2  = quaternion(txs[nzs], np.arctan2(dot(sks, K), dot(sks, J[nzs])))

        qrot[nzs] = q_mul( q2, qrot[nzs])


    # Let's return a single quaternion if singles were passed

    if single_axis and single_target:
        return qrot[0]
    else:
        return qrot

# -----------------------------------------------------------------------------------------------------------------------------
# Rotate vertices around a pivot towards a direction

def rotate_towards(v, target, center=(0., 0., 0.), axis='Z', up='Y', no_up=False):

    # The rotation quaternion
    q = q_tracker(axis, target, up, no_up=no_up)

    # Vertices and centers
    vs = np.array(v, ftype)
    cs = np.array(center, ftype)

    return q_rotate(q, vs-cs) + cs

# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Transformation Matrices 4x4

def tmatrix(translation=(0., 0., 0.), mat=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)), scale=(1., 1., 1.), count=None):

    if count is None:    
        count = max(np.size(translation), np.size(mat) // 3, np.size(scale)) // 3

    # With no control : we are not idiotproof !
    ntrans = np.resize(translation, (count, 3))
    nmat   = np.resize(mat,         (count, 3, 3))
    nscale = np.resize(scale,       (count, 3))

    # Rotation mutiplied by scale
    scale_mat = np.resize(np.identity(3), (count, 3, 3))
    scale_mat[:, 0, 0] = nscale[:, 0]
    scale_mat[:, 1, 1] = nscale[:, 1]
    scale_mat[:, 2, 2] = nscale[:, 2]

    nmat = np.matmul(nmat, scale_mat)

    # Resulting array of matrices

    mats = np.resize(np.identity(4), (count, 4, 4))

    # set rotation and scale
    mats[:, :3, :3] = nmat

    # set translation
    mats[:, :3, 3] = ntrans

    # Result
    return mats

def decompose_tmat(tmat):

    scale = np.stack((
        np.linalg.norm(tmat[:, :3, 0], axis=1),
        np.linalg.norm(tmat[:, :3, 1], axis=1),
        np.linalg.norm(tmat[:, :3, 2], axis=1))).transpose()
    
    mat = np.array(tmat[:, :3, :3])
    
    invs = 1/scale
    scale_mat = np.resize(np.identity(3), (mat.shape[0], 3, 3))
    scale_mat[:, 0, 0] = invs[:, 0]
    scale_mat[:, 1, 1] = invs[:, 1]
    scale_mat[:, 2, 2] = invs[:, 2]
    
    return np.array(tmat[:, :3, 3]), np.matmul(mat, scale_mat), scale

def translation_from_tmat(tmat):
    return np.array(tmat[:, :3, 3])
    
def mat_from_tmat(tmat):
    return decompose_tmat(tmat)[1]

def scale_from_tmat(tmat):
    return decompose_tmat(tmat)[2]


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
    
    print("----- tmat1")
    print("tmat: ", np.linalg.norm(tmat1-tmat))
    print("trans:", np.linalg.norm(t1-t0))
    print("mat:  ", np.linalg.norm(m1-m0))
    print("scale:", np.linalg.norm(s1-s0))
    print('$'*10)

    t2, m2, s2 = decompose_tmat(tmat1)
    tmat2 = tmatrix(t2, m2, s2)
    
    print("----- tmat2")
    print("tmat: ", np.linalg.norm(tmat2-tmat))
    print("trans:", np.linalg.norm(t2-t0))
    print("mat:  ", np.linalg.norm(m2-m0))
    print("scale:", np.linalg.norm(s2-s0))
    print('$'*10)
    print(np.degrees(m_to_euler(m2)))

#test_tmat(1)




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

