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


from math import pi
import numpy as np

try:
    from .shapes import get_main_shape, get_full_shape, broadcast_shape
    from ..core.commons import WError
    
except:
    from shapes import get_main_shape, get_full_shape, broadcast_shape
    WError = RuntimeError

# Default ndarray float type
ftype = np.float

# Zero
zero = 1e-6

# -----------------------------------------------------------------------------------------------------------------------------
# At least as vector

def atleast_vector(v):
    if np.size(v) < 3:
        return np.resize(v, 3)
    else:
        return v

# -----------------------------------------------------------------------------------------------------------------------------
# A utility given the index of an axis and its sign

def signed_axis_index(axis):
    axs = get_axis(axis)
    i = np.argmax(abs(axs))
    return i, -1 if axs[i] < 0 else 1

# -----------------------------------------------------------------------------------------------------------------------------
# A single vector to string

def vect_str(v, fmt="8.3f", brackets=True, after=""):
    s = "[" if brackets else ""
    for r in v:
        s += format(r, fmt) + after
    return s + ("]" if brackets else "")

# -----------------------------------------------------------------------------------------------------------------------------
# A single martrix to string

def mat_str(m, fmt="8.3f", transpose=False):
    if transpose:
        m = np.transpose(m)
        before = "T"
        after  = ""
        brackets = False
        v_sepa = "|"
    else:
        m = np.array(m)
        before = "["
        after  = "]"
        brackets = True
        v_sepa = ""
        
    s = None
    for v in m[:, :]:
        sv = v_sepa + vect_str(v, fmt, brackets) + v_sepa
        if s is None:
            s = before + sv
        else:
            s += "\n " + sv
    return s + after + "\n"

# -----------------------------------------------------------------------------------------------------------------------------
# Vector pretty printing

def v_str(v, title="vector", fmt="9.3f"):
    
    v = np.array(v)
    m_shape = get_main_shape(v.shape, 3)
    
    s = f"{title} {m_shape}: "
    

    count = int(np.product(m_shape))
    if count == 1:
        return s + vect_str(v.reshape(3), fmt)
    
    s += "<\n"
    
    i = 0
    imax = 4
    for one_v in v.reshape(count, 3)[...,:]:
        s += "\t" + vect_str(one_v, fmt)
        i += 1
        if i == imax:
            if imax < count: s += "\n\t..."
            break
        else:
            s += "\n"
            
    return s + ">\n"

# -----------------------------------------------------------------------------------------------------------------------------
# Euler pretty print

def e_str(e, title="euler", degrees=True):
    
    e = np.array(e)
    m_shape = get_main_shape(e.shape, 3)
    
    if degrees:
        e = np.degrees(e)
        fmt = "7.1f"
        after = "°"
    else:
        fmt = "7.3f"
        after = ""
    
    s = f"{title} {m_shape}: "

    count = int(np.product(m_shape))
    if count == 1:
        return s + vect_str(e.reshape(3), fmt, after=after)
    
    s += "<\n"
    
    i = 0
    imax = 4
    for one_e in e.reshape(count, 3)[...,:]:
        s += "\t" + vect_str(one_e, fmt, after=after)
        i += 1
        if i == imax:
            if imax < count: s += "\n\t..."
            break
        else:
            s += "\n"
            
    return s + ">\n"

# -----------------------------------------------------------------------------------------------------------------------------
# Matrix pretty print

def m_str(m, title="matrix", fmt="9.3f", transpose=False):
    
    m = np.array(m)
    
    m_shape = get_main_shape(np.shape(m), (3,3))
    
    s = f"{title} {m_shape}: "

    count = int(np.product(m_shape))
    if count == 1:
        return s + "\n" + mat_str(m, fmt=fmt, transpose=transpose)
    
    s += "<\n"
    
    i = 0
    imax = 4
    for one_m in m.reshape(count, 3, 3)[...,:,:]:
        s += mat_str(one_m, fmt=fmt, transpose=transpose)
        i += 1
        if i == imax:
            if imax < count: s += "..."
            break
        else:
            s += "\n"
            
    return s + ">\n"


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Str array

def stra(array):
    
    isint = np.issubdtype(np.array(array).dtype, np.integer)
    return f"<array {np.shape(array)} of {'int' if isint else 'float'}>"


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Vectors geometry

# -----------------------------------------------------------------------------------------------------------------------------
# Norm of vectors

def vect_norm(v):
    """Compute the norm of an array of vectors.
    
    Can force null values to one.

    Parameters
    ----------
    v : array of vectors
        The vectors to normalized.

    Returns
    -------
    array of vectors
        The normalized vectors.
    """
    
    if len(np.shape(v)) == 0:
        return np.abs(v)
    
    else: 
        return np.linalg.norm(v, axis=-1)
    
# -----------------------------------------------------------------------------------------------------------------------------
# Mulitplication of a vector by a scalar

def scalar_mult(n, v):
    
    #return np.atleast_2d(v) * np.expand_dims(np.atleast_1d(n), axis=-1)
    
    def error():
        return f"scalar_mult: \n\tn= {stra(n)}\n\tv= {stra(v)}"
    
    main_shape = broadcast_shape(np.shape(n), get_main_shape(np.shape(v), 3), error=error)
    full_shape = get_full_shape(main_shape, 3)

    res = np.empty(full_shape, np.float)
    res[:] = v
    
    return (res * np.expand_dims(np.atleast_1d(n), axis=-1))


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
    
    main_shape = get_main_shape(np.shape(v), 3)
    full_shape = get_full_shape(main_shape, 3)
    
    count = 1 if main_shape == () else np.product(main_shape)
    
    vs = np.resize(v, (count, 3))
    ns = np.linalg.norm(vs, axis=-1)
    
    # Replace the null values by one
    nulls = np.where(ns < zero)
    ns[nulls] = 1.
    
    # Divide the vectors by the norms
    vs = vs / np.expand_dims(ns, axis=len(vs.shape)-1)
    
    # Replace the null vectors by something if required
    if null_replace is not None:
        vs[nulls] = np.array(null_replace)
        
    return vs.reshape(full_shape)

# -----------------------------------------------------------------------------------------------------------------------------
# Some randomization

def random_vectors(shape, bounds=[0, 5]):
    z = np.random.uniform(-1., 1., shape)
    r = np.sqrt(1 - z*z)
    a = np.random.uniform(0., 2*np.pi, shape)
    
    R = bounds[0]+ (1 - np.random.uniform(0, 1, shape)**3)*(bounds[1] - bounds[0])
    return np.stack((r*np.cos(a), r*np.sin(a), z), axis=-1)*np.expand_dims(R, axis=-1)

def random_v4(shape, bounds=[-3, 3]):
    return np.insert(random_vectors(shape, bounds), 0, 1, axis=-1)

def random_quaternions(shape):
    q = vect_normalize(random_vectors(shape))
    ags = (np.radians(np.random.randint(-18, 19, shape)*10)/2)
    q = q * np.expand_dims(np.sin(ags), axis=-1)
    return np.insert(q, 0, np.cos(ags), axis=-1)

def random_eulers(shape, bounds=[-180, 180]):
    return np.radians(np.random.randint(bounds[0]/10, bounds[1]/10 + 1, get_full_shape(shape, 3))*10)

def random_scales(shape, bounds=(0.1, 5)):
    return np.random.uniform(bounds[0], bounds[1], get_full_shape(shape, 3))

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
            raise WError(f"Unknwon axis spec: '{axis}'",
                Function = "get_axis",
                axis_shape = np.shape(axis),
                axis = axis)
            
    # ---------------------------------------------------------------------------
    # Axis is an array
    
    return vect_normalize(axis)


# -----------------------------------------------------------------------------------------------------------------------------
# Dot product between arrays of vectors

def vect_dot(v, w):
    """Dot products between vectors.
    
    If the shapes of the arrays are the same, dot vector per vector.
    If the shape are not the same:
        - The last dim must be the same: (m, n, 3) and (m, n, 3)
        - The shortest dim must be only one dim less than the longest and
          match the beginning of the longest: (m, n, 3) and (m, n, p, 3)

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
    
    # ----- One of the argument is a single vector
    
    if len(np.shape(v)) <= 1 or len(np.shape(w)) <= 1:
        if len(np.shape(v)) <= 1:
            return np.dot(w, v)
        else:
            return np.dot(v, w)
        
    # ----- Otherwise, shapes must match
    
    if np.shape(v) != np.shape(w):
        raise WError(f"The array of vectors don't have the same shape: {np.shape(v)} and {np.shape(w)}",
            Function = "vect_dot",
            v = v,
            w = w)
        
    # ----- Ok
    return np.einsum('...i,...i', v, w)

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
    
    # ----- One of the argument is a single vector
    
    if len(np.shape(v)) == 1 or len(np.shape(w)) == 1:
        return np.cross(v, w)
    
    # ----- Otherwise, shapes must match
    if np.shape(v) != np.shape(w):
        raise WError(f"The array of vectors don't have the same shape: {np.shape(v)} and {np.shape(w)}",
            Function = "vect_cross",
            v = v,
            w = w)
        
    return np.cross(v, w)

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
    
    return np.arccos(np.clip(vect_dot(vect_normalize(v), vect_normalize((w))), -1, 1))


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

def vect_plane_projection(v, perp):
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
    
    # Ensure we have normalized perppendicular vectors
    ps_ = get_axis(perp)
    vs_ = get_axis(v)
    
    def error():
        return f"vect_plane_projection: \n\tv= {stra(vs)}\n\tperp= {stra(ps)}"
    
    main_shape = broadcast_shape(
        get_main_shape(vs_.shape, 3), get_main_shape(ps_.shape, 3),
        error=error)
    full_shape = get_full_shape(main_shape, 3)
    
    vs = np.empty(full_shape, np.float)
    ps = np.empty(full_shape, np.float)
    
    vs[:] = vs_
    ps[:] = ps_
    
    return vs - ps*np.expand_dims(vect_dot(vs, ps), axis=-1)

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
    
    def error():
        return f"m_rotate: \n\tm= {stra(m)}\n\tv= {stra(v)}"
    
    main_shape = broadcast_shape(
        get_main_shape(np.shape(m), (3, 3)), get_main_shape(np.shape(v), 3),
        error=error)
    
    ms = np.empty(get_full_shape(main_shape, (3, 3)))
    vs = np.empty(get_full_shape(main_shape, 3))
    
    ms[:] = m
    vs[:] = v
    
    return np.einsum('...jk,...j', ms, vs)

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
    
    main_shape = get_main_shape(np.shape(m), (3, 3))
    count = 1 if main_shape == () else np.product(main_shape)
    ms = np.reshape(m, (count, 3, 3))

    # ---------------------------------------------------------------------------
    # Indices in the array to compute the angles

    if order == 'XYZ':

        # cz.cy              | cz.sy.sx - sz.cx   | cz.sy.cx + sz.sx
        # sz.cy              | sz.sy.sx + cz.cx   | sz.sy.cx - cz.sx
        # -sy                | cy.sx              | cy.cx

        xyz = [1, 0, 2]

        ls0, cs0, sgn = (2, 0, -1)          # sy
        ls1, cs1, lc1, cc1 = (2, 1, 2, 2)   # cy.sx cy.cx
        ls2, cs2, lc2, cc2 = (1, 0, 0, 0)   # cy.sz cy.cz

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
        raise WError(f"m_to_euler error: '{order}' is not a valid euler order",
            Function = "m_to_euler",
            order = order,
            m = m)
        
        
    # ---------------------------------------------------------------------------
    # Compute the euler angles

    angles = np.zeros((len(ms), 3), ftype)   # Place holder for the angles in the order of their computation
    
    # Computation depends upon sin(angle 0) == ±1

    neg_1  = np.where(np.abs(ms[:, cs0, ls0] + 1) < zero)[0] # sin(angle 0) = -1
    pos_1  = np.where(np.abs(ms[:, cs0, ls0] - 1) < zero)[0] # sin(angle 0) = +1
    rem    = np.delete(np.arange(len(ms)), np.concatenate((neg_1, pos_1)))
    
    if len(neg_1) > 0:
        angles[neg_1, xyz[0]] = -pi/2 * sgn
        angles[neg_1, xyz[1]] = 0
        angles[neg_1, xyz[2]] = np.arctan2(sgn * ms[neg_1, cs3, ls3], ms[neg_1, cc3, lc3])

    if len(pos_1) > 0:
        angles[pos_1, xyz[0]] = pi/2 * sgn
        angles[pos_1, xyz[1]] = 0
        angles[pos_1, xyz[2]] = np.arctan2(sgn * ms[pos_1, cs3, ls3], ms[pos_1, cc3, lc3])

    if len(rem) > 0:
        angles[rem, xyz[0]] = sgn * np.arcsin(ms[rem, cs0, ls0])
        angles[rem, xyz[1]] = np.arctan2(-sgn * ms[rem, cs1, ls1], ms[rem, cc1, lc1])
        angles[rem, xyz[2]] = np.arctan2(-sgn * ms[rem, cs2, ls2], ms[rem, cc2, lc2])
        
    # ---------------------------------------------------------------------------
    # At this stage, the result could be two 180 angles and a value ag
    # This is equivalent to two 0 values and 180-ag
    # Let's correct this
    
    # -180° --> 180°
        
    angles[abs(angles+np.pi) < zero] = np.pi
    
    # Let's change where we have two 180 angles
    
    idx = np.where(np.logical_and(abs(angles[:, 0]-np.pi) < zero, abs(angles[:, 1]-np.pi) < zero))[0]
    angles[idx, 0] = 0
    angles[idx, 1] = 0
    angles[idx, 2] = np.pi - angles[idx, 2]
    
    idx = np.where(np.logical_and(abs(angles[:, 0]-np.pi) < zero, abs(angles[:, 2]-np.pi) < zero))[0]
    angles[idx, 0] = 0
    angles[idx, 2] = 0
    angles[idx, 1] = np.pi - angles[idx, 1]
    
    idx = np.where(np.logical_and(abs(angles[:, 1]-np.pi) < zero, abs(angles[:, 2]-np.pi) < zero))[0]
    angles[idx, 1] = 0
    angles[idx, 2] = 0
    angles[idx, 0] = np.pi - angles[idx, 0]
    
    # ---------------------------------------------------------------------------
    # Returns the result
    
    return np.reshape(angles, get_full_shape(main_shape, 3))


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
    
    if True:
        m_shape = get_main_shape(np.shape(m), (3, 3))
        if m_shape == ():
            return m_to_quat(np.reshape(m, (1, 3, 3)))[0]
        
        count = np.product(m_shape)
        
        # ----- The result
        q = np.zeros((count, 4), np.float)
        
        # ----- Let's go
        
        m = m.reshape(count, 3, 3)
        v = 1 + m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2]
        
        # ----- Null values, need a specific computation
        nulls = np.where(v <= zero)[0]
        q[nulls, :] = e_to_quat(m_to_euler(m[nulls, :, :], 'XYZ'), 'XYZ')
        
        # ----- Not null computation
        
        oks = np.where(v > zero)[0]
        q[oks, 0] = np.sqrt(v[oks]) / 2
        
        v = q[oks, 0]*4
        q[oks, 1] = (m[oks, 1, 2] - m[oks, 2, 1]) / v
        q[oks, 2] = (m[oks, 2, 0] - m[oks, 0, 2]) / v
        q[oks, 3] = (m[oks, 0, 1] - m[oks, 1, 0]) / v
    
        return q.reshape(get_full_shape(m_shape, 4))  
    
    m_shape = get_main_shape(np.shape(m), (3, 3))

    q = np.zeros(get_full_shape(m_shape, 4))
    q[..., 0] = np.sqrt(1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]) / 2

    q4 = 4*q[..., 0]
    
    q[..., 1] = (m[..., 1, 2] - m[..., 2, 1]) / q4
    q[..., 2] = (m[..., 2, 0] - m[..., 0, 2]) / q4
    q[..., 3] = (m[..., 0, 1] - m[..., 1, 0]) / q4

    
    
    
    
    
    q = np.zeros(get_full_shape(get_main_shape(np.shape(m), (3, 3)), 4))
    
    
    q[..., 0] = np.sqrt(1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2]) / 2

    q4 = 4*q[..., 0]
    
    q[..., 1] = (m[..., 1, 2] - m[..., 2, 1]) / q4
    q[..., 2] = (m[..., 2, 0] - m[..., 0, 2]) / q4
    q[..., 3] = (m[..., 0, 1] - m[..., 1, 0]) / q4
    
    return q

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
    
    axs_ = get_axis(axis)
    
    def error():
        return f"quaternion: \n\taxis= {stra(axs_)}\n\tangle= {stra(angle)}"
    
    main_shape = broadcast_shape(
        get_main_shape(np.shape(axs_), 3), np.shape(angle),
        error=error)
    
    axs = np.empty(get_full_shape(main_shape, 3), np.float)
    ags = np.atleast_1d(np.empty(get_full_shape(main_shape, ()), np.float))
    
    axs[:] = axs_
    ags[:] = angle/2
        
    return np.insert(axs * np.expand_dims(np.sin(ags), 1), 0, np.cos(ags), axis=-1).reshape(get_full_shape(main_shape, 4))

def quaterniond(axis, angle):
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
    
    return quaternion(axis, np.radians(angle))

# -----------------------------------------------------------------------------------------------------------------------------
# Quaternions to axis and angles

def axis_angle(q):
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
    
    if get_main_shape(np.shape(q), ()) == ():
        ax, ag = axis_angle(np.reshape(q, (1, 4)))
        return ax[0], ag[0]
    
    sn = np.linalg.norm(q[..., 1:], axis=-1)
    ags = 2*np.arctan2(sn, q[..., 0])
    
    return vect_normalize(q[..., 1:], null_replace = (0, 0, 1)), ags

def axis_angled(q):
    ax, ag = axis_angle(q)
    return ax, np.degrees(ag)

# -----------------------------------------------------------------------------------------------------------------------------
# Pretty printing

def quat_str(q):
    ax, ag = axis_angle(q)
    return f"[{ax[0]:6.1f} {ax[1]:6.1f} {ax[2]:6.1f}] {np.degrees(ag):6.1f}° / {vect_str(q)}"
    
# Several quats

def q_str(q, title="quaternion"):
    
    m_shape = get_main_shape(np.shape(q), 4)
    s = f"{title} {m_shape}: <"

    count = int(np.product(m_shape))
    if count == 1:
        return s + quat_str(q.reshape(4)) + ">"
    
    s += "\n"
    
    i = 0
    imax = 4
    for one_q in q.reshape(count, 4)[...,:]:
        s += "\t" + quat_str(one_q)
        i += 1
        if i == imax:
            if imax < count: s += "\n\t..."
            break
        else:
            s += "\n"
            
    return s + ">\n"

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
    
    p = np.array(q)
    p[..., 1:] *= -1
    return p

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
    
    def error():
        return f"q_mul: \n\tqa= {stra(qa)}\n\tqb= {stra(qb)}"
    
    main_shape = broadcast_shape(
        get_main_shape(np.shape(qa), 4), get_main_shape(np.shape(qb), 4),
        error=error)
    full_shape = get_full_shape(main_shape, 4)

    qas = np.empty(full_shape, np.float)
    qbs = np.empty(full_shape, np.float)
    
    qas[:] = qa
    qbs[:] = qb
    
    # ---------------------------------------------------------------------------
    # a = s*t - sum(p*q)

    w = qas[..., 0] * qbs[..., 0] - np.sum(qas[..., 1:] * qbs[..., 1:], axis=-1)

    # v = s*q + t*p + np.cross(p,q)
    v  = qbs[..., 1:] * np.expand_dims(qas[..., 0], -1) + \
         qas[..., 1:] * np.expand_dims(qbs[..., 0], -1) + \
         np.cross(qas[..., 1:], qbs[..., 1:])
         
    return np.insert(v, 0, w, axis=-1).reshape(get_full_shape(main_shape, 4))


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
    
    vr = q_mul(q, q_mul(np.insert(v, 0, 0, axis=len(np.shape(v))-1), q_conjugate(q)))
    return vr[..., 1:]
    
    if len(vr.shape) == 1:
        return vr[1:]
    else:
        return np.delete(vr, 0, axis=len(vr.shape)-1)

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
    
    main_shape = get_main_shape(np.shape(q), 4)
    count = 1 if main_shape == () else np.product(main_shape)
    
    qs = np.reshape(q, (count, 4))
    

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
    
    return np.matmul(m1, m2)[:, :3, :3].reshape(get_full_shape(main_shape, (3, 3)))

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
    
    if not order in euler_orders:
        raise WError(f"e_to_mat error: '{order}' is not a valid code for euler order, must be in {euler_orders}",
            Function = "e_to_matrix",
            e_shape = np.shape(e),
            order = order,
            e = e)
        
    main_shape = get_main_shape(np.shape(e), 3)
    count = 1 if main_shape == () else np.product(main_shape)
    
    es = np.resize(e, (count, 3)) # Not fully idiotproof !
    
    # ----- Let's go

    m = np.zeros((count, 3, 3), np.float)

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
        
    return m.reshape(get_full_shape(main_shape, (3, 3)))


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
        raise WError(f"'{order}' is not a valid code for euler order, must be in {euler_orders}",
            Function = "e_to_quat",
            e_shape = np.shape(e),
            order = order,
            e = e)
        
    main_shape = get_main_shape(np.shape(e), 3)
    count = 1 if main_shape == () else np.product(main_shape)
    es = np.resize(e, (count, 3))
    
    qs = [quaternion((1, 0, 0), es[:, 0]),
          quaternion((0, 1, 0), es[:, 1]),
          quaternion((0, 0, 1), es[:, 2])]

    i, j, k = euler_i[order]
    return q_mul(qs[k], q_mul(qs[j], qs[i])).reshape(get_full_shape(main_shape, 4))
    

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
    
    axs_ = get_axis(axis)       # Vectors to rotate
    txs_ = get_axis(target)     # The target direction after rotation
    Z_   = get_axis(sky)        # Direction of the sky
    ups_ = get_axis(up)         # up axis
    
    def error():
        return (f"q_tracker: \n\taxis= {stra(axs_)}\n\ttarget= {stra(txs_)}" +
               f"\n\tup= {stra(ups_)}\n\tsky= {stra(Z_)}")

    main_shape = broadcast_shape(
        get_main_shape(axs_.shape, 3),
        get_main_shape(txs_.shape, 3),
        get_main_shape(Z_.shape, 3),
        get_main_shape(ups_.shape, 3),
        error = error)
    
    full_shape = get_full_shape(main_shape, 3)
    
    axs = np.empty(full_shape, np.float)
    txs = np.empty(full_shape, np.float)
    
    axs[:] = axs_
    txs[:] = txs_
    
    count = 1 if main_shape == () else np.product(main_shape)
    
    axs = axs.reshape(count, 3)
    txs = txs.reshape(count, 3)
    
    # ===========================================================================
    # First rotation
    
    # ---------------------------------------------------------------------------
    # First rotation will be made around a vector perp to  (axs, txs)
    
    perps   = np.cross(axs, txs)  # Perp vector with norm == sine
    rot_sin = np.linalg.norm(perps, axis=-1)    # Sine
    rot_cos = np.einsum('...i,...i', axs, txs) # Cosine
    
    qrot = quaternion(perps, np.arctan2(rot_sin, rot_cos))
    
    # ---------------------------------------------------------------------------
    # When the axis is already in the proper direction, it can point in the wrong way
    # Make final adjustment
    
    qrot[np.logical_and(rot_sin < zero, rot_cos < 0)] = quaternion(up, np.pi)
    
    # ---------------------------------------------------------------------------
    # No up management (for cylinders for instance)
    
    if no_up:
        return qrot.reshape(get_full_shape(main_shape, 4))
        
    # ===========================================================================
    # Second rotation around the target axis
        
    # ---------------------------------------------------------------------------
    # The first rotation places the up axis in a certain direction
    # An additional rotation around the target is required
    # to put the up axis in the plane (target, up_direction)

    # The "sky" is normally the Z direction. Let's name it Z for clarity
    # If there are only one vector, the number of sky can give the returned shape
    
    Z   = np.empty(full_shape, np.float)
    ups = np.empty(full_shape, np.float)
    
    Z[:] = Z_
    ups[:] = ups_
    
    Z   = Z.reshape(count, 3)
    ups = ups.reshape(count, 3)
    
    # Since with must rotate 'up vector' in the plane (Z, target),
    # let's compute a normalized vector perpendicular to this plane
    
    N = np.cross(Z, txs)
    nrm = np.linalg.norm(N, axis=-1)
    nrm[nrm < zero] = 1.
    N = N / np.expand_dims(nrm, axis=-1)
    
    # Let's compute where is now the up axis
    # Note that 'up axis' is supposed to be perpendicular to the axis.
    # Hence, the rotated 'up' direction is perpendicular to the plane (Z, target)
    
    rotated_up = q_rotate(qrot, ups)
    
    # Let's compute the angle betweent the 'rotated up' and the plane (Z, target)
    # The sine is the projection of the vector on the normal axis,
    # ie the scalar product
    
    r_sin = np.einsum('...i,...i', rotated_up, N)
    
    # The cosine is the projection on the plane, ie the vector minus the 
    # component along the normal.
    # Let's compute the vector first
    
    v_proj = rotated_up - N*np.expand_dims(r_sin, -1)
    
    # And the absolute value of the cosine now
    
    r_cos = np.linalg.norm(v_proj, axis=-1)
    
    # If the projected vector in the plane doesn't point in the Z direction
    # we must invert the cosine
    
    r_cos[np.einsum('...i,...i', v_proj, Z) < 0] *= -1
    
    # We have the second rotation now that we can combine with the previous one
    
    return q_mul(quaternion(txs, np.arctan2(r_sin, r_cos)), qrot).reshape(get_full_shape(main_shape, 4))

    
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
    
    main_shape = get_main_shape(np.shape(target), 3)
    txs = np.resize(target, get_full_shape(main_shape, 3))
    
    
    """
    # ---------------------------------------------------------------------------
    # Make sure we manage an array of vectors
    
    txs = get_axis(target)
    shape = list(txs.shape)
    count = max(1, np.size(txs)//3)
    txs = np.resize(txs, (count, 3))
    shape.append(3)
    """
    
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
    
    N = np.cross(txs, np.resize((0., 0., 1), get_full_shape(main_shape, 3)))
    
    # Theses vectors can be null is the target is Z
    
    N_norm = np.atleast_1d(np.linalg.norm(N, axis=-1))
    
    if main_shape == ():
        if N_norm < zero:
            N[0] = (0, 1, 0)
        else:
            N[0] / N_norm
    else:
        not_null = np.where(N_norm > zero)

        N[not_null] = N[not_null] / np.expand_dims(N_norm[not_null], axis=-1)
        
        # Let's choose Y when the the target is vertical
        N[N_norm <= zero] = (0, 1, 0)
    
    # Now, let's compute the third vector
    up_target = np.cross(N, txs)
    
    # ----- We can build the resulting matrix
    
    M = np.zeros(get_full_shape(main_shape, (3, 3)), np.float)
    M[..., ix] = txs * sx
    M[..., iy] = up_target * sy
    M[..., iz] = N * sz
    
    return M


# -----------------------------------------------------------------------------------------------------------------------------
# -----------------------------------------------------------------------------------------------------------------------------
# Transformation Matrices 4x4

def tmatrix(location=0., matrix=np.identity(3, np.float), scale=1., count=None):
    """Build an array of transformation matrices from translations, matrices and scales.
    
    A transformation matrix is a (4x4) matrix used to apply the three transformation
    on 4-vectors (3-vectors plus 1 fourth component equal to 1).
    
    mat argument must be an array of valid rotation matrices. No check is made.
    If the matrices are not normal, a unwanted scale factor will be applied.
    
    Arguments are broadcasted to fit with the length of the resulting array.
    If the argument is not specified, the max length of the orther arguments is used.

    Parameters
    ----------
    locations : array of vectors, optional
        The translation to apply. The default is (0., 0., 0.).
    matrices : array of matrices(3x3), optional
        The rotation matrices. The default is ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)).
    scales : array of vectors, optional
        The scale to apply. The default is (1., 1., 1.).
    count : int or list, optional
        The shape of the resulting array. The default is None.

    Returns
    -------
    array of (count, 4, 4) of matrices
        The transformation matrices.
    """
    
    def error():
        return (f"tmatrix: \n\tlocation= {stra(location)}" + 
                f"\n\tmatrix= {stra(matrix)}" +
                f"\n\tscale= {stra(scale)}" +
                f"\n\tcount = {None if count is None else count}")

    main_shape = broadcast_shape(
        get_main_shape(np.shape(location), 3),
        get_main_shape(np.shape(matrix), (3, 3)),
        get_main_shape(np.shape(scale), 3),
        () if count is None else count,
        error = error)
    
    if main_shape == ():
        return tmatrix(location, matrix, scale, count=1)[0]
    
    # ----- Resulting array of matrices initialized as identity
    
    mats = np.resize(np.identity(4), get_full_shape(main_shape, (4, 4)))
    
    # ----- Broadcast the matrices
    
    mats[..., :3, :3] = matrix
    
    # ----- Broadcast the scales
    
    scls = np.empty(get_full_shape(main_shape, 3))
    scls[:] = scale
    
    mats[..., 0, :3] *= np.expand_dims(scls[..., 0], -1)
    mats[..., 1, :3] *= np.expand_dims(scls[..., 1], -1)
    mats[..., 2, :3] *= np.expand_dims(scls[..., 2], -1)

    # ----- Broadcast the locations

    mats[..., 3, :3] = location
    
    # ----- Done

    return mats 

# -----------------------------------------------------------------------------------------------------------------------------
# Individual decompositions

def location_from_tmat(tmat):
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
    
    return np.array(tmat[..., 3, :3])

# -----------------------------------------------------------------------------------------------------------------------------
# Scale from tmat

def scale_from_tmat(tmat):
    """Compute the scale transformations from the transformation matrices.
    
    Parameters
    ----------
    tmat : array of (4x4) matrices
        The transformations matrices.

    Returns
    -------
    scale : array of 3-vectors
        The scale part.
    """
    
    return np.linalg.norm(tmat[...,:3, :3], axis=-1)
    
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
    
    # ----- The scales
    
    scale = np.linalg.norm(tmat[...,:3, :3], axis=-1)

    # ----- The matrices part
    
    mat = np.array(tmat[..., :3, :3])
    
    # ----- Divide by the scales
    
    # No division by 0
    nulls = np.where(scale < zero)
    scale[nulls] = 1.
    
    mat = mat / np.expand_dims(scale, -1)
    
    # Restore the nulls
    scale[nulls] = 0.
    
    # ----- Let's return the results

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

    return np.array(tmat[..., 3, :3]), m, s

    
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

# =============================================================================================================================
# Transformation
#
# The shape of the array of matrices is (shape, 4, 4)
# The shape of the array of vectors can be:
# - (n, 4)        --> (shape, n, 4)
# - (shape, n, 4) --> (shape, n, 4)
#
# In case we have one vertex per matrix, use one per on argument:
# - (shape, 4)   --> (shape, 4)

def tmat_transform4(tmat, v4, one_one=False):
    
    # ----- One vertex per matrix
    if one_one:
        m_shape = get_main_shape(tmat.shape, (4, 4))
        v_shape = get_main_shape(np.shape(v4), 4)
        
        if m_shape != v_shape:
            raise WError(f"with one_one argument = True, the sub shapes must be equal: (shape, 4, 4) and (shape, 4). Here {np.shape(tmat)} and {np.shape(v4)}.",
                Function = "tmat_transform4",
                tmat_shape = np.shape(tmat),
                v4_shape = np.shape(v4),
                one_one = one_one)
                #tmat = tmat,
                #v4 = v4)
            
        return tmat_transform4(tmat, 
                    np.reshape(v4, get_full_shape(v_shape, (1, 4))), 
                    one_one=False).reshape(get_full_shape(m_shape, 4))
    
    # ----- Only one vector or an array of vectors
    if len(np.shape(v4)) <= 2:
        return np.matmul(v4, tmat)
    
    # ----- Check the shapes compatibility
    
    try:
        return np.matmul(v4, tmat)
    except:
        raise WError(f"shapes are not compatible with multiplication Here {np.shape(tmat)} and {np.shape(v4)}.",
                Function = "tmat_transform4",
                tmat_shape = np.shape(tmat),
                v4_shape = np.shape(v4),
                one_one = one_one)
                #tmat = tmat,
                #v4 = v4)
    
    # OLD
        
    
    
    
    
    m_shape = get_main_shape(tmat.shape, (4, 4))
    v_shape = get_main_shape(v4.shape, (1, 4))   # CAUTION: (1, 4) not 4

    #m_shape = sub_shape(tmat.shape, 2)
    #v_shape = sub_shape(v4.shape, 2)
    
    if m_shape == v_shape:
        return np.matmul(v4, tmat)
    
    raise WError(f"shapes are not compatible with multiplication Here {np.shape(tmat)} and {np.shape(v4)}.",
            Function = "tmat_transform4",
            tmat_shape = np.shape(tmat),
            v4_shape = np.shape(v4),
            one_one = one_one,
            #tmat = tmat,
            v4 = v4)
    
def tmat_transform43(tmat, v4, one_one=False):
    return np.delete(tmat_transform4(tmat, v4, one_one=one_one), 3, axis=-1)

def tmat_transform(tmat, v3, one_one=False):
    return tmat_transform43(tmat, np.insert(atleast_vector(v3), 3, 1, axis=-1), one_one=one_one)

def tmat_inv_transform(tmat, v3, one_one=False):
    return tmat_transform(np.linalg.inv(tmat), atleast_vector(v3), one_one=one_one)

# ---------------------------------------------------------------------------
# Vectors transformations

def tmat_vect_transform(tmat, vect, one_one=False):
    return tmat_transform(tmat, atleast_vector(vect), one_one=one_one) - location_from_tmat(tmat)

def tmat_vect_inv_transform(tmat, vect, one_one=False):
   return tmat_inv_transform(tmat, location_from_tmat(tmat) + atleast_vector(vect), one_one=one_one) 

# ---------------------------------------------------------------------------
# Translation

def tmat_translate(tmat, translation):
    tm = np.array(tmat)
    tm[..., 3, :3] += translation
    return tm

# ---------------------------------------------------------------------------
# Composition

def tmat_compose(before, after, center=0.):
    
    center = atleast_vector(center)
    
    # ----- Quick
    if center is None:
        return np.matmul(before, after)
    
    # ---- With pivot
    loc = tmat_transform(before, center)
    return tmat_translate(np.matmul(tmat_translate(before, -loc), after), loc)




