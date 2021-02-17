from .blender import get_create_object
from .bezier import tangent, PointsInterpolation
from .meshbuilder import MeshBuilder, get_axis

# -----------------------------------------------------------------------------------------------------------------------------
# Create an arrow

def arrow(name="Arrow", axis='Z', length=1., radius=0.1, segments=12):

    v = get_axis(axis)
    def f(t):
        return v*t

    mb = MeshBuilder()
    mb.cylinder(f, t0=0., t1=length, radius=radius, steps=2, segments=segments, bot='CAP', top='ARROW')

    arrow = get_create_object(name, 'MESH')

    verts = mb.verts
    polys = mb.faces

    if len(verts) == len(arrow.wdata.verts):
        arrow.wdata.verts = verts
    else:
        arrow.wdata.new_geometry(verts, polygons=polys)

    return arrow

# -----------------------------------------------------------------------------------------------------------------------------
# Create a curved arrow

def curved_arrow(name="Curved arrow", points=None, sections=10, radius=0.1, segments=12):

    f = PointsInterpolation(points)

    mb = MeshBuilder()
    mb.cylinder(f, t0=0., t1=1., radius=radius, steps=sections, segments=segments, bot='CAP', top='ARROW')

    arrow = get_create_object(name, 'MESH')

    verts = mb.verts
    polys = mb.faces

    if len(verts) == len(arrow.wdata.verts):
        arrow.wdata.verts = verts
    else:
        arrow.wdata.new_geometry(verts, polygons=polys)

    return arrow
