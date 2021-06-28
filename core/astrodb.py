# -----------------------------------------------------------------------------------------------------------------------------
# Paramètres des planètes
# https://www.windows2universe.org/our_solar_system/planets_orbits_table.html

# 1: Semimajor Axis (AU)
# 2: Orbital Period (yr)
# 3: Orbital Speed (km/s)
# 4: Orbital Eccentricity (e)
# 5: Inclination of Orbit to Ecliptic (°)
# 6; Rotation Period (days)
# 7: Inclination of Equator to Orbit (°)

# Planet    1       2       3       4       5       6       7
# Mercury	0.3871	0.2408	47.9	0.206	7.00	58.65	0
# Venus	    0.7233	0.6152	35.0	0.007	3.39   -243.01*	177.3
# Earth	    1.000	1	    29.8	0.017	0.00	0.997	23.4
# Mars	    1.5273	1.8809	24.1	0.093	1.85	1.026	25.2
# Jupiter	5.2028	11.862	13.1	0.048	1.31	0.410	3.1
# Saturn	9.5388	29.458	9.6	    0.056	2.49	0.426	26.7
# Uranus	19.1914	84.01	6.8	    0.046	0.77   -0.746*	97.9
# Neptune	30.0611	164.79	5.4     0.010	1.77	0.718	29.6


from math import radians, pi, sqrt
from mathutils import Euler

import numpy as np

pi2 = 2*pi

AU_km        = 149597887.5
AU_sim       = 30.

EARTH_RADIUS = 6378.137 #km

PERIOD_ZERO  = 1000000
ONE_YEAR     = 365.25
SIDERAL_DAY  = 365.25/366.25


PLANETS = {
    "SUN": {
        "name"          : "Sun",
        "rotation_p"    : PERIOD_ZERO,
        "radius"        : 108.,
        "obliquity"     : 0.,
        "inclination"   : 0.,
        "eccentricity"  : 0.,
        "semi_major"    : 0.,
        "apoapsis_or"   : 0.,
        "orbital_p"     : PERIOD_ZERO,
        "orb_speed"     : 0.,
        },
    "MERCURY": {
        "name"          : "Mercury",
        "rotation_p"    : 58.65,
        "radius"        : 0.382,
        "obliquity"     : radians(0.04),
        "inclination"   : radians(7.00),
        "eccentricity"  : 0.206,
        "semi_major"    : 0.387,
        "apoapsis_or"   : 0.,
        "orbital_p"     : ONE_YEAR * 0.2408,
        "orb_speed"     : 47.9,
        },
    "VENUS": {
        "name"          : "Venus",
        "rotation_p"    : -243.01,
        "radius"        : 0.949,
        "obliquity"     : radians(177.36),
        "inclination"   : radians(3.39),
        "eccentricity"  : 0.007,
        "semi_major"    : 0.723,
        "apoapsis_or"   : 0.,
        "orbital_p"     : ONE_YEAR * 0.6152,
        "orb_speed"     : 35.,
        },
    "EARTH": {
        "name"          : "Earth",
        "rotation_p"    : SIDERAL_DAY,
        "radius"        : 1.,
        "obliquity"     : radians(23.44),
        "inclination"   : 0.,
        "eccentricity"  : 0.017,
        "semi_major"    : 1.,
        "apoapsis_or"   : 0.,
        "orbital_p"     : ONE_YEAR,
        "orb_speed"     : 29.8,
        },
    "MARS": {
        "name"          : "Mars",
        "rotation_p"    : 1.026,
        "radius"        : 0.532,
        "obliquity"     : radians(25.19),
        "inclination"   : radians(1.85),
        "eccentricity"  : 0.093,
        "semi_major"    : 1.523,
        "apoapsis_or"   : 0.,
        "orbital_p"     : ONE_YEAR * 1.8809,
        "orb_speed"     : 24.1,
        },
    "JUPITER": {
        "name"          : "Jupiter",
        "rotation_p"    : 0.410,
        "radius"        : 11.209,
        "obliquity"     : radians(3.13),
        "inclination"   : radians(1.31),
        "eccentricity"  : 0.048,
        "semi_major"    : 5.203,
        "apoapsis_or"   : 0.,
        "orbital_p"     : ONE_YEAR * 11.862,
        "orb_speed"     : 13.1,
        },
    "SATURN": {
        "name"          : "Saturn",
        "rotation_p"    : 0.426,
        "radius"        : 9.449,
        "obliquity"     : radians(26.73),
        "inclination"   : radians(2.49),
        "eccentricity"  : 0.054,
        "semi_major"    : 9.537,
        "apoapsis_or"   : 0.,
        "orbital_p"     : 365.25 * 29.458,
        "orb_speed"     : 9.6,
        },
    "URANUS": {
        "name"          : "Uranus",
        "rotation_p"    : -0.746,
        "radius"        : 4.007,
        "obliquity"     : radians(97.77),
        "inclination"   : radians(0.77),
        "eccentricity"  : 0.047,
        "semi_major"    : 19.229,
        "apoapsis_or"   : 0.,
        "orbital_p"     : ONE_YEAR * 84.01,
        "orb_speed"     : 6.8,
        },
    "NEPTUNE": {
        "name"          : "Neptune",
        "rotation_p"    : 0.718,
        "radius"        : 3.883,
        "obliquity"     : radians(28.32),
        "inclination"   : radians(1.77),
        "eccentricity"  : 0.009,
        "semi_major"    : 30.069,
        "apoapsis_or"   : 0.,
        "orbital_p"     : ONE_YEAR * 164.79,
        "orb_speed"     : 5.4,
        },
    "MOON": {
        "name"          : "Moon",
        "rotation_p"    : 27.31,
        "radius"        : 0.273,
        "obliquity"     : radians(6.687),
        "inclination"   : radians(5.145),
        "eccentricity"  : 0.0549,
        "semi_major"    : 0.002569548,
        "apoapsis_or"   : 0.,
        "orbital_p"     : 27.31,
        "orb_speed"     : 1.,
        },
    "PLUTO": {
        "name"          : "Pluto",
        "rotation_p"    : 1.0,
        "radius"        : 1.,
        "obliquity"     : 0.,
        "inclination"   : radians(57.47),
        "eccentricity"  : 0.,
        "semi_major"    : 39.44,
        "apoapsis_or"   : 0.,
        "orbital_p"     : 365.25,
        "orb_speed"     : 4.74,
        },
}


# ============================================================================================================
# Numpy rotation matrix

def np_rotation(angle, axis='Z'):
    cs = np.cos(angle)
    sn = np.sin(angle)
    _0 = np.zeros_like(angle)
    _1 = np.ones_like(angle)

    # Definition is "inverted" because of final transposition
    # Works both for one angle and an array of angles

    if axis.upper() == 'X':
        return np.array((
            (_1,  _0, _0),
            (_0,  cs, sn),
            (_0, -sn, cs))).transpose()

    elif axis.upper() == 'Y':
        return np.array((
            (cs, _0, -sn),
            (_0, _1,  _0),
            (sn, _0,  cs))).transpose()

    elif axis.upper() == 'Z':
        return np.array((
            ( cs, sn, _0),
            (-sn, cs, _0),
            ( _0, _0, _1))).transpose()

    else:
        raise RuntimeError(f"numpy rotation matrix: axis must be in (X, Y, Z), not '{axis}'")

def np_rotate(m, v):
    return (m @ v.transpose()).transpose()

# ============================================================================================================
# One planet initialized from the db

class Planet():
    def __init__(self, name, object=None):
        
        planet      = PLANETS[name.upper()]
        self.name   = planet["name"]
        self.around = None
        self.object = object
    
        # Dimensions
        
        self.radius_            = planet["radius"]
        self.rotation_period_   = planet["rotation_p"]
        self.obliquity          = planet["obliquity"]
        self.rotation_phase     = 0.
        
        # Revolution
        self.semi_major_        = planet["semi_major"]
        self.eccentricity       = planet["eccentricity"]
        self.revolution_period_ = planet["orbital_p"]
        self.inclination        = planet["inclination"]
        self.apoapsis_angle     = planet["apoapsis_or"]
        self.revolution_phase   = 0.
        self.orbit_speed        = planet["orb_speed"]
        
        # Scale
        self.orbit_unit_        = 30.
        self.radius_unit_       = 1.
        
        # Time
        self.rotation_time_     = 1.
        self.revolution_time_   = 1.
        
        # Compute
        self.motion_compute()
        
    # ------------------------------------------------------------------------------------------
    # Scales
    
    @property
    def orbit_unit(self):
        return self.orbit_unit_
    
    @orbit_unit.setter
    def orbit_unit(self, value):
        self.orbit_unit_ = value
        self.motion_compute()

    @property
    def radius_unit(self):
        return self.radius_unit_
    
    @radius_unit.setter
    def radius_unit(self, value):
        self.radius_unit_ = value
        self.motion_compute()

    @property
    def rotation_time(self):
        return self.rotation_time_
    
    @rotation_time.setter
    def rotation_time(self, value):
        self.rotation_time_ = value
        self.motion_compute()

    @property
    def revolution_time(self):
        return self.revolution_time_
    
    @revolution_time.setter
    def revolution_time(self, value):
        self.revolution_time = value
        self.motion_compute()
        
    # ------------------------------------------------------------------------------------------
    # Access to scales properties
        
    @property
    def radius(self):
        return self.radius_ * self.radius_unit
    
    @property
    def semi_major(self):
        return self.semi_major_ * self.orbit_unit
    
    @property
    def rotation_period(self):
        return self.rotation_period_ * self.rotation_time
    
    @property
    def revolution_period(self):
        return self.revolution_period_ * self.revolution_time
    
    # ------------------------------------------------------------------------------------------
    # Compute the vars
    
    def motion_compute(self):
        
        # Angular velocities
        
        self.rot_omega = 0. if self.rotation_period_   == PERIOD_ZERO else 2*pi/self.rotation_period
        self.rev_omega = 0. if self.revolution_period_ == PERIOD_ZERO else 2*pi/self.revolution_period
        
        # Ellipsis
        
        self.a  = self.semi_major
        self.e  = self.eccentricity
        self.e2 = self.e * self.e
        self.c  = self.a * self.e
        self.b  = self.a * sqrt(1 - self.e2)
        self.p  = self.a * (1 - self.e2)
        
        # K value : sqrt(GM/p**3)
        # omega = self.rev_omga + self.omega_emp*cos(self.rev_omga*t)
        
        self.K = self.rev_omega/(1+self.e2)
        self.omega_amp = 2*self.e*self.K
        self.theta_var = 2*self.e/(1 + self.e2)   
        
        # Inclination matrix (around 'X' axis)
        self.incl_mat = np_rotation(self.inclination, 'X')

        # Obliquity matrix (around 'X' axis)
        self.obl_mat = np_rotation(self.obliquity, 'X')

    # ------------------------------------------------------------------------------------------
    # Polar coordinates
    
    def polar(self, t):
        w0    = self.rev_omega
        theta = self.revolution_phase + w0*t + self.theta_var*np.sin(w0*t)
        r     = self.p/(1 + self.e*np.cos(theta))
        return r, theta
    
    # ------------------------------------------------------------------------------------------
    # Center
    
    def rev_center(self, t):
        return np.array((0., 0., 0.)) if self.around is None else self.around.location(t)
    
    # ------------------------------------------------------------------------------------------
    # Location
    
    def location(self, t):
        # Location on the ellipsis
        r, theta = self.polar(t)
        loc = r*np.array((np.cos(theta), np.sin(theta), np.zeros_like(theta)))
        
        # Return the result
        return self.rev_center(t) + (self.incl_mat @ loc).transpose()

    # ------------------------------------------------------------------------------------------
    # z rotation of the planet
    
    def z_rotation(self, t):
        return np.mod(self.rotation_phase + self.rot_omega*t, pi2)

    # ------------------------------------------------------------------------------------------
    # Rotation of the planet - Euler
    
    def rotation_euler(self, t):
        return Euler((self.obliquity, 0, self.z_rotation(t)), 'ZYX')

    # ------------------------------------------------------------------------------------------
    # Rotation of the planet - Matrix

    def rotation_matrix(self, t):
        return self.obl_mat @ np_rotation(self.z_rotation(t), 'Z')

    # ------------------------------------------------------------------------------------------
    # Inverted rotation of the planet - Matrix

    def inverted_rotation_matrix(self, t):
        return np_rotation(-self.z_rotation(t), 'Z') @ (-self.obl_mat)

    # ------------------------------------------------------------------------------------------
    # View point
    
    def local_view(self, t, other):
        loc  = other.location(t) - self.location(t)
        mrot = self.inverted_rotation_matrix(t)
        return np.einsum('kij,kj->ki', mrot, loc)

    # ------------------------------------------------------------------------------------------
    # Trajectory
    
    def trajectory(self, t0, t1, count=100, view_point=None):
        
        t = np.linspace(t0, t1, count)
        
        if view_point is None:
            return self.location(t)
        else:
            return view_point.local_view(t, self)


# ============================================================================================================
# The planets
        
class Planets():
    
    def __init__(self):
        self.planets = {}
        for name in PLANETS:
            planet = Planet(name)
            self.planets[planet.name.upper()] = planet
            
        self["Moon"].around = self["Earth"]
        
    # ------------------------------------------------------------------------------------------
    # Item access
    
    def __len__(self):
        return len(self.planets)
        
    def __getitem__(self, index):
        if type(index) is str:
            return self.planets[index.upper()]
        else:
            return list(self.planets.items())[index][1]
    
    # ------------------------------------------------------------------------------------------
    # Link a planet to an object
    
    def link_object(self, name, object):
        self[name].object = object
        self[name].object.rotation_mode = 'ZYX'
            
    # ------------------------------------------------------------------------------------------
    # Set the scales

    def orbit_unit(self, value, planets = []):
        if len(planets) == 0:
            planets = self.planets.keys()
        for name in planets:
            self.planets[p.upper()].orbit_unit = value
    
    def radius_unit(self, value, planets = []):
        if len(planets) == 0:
            planets = self.planets.keys()
        for name in planets:
            self.planets[p.upper()].radius_unit = value
    
    def rotation_time(self, value, planets = []):
        if len(planets) == 0:
            planets = self.planets.keys()
        for name in planets:
            self.planets[p.upper()].rotation_time = value
    
    def revolution_time(self, value, planets = []):
        if len(planets) == 0:
            planets = self.planets.keys()
        for name in planets:
            self.planets[p.upper()].revolution_time = value
            
    # ------------------------------------------------------------------------------------------
    # Update locations
    
    def update(self, t):
        for planet in self:
            if planet.object is not None:
                o = planet.object 
                o.location = planet.location(t)
                o.rotation_euler = planet.rotation_euler(t)
                

