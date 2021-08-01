#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 07:52:00 2021

@author: alain
"""

# =============================================================================================================================
# Module exception

class WError(Exception):
    def __init__(self, message, **kwargs):
        self.message = message
        self.vals = kwargs
        
    def __str__(self):
        s = "\n" + "-"*100 + "\n"
        s += "ERROR in Blender Wrap module\n"
        s += self.message + "\n"
        for k, v in self.vals.items():
            s += f"- {k.replace('_', ' '):15s}: {v}\n"
        return s

# =============================================================================================================================
# Access to chained attrs

# ---------------------------------------------------------------------------
# Get access to a prop with chained syntax: attr1.attr2[index].attr3
# Return the before last attribute, the name of the last attribute and the index
# object, "location.x' --> object.location, "x", None
# object, "data.vertices[0].co" --> object.data.vertices[0], "co", None
# object, "data.vertices[0]" --> object.data, "vertices", 0

def chained_attr(obj, prop):
    # ---------------------------------------------------------------------------
    # An attribute can be an array item : array[index]
    # The index can be a string or an integer

    def array_index(s):

        # Nothing to do
        if s[-1] != ']':
            return s, None

        # Find the opening bracket
        left = s.find('[')
        if left < 0:
            raise WError(
                f"Attribute error in '{prop}': '{s}' is an incorrect array access ('[' is missing)",
                Function = "chained_attr",
                obj = obj,
                prop = prop)

        # Read the index
        index = s[left + 1:-1]
        s = s[:left]

        # Convert in a int if not an str
        if not index[0] in ["'", '"']:
            try:
                index = int(index)
            except:
                pass

        # Return the result
        return s, index

    # ---------------------------------------------------------------------------
    # If the last attribute is x, y, z or w, replace by index version
    # This transform location.x by location[0] which is compatible with np version

    if (len(prop) > 2) and (prop[-2:] in ['.x', '.y', '.z', '.w']):
        if prop[-1] == 'W':
            index = 3
        else:
            index = ord(prop[-1]) - ord('x')
        prop = prop[:-2] + f"[{index}]"

    # ---------------------------------------------------------------------------
    # Split the chained attrs and initialize the loop

    attrs = prop.split('.')
    o = obj
    debug = "object"

    # ---------------------------------------------------------------------------
    # Loop on the attrs but the last one

    for i in range(len(attrs) - 1):

        # ----- Current attr to manage
        # Since it can be an array acces, split it in array name, index name

        s, index = array_index(attrs[i])

        # ----- Set the current object to the attr if exists

        if hasattr(o, s):
            o = getattr(o, s)
            debug += '.' + s
        else:
            raise WError(f"Attribute error in '{prop}': '{debug}' has not attribute named '{s}'",
                Function = "chained_attr",
                obj = obj,
                prop = prop)

        # ----- Time to go to the item if the array is an index
        if index is not None:
            try:
                o = o[index]
            except:
                raise WError(f"Attribute error in '{prop}': improper index '{index}' for '{debug}'",
                    Function = "chained_attr",
                    obj = obj,
                    prop = prop)
                                   

    # ---------------------------------------------------------------------------
    # Let's return the result

    attr, index = array_index(attrs[-1])
    # print(f"{prop:25}: {o}, {attr}, {index}")
    return o, attr, index

# -----------------------------------------------------------------------------------------------------------------------------
# Extend get_attr with chained syntex

def get_chained_attr(object, props):
    obj, attr, index = chained_attr(object, props)
    if index is None:
        return getattr(obj, attr)
    else:
        return getattr(obj, attr)[index]

# -----------------------------------------------------------------------------------------------------------------------------
# Extend set_attr with chained syntax

def set_chained_attr(object, props, value):
    obj, attr, index = chained_attr(object, props)
    if index is None:
        setattr(obj, attr, value)
    else:
        getattr(obj, attr)[index] = value

