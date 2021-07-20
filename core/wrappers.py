#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 21:41:34 2021

@author: alain
"""

from math import degrees, radians
import numpy as np

import bpy
from mathutils import Quaternion

from .frames import get_frame
from .plural import to_shape, setattrs, getattrs
from .bezier import  control_points, PointsInterpolation
from .interpolation import BCurve

from .geometry import q_tracker
from .bmatrix import *

from .commons import base_error_title, get_chained_attr, set_chained_attr
error_title = base_error_title % "wrappers.%s"

# =============================================================================================================================
# bpy_struct wrapper
# wrapped : bpy_struct

class WStruct():
    """Blender Struct Wrapper
    
    Root class of all the wrappers. Wraps the Struct class which
    is the root class of almost veerything in Blender.
    
    As with Blender, this class is not supposed to be instanciated directly.
    
    Wraps the management of the fcurves and key frames.
    """

    def __init__(self, wrapped=None, name=None, coll=None):
        """Blender Struct Wrapper
        
        Can be initialized either by a Struct or by a name in a Blender Collection.
        
        Parameters
        ----------
        wrapped : Blender Struct, optional
            The Struct to wrap. The default is None.
        name : Str, optional
            The name of the Struct in the collection. The default is None.
        coll : Blender Collection, optional
            The collection (dictionnary) into which looking for the name. The default is None.

        Returns
        -------
        None.

        """
        super().__setattr__("wrapped_", wrapped)
        super().__setattr__("name_",    name)
        super().__setattr__("coll_",    coll)

    @property
    def wrapped(self):
        """The wrapped Blender instance.

        Returns
        -------
        Struct
            The wrapped object.
        """
        
        if self.wrapped_ is None:
            return self.coll_[self.name_]
        else:
            return self.wrapped_

    def __repr__(self):
        return f"[Wrapper {self.__class__.__name__} of {self.class_name} '{self.wrapped}']"

    def __getattr__(self, name):
        """Capture the attributes of the wrapped object.

        Parameters
        ----------
        name : str
            Attribute name.

        Raises
        ------
        RuntimeError
            The attributes does'nt exist for the wrapped object.

        Returns
        -------
        Anything
            The attribute value.
        """
        
        try:
            return getattr(self.wrapped, name)
        except:
            raise RuntimeError(f"Attribute '{name}' doesn't exist for class '{self.__class__.__name__}'")

    def __setattr__(self, name, value):
        """Capture the attributes of the wrapped object.
        
        Setter version (see __getattr__)
        """
        
        try:
            setattr(self.wrapped, name, value)
        except:
            super().__setattr__(name, value)

    @property
    def class_name(self):
        """The class name of the wrapped object.        

        Returns
        -------
        str
            Blender class name.
        """
        
        return self.wrapped.__class__.__name__

    # ----------------------------------------------------------------------------------------------------
    # Ensure update

    def mark_update(self):
        """Mark the object to be refreshed by Blender engine.
        
        Must be called when attributes changes is not reflected in the viewport.

        Returns
        -------
        None.
        """
        
        #self.wrapped.id_data.update_tag(refresh={'OBJECT', 'DATA', 'TIME'})
        #self.wrapped.id_data.update_tag(refresh={'OBJECT', 'DATA', 'TIME'})
        #self.wrapped.id_data.update_tag(refresh={'TIME'})
        self.wrapped.id_data.update_tag()

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Chained attrs

    def get_attr(self, attribute):
        """Get access to a prop with chained syntax: attr1.attr2[index].attr3
        
        Possible syntaxes:
        - object, "location.x'
        - object, "data.vertices[0].co"
        - object, "data.vertices[0]"

        Parameters
        ----------
        attribute : TYPE
            DESCRIPTION.

        Returns
        -------
        Anything
            Attribute value
        """
        
        return get_chained_attr(self, attribute)

    def set_attr(self, attribute, value):
        """Access to a prop with chained syntax: attr1.attr2[index].attr3
        
        Setter version, see get_attr
        """
        
        set_chained_attr(self, attribute, value)

    # ----------------------------------------------------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------
    # Keyframes

    # ----------------------------------------------------------------------------------------------------
    # data_path, index
    # Syntax name.x overrides index value if -1

    @staticmethod
    def data_path_index(name, index=-1):
        """Key frame utility: normalize the syntax name, index of a key frame.
        
        Used for instance for location. The syntax "location.x" can be used rather
        than the Blender syntax: "location", index=0.
        
        Transforms: location.y, -1 --> location, 1
        
        If index is not -1 and the dotted syntaxe is used, an error is raised.

        Parameters
        ----------
        name : str
            The name of the attribute with possible the syntax attr.x.
        index : int, optional
            For array attribute, the index of the entry. The default is -1.

        Raises
        ------
        RuntimeError
            1) If dotted name is not in name.x, name.y, name.z or name.w.
            2) If dotted named is passed with an index different from 1

        Returns
        -------
        str
            A not dotted attribute name.
        int
            The index in the array.
        """

        if len(name) < 3:
            return name, index

        if name[-2] == ".":
            try:
                idx = ["x", "y", "z", "w"].index(name[-1])
            except:
                raise RuntimeError(
                    error_title % "WStruct.data_path_index" +
                    f"{name}: suffix for index must be in (x, y, z, w), not '{name[-1]}'."
                    )
            if index >= 0 and idx != index:
                raise RuntimeError(
                    error_title % "data_path_index" +
                    f"Suffix of '{name}' gives index {idx} which is different from passed index {index}."
                    )

            return name[:-2], idx

        return name, index

    # ----------------------------------------------------------------------------------------------------
    # Size of an attribute (gives possible values for index)

    def attribute_size(self, attr):
        """Returns the size of an attribute.

        Gives possibles values for the index.        

        Parameters
        ----------
        attr : anything
            A python object.

        Returns
        -------
        int
            Array size if array, otherwise 1.
        """
        
        return np.size(getattr(self.wrapped, attr))

    # ----------------------------------------------------------------------------------------------------
    # Is the object animated

    @property
    def is_animated(self):
        """Is the object animated        

        Returns
        -------
        bool
            True if animated, False otherwise.
        """
        
        return self.wrapped.animation_data is not None

    # ----------------------------------------------------------------------------------------------------
    # Get animation_data. Create it if it doesn't exist

    def animation_data(self, create=True):
        """Returns the Blender animation_data of the instance.

        Creates the animation date if create argument is True.        

        Parameters
        ----------
        create : bool, optional
            Creates the animation_data if it doesn't exist. The default is True.

        Returns
        -------
        None or animation_data
            The animation_data of the instance.
        """
        
        animation = self.wrapped.animation_data
        if create and (animation is None):
            return self.wrapped.animation_data_create()
        else:
            return animation

    # ----------------------------------------------------------------------------------------------------
    # Get animation action. Create it if it doesn't exist

    def animation_action(self, create=True):
        """Get the animation action of the instance.

        Create the animation_action, and possibly animation_data, if create is True.        

        Parameters
        ----------
        create : bool, optional
            Creates the animation_action for the wrapped object if True. The default is True.

        Returns
        -------
        animation_action
            The animation action of the Blender instance.
        """
        
        animation = self.animation_data(create)
        if animation is None:
            return None

        action = animation.action
        if create and (action is None):
            animation.action = bpy.data.actions.new(name="WA action")

        return animation.action

    # ----------------------------------------------------------------------------------------------------
    # Get fcurves. Create it if it doesn't exist

    def get_fcurves(self, create=True):
        """Get the animation fcurves of the Blender object.
        
        Creates the fcurves collection if create is True.

        Parameters
        ----------
        create : bool, optional
            Create the fcurves collection if true. The default is True.

        Returns
        -------
        fcurves
            The fcurves collection of the object.
        """

        aa = self.animation_action(create)

        if aa is None:
            return None
        else:
            return aa.fcurves

    # ----------------------------------------------------------------------------------------------------
    # Check if a fcurve is an animation of a property

    @staticmethod
    def is_fcurve_of(fcurve, name, index=-1):
        """Check if a fcurve is the animation of the given property
        
        For arrays, if index == -1, returns True whatever the value of array_index.
        If index != -1, check that the index value of the fcurves matches the given index.
        
        Parameters
        ----------
        fcurve : Blender fcurve
            The fcurve to check.
        name : string
            The name of the property the check.
        index : int, optional
            The index in the array when the property is an array. The default is -1.

        Returns
        -------
        bool
            True if fcurve controls the (name, attribute) property.
        """

        if fcurve.data_path == name:
            if (index == -1) or (fcurve.array_index < 0):
                return True

            return fcurve.array_index == index

        return False

    # ----------------------------------------------------------------------------------------------------
    # Return the animation curves of a property
    # Since there could be more than one curve, an array, possibly empty, is returned

    def get_acurves(self, name, index=-1):
        """All the animations curves of a property as a python array.

        Not that the method returns an array of one item even if only one fcurve
        exists.
        
        (name, index): uses dotted syntax (see data_path_index method)

        Parameters
        ----------
        name : str
            property name.
        index : int, optional
            Array index or -1. The default is -1.

        Returns
        -------
        Array of fcurves
            The animation curves of the property or an empty array.
        """

        name, index = self.data_path_index(name, index)

        acs = []
        fcurves = self.get_fcurves(create=False)
        if fcurves is not None:
            for fcurve in fcurves:
                if self.is_fcurve_of(fcurve, name, index):
                    acs.append(fcurve)

        return acs
    
    # ----------------------------------------------------------------------------------------------------
    # Return the BCurve version of an animation curve
    # Use the dotted version only for UI simplicty
    
    def bcurve(self, name):
        """Get the BCurve version of a fcurve.
        
        The BCurve is more efficient when called with an array of value:
            - Y = bcurve(X)
            - Y = [fcurve.evaluate(x) for x in X]
            
        Returns only one function. Uses the dottes syntax only.
        
        Raises an error if more that one fcurve exist for the given attribute: this occurs
        when several entries in an array are animated. For instance, if all entries of
        'location' are animated:
            - name="location"   --> error
            - name="location.x" --> ok

        Parameters
        ----------
        name : str
            Animated attribute name.

        Returns
        -------
        BCurve.
        """
        
        acs = self.get_acurves(name)
        if len(acs) > 1:
            raise RuntimeError(error_title % "bcurve" +
                    f"The attribute '{name}' has several animation curves. Can return only one.\n"
                    "Use dotted syntax such as 'location.x' to specify which curve to get."
                    )
            
        if len(acs) == 0:
            raise RuntimeError(error_title % "bcurve" +
                    f"The attribute '{name}' is not animated."
                    )
            
        return BCurve.FromFCurve(acs[0])
    
    def set_bcurve(self, name, bc):
        
        b_kfs = bc.keyframe_points
        
        acs = self.new_acurves(name, reset=True)
        fc = acs[0]
        fc.keyframe_points.add(len(b_kfs))
        
        for kfs, kft in zip(b_kfs, fc.keyframe_points):
            kft.co            = kfs.co.copy()
            kft.interpolation = kfs.interpolation
            kft.amplitude     = kfs.amplitude
            kft.back          = kfs.back
            kft.easing        = kfs.easing
            kft.handle_left   = kfs.handle_left
            kft.handle_right  = kfs.handle_right
            kft.period        = kfs.period

    # ----------------------------------------------------------------------------------------------------
    # Delete a fcurve

    def delete_acurves(self, acurves):
        """Delete the fcurves given in the argument.

        Parameters
        ----------
        acurves : array of fcurves
            The list of fcurves to delete.

        Returns
        -------
        None.
        """
        
        fcurves = self.get_fcurves()
        try:
            for fcurve in acurves:
                fcurves.remove(fcurve)
        except:
            pass

    # ----------------------------------------------------------------------------------------------------
    # fcurve integral

    @staticmethod
    def fcurve_integral(fcurve, frame_start=None, frame_end=None):
        """Computes the integral of a fcurve between two frames.        

        Parameters
        ----------
        fcurve : fcurve
            The fcurve function to compute.
        frame_start : str or int, optional
            The initial frame. The default is None.
        frame_end : str or int, optional
            The final frame. The default is None.

        Returns
        -------
        float
            The integral.
        """

        if frame_start is None:
            frame_start= bpy.context.scene.frame_start

        if frame_end is None:
            frame_end= bpy.context.scene.frame_end

        # Raw algorithm : return all the values per frame
        
        fs = get_frame(frame_start)
        fe = get_frame(frame_end)

        vals = np.array([fcurve.evaluate(i) for i in range(fs, fe+1)])
        vals -= fcurve.evaluate(fs)
        return np.cumsum(vals)

    # ----------------------------------------------------------------------------------------------------
    # Access to an animation curve

    def get_animation_function(self, name, index=-1):
        """Return the value of a property at a given frame.
        
        Used to get a function returning the value of the property at a given frame,
        whatever the property is animated or not
        
        
        examples:
            Cube location.x is animated
            
            wcube.get_animated_value('location', 125)
            
            returns the (x, y, z) location of the object at frame 125 with location.y and location.z

        (name, index): uses dotted syntax (see data_path_index method)

        Parameters
        ----------
        name : str
            property name.
        index : int, optional
            Array index or -1. The default is -1.

        Returns
        -------
        function of template: f(frame) -> property value at the given frame
            The function computing the property value at the required frame
        """

        # Dotted syntax
        name, index = self.data_path_index(name, index)
        
        # The non animated value
        val = getattr(self.wrapped, name)
        
        # Array of animation curves
        acurves = self.get_acurves(name, index)
        
        # No animation: value is constant
        if len(acurves) == 0:
            if index < 0:
                return lambda frame: val
            else:
                return lambda frame: val[index]
            
        
        # Array animation function
        def array_animation(frame, acurves, val):
            res = np.array(val)
            for i, fcurve in enumerate(acurves):
                res[fcurve.array_index] = fcurve.evaluate(get_frame(frame))
            return res
            
        
        # Attribute is an array
        if hasattr(val, '__len__'):
            
            # A specific entry in the array
            if index >= 0:
                for fcurve in acurves:
                    if fcurve.array_index == index:
                        return lambda frame: fcurve.evaluate(get_frame(frame))
                    
                return lambda frame: val[index]
            
            # All the entries
            return lambda frame: array_animation(frame, acurves, val)
        
        # Attribute is not an array
        return lambda frame: acurves[0].evaluate(get_frame(frame))

    # ----------------------------------------------------------------------------------------------------
    # Get a keyframe at a given frame

    def get_kfs(self, name, frame, index=-1):
        """Get the key frames of a property at a given frame.
        
        (name, index): uses dotted syntax (see data_path_index method)

        Parameters
        ----------
        name : str
            property name.
        frame : float or str
            The frame where to compute the property.
        index : int, optional
            Array index or -1. The default is -1.

        Returns
        -------
        array of keyframes
            The key frames set at this frame.
        """

        acurves = self.get_acurves(name, index)
        frame = get_frame(frame)

        kfs = []
        for fcurve in acurves:
            for kf in fcurve.keyframe_points:
                if kf.co[0] == frame:
                    kfs.append(kfs)
                    break

        return kfs

    # ----------------------------------------------------------------------------------------------------
    # Create an animation curve

    def new_acurves(self, name, index=-1, reset=False):
        """Create anmation curves for a property.
        
        (name, index): uses dotted syntax (see data_path_index method)

        Parameters
        ----------
        name : str
            property name.
        index : int, optional
            Array index or -1. The default is -1.
        reset : bool, optional
            Delete the existing fcurves if True. The default is False.
        
        Returns
        -------
        array of fcurves
            The created animation curves.

        """

        name, index = self.data_path_index(name, index)
        size = self.attribute_size(name)

        acurves = self.get_acurves(name, index)

        # Not an array, or a particular index in an array
        if (size == 1) or (index >= 0):
            if len(acurves) == 0:
                fcurves = self.get_fcurves()
                fcurve  = fcurves.new(data_path=name, index=index)
                acurves.append(fcurve)

        # All entries of an array
        else:
            if len(acurves) != size:
                fcurves = self.get_fcurves(create=True)
                for i in range(size):
                    if len(self.get_acurves(name, index=i)) == 0:
                        acurves.append(fcurves.new(data_path=name, index=i))

        # Reset
        if reset:
            for fcurve in acurves:
                count = len(fcurve.keyframe_points)
                for i in range(count):
                    fcurve.keyframe_points.remove(fcurve.keyframe_points[0], fast=True)

        # Result
        return acurves

    # ----------------------------------------------------------------------------------------------------
    # Set an animation fcurve to a property

    def set_acurves(self, name, acurves, index=-1):
        """Set an animation fcurve to a property
        
        (name, index): uses dotted syntax (see data_path_index method)

        Parameters
        ----------
        name : str
            property name.
        acurves : array of fcurves
            The animation curves to set on the property.
        index : int, optional
            Array index or -1. The default is -1.

        Raises
        ------
        RuntimeError
            If the number of animation curves doesn't match the length of the property.

        Returns
        -------
        None.
        """

        # Get / create the fcurves
        acs = self.new_acurves(name, index, reset=True)

        # Check the size
        if len(acs) != len(acurves):
            raise RuntimeError(
                error_title % "set_acurves" +
                f"The number of fcurves to set ({len(acs)}) doesn't match the number of passed fcurves ({len(acurves)}).\n" +
                f"name: {name}, index: {index}"
                )
            
        # Copy the keyframes
        for f_source, f_target in zip(acurves, acs):

            kfp = f_source.keyframe_points
            if len(kfp) > 0:

                f_target.extrapolation = f_source.extrapolation
                f_target.keyframe_points.add(len(kfp))

                for kfs, kft in zip(kfp, f_target.keyframe_points):
                    kft.co            = kfs.co.copy()
                    kft.interpolation = kfs.interpolation
                    kft.amplitude     = kfs.amplitude
                    kft.back          = kfs.back
                    kft.easing        = kfs.easing
                    kft.handle_left   = kfs.handle_left
                    kft.handle_right  = kfs.handle_right
                    kft.period        = kfs.period
                    

    # ----------------------------------------------------------------------------------------------------
    # Delete keyframes

    def del_kfs(self, name, frame0=None, frame1=None, index=-1):
        """Delete all the key frames between two frames
        
        (name, index): uses dotted syntax (see data_path_index method)

        Parameters
        ----------
        name : str
            property name.
        frame0 : float or str, optional
            Frame to start with. The default is None.
        frame1 : float or str, optional
            Frame to end with. The default is None.
        index : int, optional
            Array index or -1. The default is -1.

        Returns
        -------
        None.

        """

        # Starting and ending frames
        okframe0 = frame0 is not None
        okframe1 = frame1 is not None

        if okframe0:
            frame0 = get_frame(frame0)
        if okframe1:
            frame1 = get_frame(frame1)
            
        # Loop on the animation curves
        acurves = self.get_acurves(name, index)
        for fcurve in acurves:
            kfs = []
            for kf in fcurve.keyframe_points:
                ok = True
                if okframe0:
                    ok = kf.co[0] >= frame0
                if okframe1:
                    if kf.co[0] > frame1:
                        ok = False
                if ok:
                    kfs.append(kf)

            for kf in kfs:
                try:
                    fcurve.keyframe_points.remove(kf)
                except:
                    pass

    # ----------------------------------------------------------------------------------------------------
    # Insert key frames

    def set_kfs(self, name, frame, value=None, interpolation=None, index=-1):
        """Set key frames.
        
        (name, index): uses dotted syntax (see data_path_index method)

        Parameters
        ----------
        name : str
            property name.
        frame : float or str
            The frame to key.
        value : Anything, optional
            The value to set to the property. If None, keep the current value. The default is None.
        interpolation : str, optional
            A valid interpolation code. The default is None.
        index : int, optional
            Array index or -1. The default is -1.

        Returns
        -------
        None.
        """
        
        # The frame as a float
        frame = get_frame(frame)

        # Dotted syntax
        name, index = self.data_path_index(name, index)
        
        # If value is not None, set the property at the target value
        if value is not None:
            curr = getattr(self.wrapped, name)
            if index == -1:
                new_val = value
            else:
                new_val = curr
                new_val[index] = value
            setattr(self.wrapped, name, new_val)
            
        # Insert the keyframe
        self.wrapped.keyframe_insert(name, index=index, frame=frame)
        
        # Interpolation if specified
        if interpolation is not None:
            kfs = self.get_kfs(name, frame, index)
            for kf in kfs:
                kf.interpolation = interpolation
                
        # Restore the value if changed
        if value is not None:
            setattr(self.wrapped, name, curr)
            

    def hide(self, frame, show_before=False, viewport=True):
        """Add a key frame to hide the object.
        
        Hide in render mode. To hide also in the viewport, use viewport argument.

        Parameters
        ----------
        frame : float or int
            The frame where to hide the object.
        show_before : bool, optional
            Set a keyframe just before to show the object. The default is False.
        viewport : bool, optional
            Also hide in the viewport. The default is True.

        Returns
        -------
        None.
        """
        
        frame = get_frame(frame)
        
        self.set_kfs("hide_render", frame, True)
        if show_before:
            self.set_kfs("hide_render", frame-1, False)

        if viewport:
            self.set_kfs("hide_viewport", frame, True)
            if show_before:
                self.set_kfs("hide_viewport", frame-1, False)
                

    def show(self, frame, hide_before=False, viewport=True):
        """Add a key frame to show the object.
        
        Show in render mode. To show also in the viewport, use viewport argument.

        Parameters
        ----------
        frame : float or int
            The frame where to show the object.
        hide_before : bool, optional
            Set a keyframe just before to hide the object. The default is False.
        viewport : bool, optional
            Also show in the viewport. The default is True.

        Returns
        -------
        None.
        """
        
        frame = get_frame(frame)
        
        self.set_kfs("hide_render", frame, False)
        if hide_before:
            self.set_kfs("hide_render", frame-1, True)

        if viewport:
            self.set_kfs("hide_viewport", frame, False)
            if hide_before:
                self.set_kfs("hide_viewport", frame-1, True)



# ---------------------------------------------------------------------------
# Root wrapper
# wrapped = ID

class WID(WStruct):
    """Wrapper for the Blender ID Struct.
    
    Implements the evaluated property to give access to the evaluated object.
    """

    # ---------------------------------------------------------------------------
    # Evaluated ID

    @property
    def evaluated(self):
        """Wraps the evaluated object.

        Returns
        -------
        WID
            Wrapped of the evaluated object.
        """
        
        if self.wrapped.is_evaluated:
            return self

        else:
            depsgraph = bpy.context.evaluated_depsgraph_get()
            return self.__class__(self.wrapped.evaluated_get(depsgraph))


# ---------------------------------------------------------------------------
# Shape keys data blocks wrappers
# wrapped = Shapekey (key_blocks item)

class WShapekey(WStruct):
    """Wraps the key_blocks collection of a shapekey class
    """

    @staticmethod
    def sk_name(name, step=None):
        """Returns then name of a shape key within a series        

        Parameters
        ----------
        name : str
            Base name of tyhe shape key.
        step : int, optional
            The step number. The default is None.

        Returns
        -------
        str
            Full shape key name: "shapekey 999".
        """
        
        return name if step is None else f"{name} {step:3d}"

    def __len__(self):
        """The wrapper behaves as an array.        

        Returns
        -------
        int
            Le length of the collection.
        """
        
        return len(self.wrapped.data)

    def __getitem__(self, index):
        """The wrapper behaves as an array.

        Parameters
        ----------
        index : int
            Item index.

        Returns
        -------
        ShapeKey
            The indexed shape key.
        """
        
        return self.wrapped.data[index]

    def check_attr(self, name):
        """Check if an attribute exists.
        
        Return only if the attr exist, raise an error otherwise.
        Shape key is used for meshes and splines. This utility raises an error if the user
        makes a mistake with the type of keyed shape.

        Parameters
        ----------
        name : str
            Attribute name.

        Raises
        ------
        RuntimeError
            If the attr doesn't exist.

        Returns
        -------
        None.
        """
        
        if name in dir(self.wrapped.data[0]):
            return
        
        raise RuntimeError(
            error_title % "WShapekey" +
            f"The attribut '{name}' doesn't exist for this shape key '{self.name}'."
            )

    @property
    def verts(self):
        """The vertices of the shape key.

        Returns
        -------
        numpy array of shape (len, 3)
            The vertices.
        """
        
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count*3, np.float)
        data.foreach_get("co", a)
        return a.reshape((count, 3))

    @verts.setter
    def verts(self, value):
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count*3)
        data.foreach_set("co", a)

    @property
    def lefts(self):
        """Left handles of spline.   

        Returns
        -------
        numpy array of shape (len, 3)
            The left handles.
        """
        
        self.check_attr("handle_left")
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count*3, np.float)
        data.foreach_get("handle_left", a)
        return a.reshape((count, 3))

    @lefts.setter
    def lefts(self, value):
        self.check_attr("handle_left")
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count*3)
        data.foreach_set("handle_left", a)

    @property
    def rights(self):
        """Right handles of spline.   

        Returns
        -------
        numpy array of shape (len, 3)
            The right handles.
        """
        
        self.check_attr("handle_right")
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count*3, np.float)
        data.foreach_get("handle_right", a)
        return a.reshape((count, 3))

    @rights.setter
    def rights(self, value):
        self.check_attr("handle_right")
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count*3)
        data.foreach_set("handle_right", a)

    @property
    def radius(self):
        """Radius of spline.   

        Returns
        -------
        numpy array of shape (len)
            The radius.
        """
        
        self.check_attr("radius")
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count, np.float)
        data.foreach_get("radius", a)
        return a

    @radius.setter
    def radius(self, value):
        self.check_attr("radius")
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count)
        data.foreach_set("radius", a)

    @property
    def tilts(self):
        """Tilts of spline.   

        Returns
        -------
        numpy array of shape (len)
            The tilts.
        """
        
        self.check_attr("tilt")
        data = self.wrapped.data
        count = len(self.data)
        a = np.empty(count, np.float)
        data.foreach_get("tilt", a)
        return a

    @tilts.setter
    def tilts(self, value):
        self.check_attr("tilt")
        data = self.wrapped.data
        count = len(self.data)
        a = to_shape(value, count)
        data.foreach_set("tilt", a)

# ---------------------------------------------------------------------------
# Mesh mesh wrapper
# wrapped : data block of mesh object

class WMesh(WID):
    """Wrapper of a Mesh structure.
    """

    def __init__(self, wrapped, evaluated=False):
        if evaluated:
            super().__init__(wrapped, name=wrapped.name)
        else:
            super().__init__(name=wrapped.name, coll=bpy.data.meshes)

    @property
    def owner(self):
        """The object owning this mesh
        """
        
        for obj in bpy.data.objects:
            if obj.data is not None:
                if obj.data.name == self.name:
                    return obj
        return None

    # Mesh vertices update

    def mark_update(self):
        super().mark_update()
        self.wrapped.update()

    # Vertices count

    @property
    def verts_count(self):
        """The number of vertices in the mesh.
        """
        
        return len(self.wrapped.vertices)

    # Vertices (uses verts not to override vertices attributes)

    @property
    def verts(self):
        """The vertices of the mesh

        Returns
        -------
        array(len, 3) of floats
            numpy array of the vertices.
        """
        
        verts = self.wrapped.vertices
        a    = np.empty(len(verts)*3, np.float)
        verts.foreach_get("co", a)
        return np.reshape(a, (len(verts), 3))

    @verts.setter
    def verts(self, vectors):
        verts = self.wrapped.vertices
        a     = to_shape(vectors, (len(verts)*3))
        verts.foreach_set("co", a)
        self.mark_update()
        
    # x, y, z vertices access

    @property
    def xs(self):
        """x locations of the vertices
        """
        
        return self.verts[:, 0]

    @xs.setter
    def xs(self, values):
        locs = self.verts
        locs[:, 0] = to_shape(values, self.vcount)
        self.verts = locs

    @property
    def ys(self):
        """y locations of the vertices
        """
        
        return self.verts[:, 1]

    @ys.setter
    def ys(self, values):
        locs = self.verts
        locs[:, 1] = to_shape(values, self.vcount)
        self.verts = locs

    @property
    def zs(self):
        """z locations of the vertices
        """
        
        return self.verts[:, 2]

    @zs.setter
    def zs(self, values):
        locs = self.verts
        locs[:, 2] = to_shape(values, self.vcount)
        self.verts = locs

    # vertices attributes

    @property
    def bevel_weights(self):
        """bevel weights of the vertices
        """
        
        return getattrs(self.wrapped.vertices, "bevel_weight", 1, np.float)

    @bevel_weights.setter
    def bevel_weights(self, values):
        setattrs(self.wrapped.vertices, "bevel_weight", values, 1)

    # edges as indices

    @property
    def edge_indices(self):
        """A python array with the edges indices.
        
        Indices can be used in th 
        
        Returns
        -------
        array of couples of ints 
            A couple of ints per edge.
        """
        
        edges = self.wrapped.edges
        return [e.key for e in edges]

    # edges as vectors

    @property
    def edge_vertices(self):
        """The couple of vertices of the edges.

        Returns
        -------
        numpy array of couple of vertices (n, 2, 3)
            The deges vertices.

        """
        
        return self.verts[np.array(self.edge_indices)]

    # polygons as indices
    
    @property
    def poly_count(self):
        return len(self.wrapped.polygons)

    @property
    def poly_indices(self):
        """The indices of the polygons

        Returns
        -------
        python array of array of ints
            Shape (d1, ?) where d1 is the number of polygons and ? is the number of vertices
            of the polygon.
        """

        polygons = self.wrapped.polygons
        return [tuple(p.vertices) for p in polygons]

    # polygons as vectors

    @property
    def poly_vertices(self):
        """The vertices of the polygons.

        Returns
        -------
        python array of array of triplets.
            Shape is (d1, ?, 3) where d1 is the number of polygons and ? is the number of vertices
            of the polygon.
        """
        
        polys = self.poly_indices
        verts = self.verts
        return [ [list(verts[i]) for i in poly] for poly in polys]

    # ---------------------------------------------------------------------------
    # Polygons centersand normals

    @property
    def poly_centers(self):
        """Polygons centers.
        """

        polygons = self.wrapped.polygons
        a = np.empty(len(polygons)*3, np.float)
        polygons.foreach_get("center", a)
        return np.reshape(a, (len(polygons), 3))

    @property
    def normals(self):
        """Polygons normals
        """
        
        polygons = self.wrapped.polygons
        a = np.empty(len(polygons)*3, np.float)
        polygons.foreach_get("normal", a)
        return np.reshape(a, (len(polygons), 3))

    # ---------------------------------------------------------------------------
    # Materials
    
    def copy_materials_from(self, other):
        """Copy the list of materials from another object.

        Parameters
        ----------
        other : object with materials
            The object to copy the materials from.

        Returns
        -------
        None.
        """
        
        self.wrapped.materials.clear()
        for mat in other.materials:
            self.wrapped.materials.append(mat)
            
    @property
    def material_indices(self):
        """Material indices from the faces.
        """
        
        inds = np.zeros(self.poly_count)
        self.wrapped.polygons.foreach_get("material_index", inds)
        return inds
    
    @material_indices.setter
    def material_indices(self, value):
        inds = np.resize(value, self.poly_count)
        self.wrapped.polygons.foreach_set("material_index", inds)

    # ---------------------------------------------------------------------------
    # uv management
    
    @property
    def uvmaps(self):
        return [uvl.name for uvl in self.wrapped.uv_layers]
    
    def get_uvmap(self, name, create=False):
        try:
            return self.wrapped.uv_layers[name]
        except:
            pass
        
        if create:
            self.wrapped.uv_layers.new(name=name)
            return self.wrapped.uv_layers[name]
        
        raise RuntimeError(f"WMesh error: uvmap '{name}' doesn't existe for object '{self.name}'")
    
    def create_uvmap(self, name):
        return self.get_uvmap(name, create=True)
    
    def get_uvs(self, name):
        uvmap = self.get_uvmap(name)
        
        count = len(uvmap.data)
        uvs = np.empty(2*count, np.float)
        uvmap.data.foreach_get("uv", uvs)
        
        return uvs.reshape((count, 2))
    
    def set_uvs(self, name, uvs):
        uvmap = self.get_uvmap(name)

        count = len(uvmap.data)
        uvs = np.resize(uvs, count*2)
        uvmap.data.foreach_set("uv", uvs)
        
    def get_poly_uvs(self, name, poly_index):
        uvmap = self.get_uvmap(name)
        
        poly = self.wrapped.polygons[poly_index]
        return np.array([uvmap.data[i].uv for i in poly.loop_indices])
    
    def set_poly_uvs(self, name, poly_index, uvs):
        uvmap = self.get_uvmap(name)
        
        poly = self.wrapped.polygons[poly_index]
        uvs = np.resize(uvs, (poly.loop_total, 2))
        for i, iv in enumerate(poly.loop_indices):
            uvmap.data[iv].uv = uvs[i]
            
    def get_poly_uvs_indices(self, poly_index):
        return self.wrapped.polygons[poly_index].loop_indices

    # ---------------------------------------------------------------------------
    # Set new points

    def new_geometry(self, verts, polygons=[], edges=[]):
        """Replace the existing geometry by a new one: vertices and polygons.
        
        Parameters
        ----------
        verts : array(n, 3) of floats
            The new vertices of the mesh.
        polygons : array of array of ints, optional
            The new polygons of the mesh. The default is [].
        edges : array of couples of ints, optional
            The new edges of the mesh. The default is [].

        Returns
        -------
        None.
        """

        mesh = self.wrapped
        obj  = self.owner

        # Clear
        obj.shape_key_clear()
        mesh.clear_geometry()

        # Set
        mesh.from_pydata(verts, edges, polygons)

        # Update
        mesh.update()
        mesh.validate()

    # ---------------------------------------------------------------------------
    # Detach geometry to create a new mesh
    # polygons: an array of arrays of valid vertex indices

    def detach_geometry(self, polygons, independant=False):
        """Detach geometry to create a new mesh.
        
        The polygons is an array of array of indices within the array of vertices.
        Only the required vertices are duplicated.
        
        The result can then be used to create a mesh with indenpendant meshes.

        Parameters
        ----------
        polygons : array of array of ints
            The polygons to detach.
        independant : bool, optional
            Make resulting polygons independant by duplicating the vertices instances if True.

        Returns
        -------
        array(, 3) of floats
            The vertices.
        array of array of ints
            The polygons with indices within the new vertices array.
        """

        new_verts = []
        new_polys = []
        
        if independant:
            for poly in polygons:
                new_polys.append([i for i in range(len(new_verts), len(new_verts)+len(poly))])
                new_verts.extend(poly)
                
        else:
            new_inds  = np.full(self.verts_count, -1)
            for poly in polygons:
                new_poly = []
                for vi in poly:
                    if new_inds[vi] == -1:
                        new_inds[vi] = len(new_verts)
                        new_verts.append(vi)
                    new_poly.append(new_inds[vi])
                new_polys.append(new_poly)

        return self.verts[new_verts], new_polys

    # ---------------------------------------------------------------------------
    # Copy

    def copy_mesh(self, mesh, replace=False):
        """Copy the geometry of another mesh.

        Parameters
        ----------
        mesh : a mesh object or a mesh.
            The geometry to copy from.
        replace : bool, optional
            Replace the existing geometry if True or extend the geometry otherwise. The default is False.

        Returns
        -------
        None.
        """

        wmesh = wrap(mesh)

        verts = wmesh.verts
        edges = wmesh.edge_indices
        polys = wmesh.poly_indices

        if not replace:
            x_verts = self.verts
            x_edges = self.edge_indices
            x_polys = self.poly_indices

            verts = np.concatenate((x_verts, verts))

            offset = len(x_verts)

            x_edges.extennd([(e[0] + offset, e[1] + offset) for e in edges])
            edges = x_edges

            x_polys.extend([ [p + offset for p in poly] for poly in polys])
            polys = x_polys

        self.new_geometry(verts, polys, edges)

    # ---------------------------------------------------------------------------
    # To python source code

    def python_source_code(self):
        
        def gen():
            verts = self.verts

            s      = "verts = ["
            count  = 3
            n1     = len(verts)-1
            for i, v in enumerate(verts):
                s += f"[{v[0]:.8f}, {v[1]:.8f}, {v[2]:.8f}]"
                if i < n1:
                    s += ", "

                count -= 1
                if count == 0:
                    yield s
                    count = 3
                    s = "\t"

            yield s + "]"
            polys = self.poly_indices
            yield f"polys = {polys}"

        source = ""
        for s in gen():
            source += s + "\n"

        return source


    # ---------------------------------------------------------------------------
    # Layers

    def get_floats(self, name, create=True):
        """Get values of a float layer.        

        Parameters
        ----------
        name : str
            Layer name.
        create : bool, optional
            Create the layer if it doesn't exist. The default is True.

        Returns
        -------
        vals : array of floats
            The values in the layer.
        """
        
        layer = self.wrapped.vertex_layers_float.get(name)
        if layer is None:
            if create:
                layer = self.wrapped.vertex_layers_float.new(name=name)
            else:
                return None
        count = len(layer.data)
        vals  = np.zeros(count, np.float)
        layer.data.foreach_get("value", vals)

        return vals

    def set_floats(self, name, vals, create=True):
        """set values to a float layer.        

        Parameters
        ----------
        name : str
            Layer name.
        vals: array of floats
            The values to set.
        create : bool, optional
            Create the layer if it doesn't exist. The default is True.

        Returns
        -------
        None
        """
        
        layer = self.wrapped.vertex_layers_float.get(name)
        if layer is None:
            if create:
                layer = self.wrapped.vertex_layers_float.new(name=name)
            else:
                return

        layer.data.foreach_set("value", np.resize(vals, len(layer.data)))

    def get_ints(self, name, create=True):
        """Get values of an int layer.        

        Parameters
        ----------
        name : str
            Layer name.
        create : bool, optional
            Create the layer if it doesn't exist. The default is True.

        Returns
        -------
        vals : array of ints
            The values in the layer.
        """
        
        layer = self.wrapped.vertex_layers_int.get(name)
        if layer is None:
            if create:
                layer = self.wrapped.vertex_layers_int.new(name=name)
            else:
                return None
        count = len(layer.data)
        vals  = np.zeros(count, np.int)
        layer.data.foreach_get("value", vals)

        return vals

    def set_ints(self, name, vals, create=True):
        """set values to an int layer.        

        Parameters
        ----------
        name : str
            Layer name.
        vals: array of ints
            The values to set.
        create : bool, optional
            Create the layer if it doesn't exist. The default is True.

        Returns
        -------
        None
        """
        
        layer = self.wrapped.vertex_layers_int.get(name)
        if layer is None:
            if create:
                layer = self.wrapped.vertex_layers_int.new(name=name)
                print("creation", layer)
            else:
                return
            
        layer.data.foreach_set("value", np.resize(vals, len(layer.data)))

# ---------------------------------------------------------------------------
# Spline wrapper
# wrapped : Spline

class WSpline(WStruct):
    """Spline wrapper.
    
    The wrapper gives access to the points.
    For Bezier curves, gives access to the left and right handles.
    For nurbs curves, the points are 4D vectors.
    """

    @property
    def use_bezier(self):
        """Use bezier or nurbs.

        Returns
        -------
        bool
            True if Bezier curve, False otherwise.
        """
        
        return self.wrapped.type == 'BEZIER'
    
    @property
    def count(self):
        """The number of points.

        Returns
        -------
        int
            Number of points in the Spline.
        """
        
        if self.use_bezier:
            return len(self.wrapped.bezier_points)
        else:
            return len(self.wrapped.points)
    
    @property
    def points(self):
        """The blender points of the spline.
        
        returns bezier_points or points depending on use_bezier

        Returns
        -------
        Collection
            The Blender collection corresponding to the curve type
        """
        
        if self.use_bezier:
            return self.wrapped.bezier_points
        else:
            return self.wrapped.points
        
# ---------------------------------------------------------------------------
# Bezier Spline wrapper
    
class WBezierSpline(WSpline):
    """Wraps a Bezier spline.
    
    The points of the curve can be managed with the left and right handles or not:
        - curve.verts = np.array(...)
        - curve.set_handles(points, lefts, rights)
        
    When lefts and rights handles are not given they are computed.
    """
    
    @property
    def verts(self):
        """Vertices of the curve.

        Returns
        -------
        array of vertices
            The vertices of the curve.
        """
        
        bpoints = self.wrapped.bezier_points
        count   = len(bpoints)
        pts     = np.empty(count*3, np.float)
        bpoints.foreach_get("co", pts)
        return pts.reshape((count, 3))

    @verts.setter
    def verts(self, verts):
        self.set_handles(verts)

    @property
    def lefts(self):
        """Left handles of the curve.
        
        Left handles can't be set solely. Use set_handles.
        
        Returns
        -------
        array of vertices
            The left handfles.
        """

        bpoints = self.wrapped.bezier_points
        count   = len(bpoints)
        pts     = np.empty(count*3, np.float)
        bpoints.foreach_get("handle_left", pts)
        return pts.reshape((count, 3))

    @property
    def rights(self):
        """Right handles of the curve.
        
        Right handles can't be set solely. Use set_handles.
        
        Returns
        -------
        array of vertices
            The right handfles.
        """

        bpoints = self.wrapped.bezier_points
        count   = len(bpoints)
        pts     = np.empty(count*3, np.float)
        bpoints.foreach_get("handle_right", pts)
        return pts.reshape((count, 3))

    # ---------------------------------------------------------------------------
    # Get the points and handles for bezier curves
    
    def get_handles(self):
        """Get the vertices and the handles of the curve.

        Returns
        -------
        3 arrays of vertices
            Vertices, left and right handles.
        """
        
        bl_points = self.wrapped.bezier_points
        count  = len(bl_points)

        pts    = np.empty(count*3, np.float)
        lfs    = np.empty(count*3, np.float)
        rgs    = np.empty(count*3, np.float)

        bl_points.foreach_get("co", pts)
        bl_points.foreach_get("handle_left", lfs)
        bl_points.foreach_get("handle_right", rgs)

        return pts.reshape((count, 3)), lfs.reshape((count, 3)), rgs.reshape((count, 3))


    # ---------------------------------------------------------------------------
    # Set the points and possibly handles for bezier curves

    def set_handles(self, verts, lefts=None, rights=None):
        """Set the vertices and the handles of the curve.
        
        The number of vertices can be greater than the number of existing vertices
        but it can't be lower. If lower, an exception is raised.
        
        To decrease the number of vertices, the spline must be replaced.
        To replace a curve without loosing the control points, the save and restore
        methods can be used.

        Parameters
        ----------
        verts : array of vertices
            The vertices to set.
        lefts : array of vertices, optional
            The left handles. The length of this array, if given, must match the one
            of the verts array. The default is None.
        rights : array of vertices, optional
            The rights handles. The length of this array, if given, must match the one
            of the verts array. The default is None.

        Raises
        ------
        RuntimeError
            Raise an error if the number of given vertices is less than the number
            of existing vertices.

        Returns
        -------
        None.
        """

        nvectors = np.array(verts)
        count = len(nvectors)

        bl_points = self.wrapped.bezier_points
        if len(bl_points) < count:
            bl_points.add(len(vectors) - len(bl_points))

        if len(bl_points) > count:
            raise RuntimeError(error_title % "Spline.set_handles" +
                "The number of points to set is not enough\n" +
                f"Splines points: {len(bl_points)}\n" +
                f"Input points:   {count}")

        bl_points.foreach_set("co", np.reshape(nvectors, count*3))

        if lefts is not None:
            pts = np.array(lefts).reshape(count*3)
            bl_points.foreach_set("handle_left", np.reshape(pts, count*3))

        if rights is not None:
            pts = np.array(rights).reshape(count*3)
            bl_points.foreach_set("handle_right", np.reshape(pts, count*3))

        if (lefts is None) and (rights is None):
            for bv in bl_points:
                bv.handle_left_type  = 'AUTO'
                bv.handle_right_type = 'AUTO'

        self.mark_update()

    # ---------------------------------------------------------------------------
    # As an interpolated function
        
    @property
    def function(self):
        """Returns a function interpolating this curve.        

        Returns
        -------
        PointsInterpolation
            The class can be called from 0. to 1. to get any vector on the curve..        
        """
        
        points, lefts, rights = self.get_handles()
        return PointsInterpolation(points, lefts, rights)
        
    # ---------------------------------------------------------------------------
    # Save and restore points when changing the number of vertices

    def save(self):
        """Save the vertices within a dictionnary.
        
        The result can be used with the restore function.

        Returns
        -------
        dictionnary {'type', 'verts', 'lefts', 'rights'}
            The vertices, left and right handles.
        """
        
        verts, lefts, rights = self.get_handles()
        return {"type": 'BEZIER', "verts": verts, "lefts": lefts, "rights": rights}
    
    def restore(self, data, count=None):
        """Restore the vertices from a dictionnary previously created with save.
        
        This method is typically used with a newly created Bezier curve with no point.

        Contrarily to set_handles, the number of vertices (and handles) to restore
        can be controlled with the count argument. The save / restore couple can be used to
        lower the number of control points.
        
        Typical use from WCurve:
            spline = self[index]
            data = spline.save
            self.delete(index)
            spline = self.new('BEZIER')
            spline.restore(date, target_count)
            
        Parameters
        ----------
        data : dictionnary {'type', 'verts', 'lefts', 'rights'}
            A valid dictionnary of type BEZIER.
        count : int, optional
            The number of vertices to restore. The default is None.

        Returns
        -------
        None.
        """

        if count is None:
            count = len(data["verts"])

        points = np.resize(data["verts"],  (count, 3))
        
        lefts = data.get("lefts")
        if lefts is not None:
            lefts  = np.resize(lefts,  (count, 3))
            
        rights = data.get("rights")
        if rights is not None:
            rights = np.resize(rights, (count, 3))
            
        self.set_handles(points, lefts, rights)

    # ---------------------------------------------------------------------------
    # Geometry from points

    def from_points(self, count, verts, lefts=None, rights=None):
        """Create a curve from a series of vertices.
        
        This function is similar to set_handles but here the number of control points
        can be controled with the count argument. The control points of the curve are
        computed to match the target number.

        Parameters
        ----------
        count : int
            The number of vertices for the curve.
        verts : array of vertices
            Interpolation vertices.
        lefts : array of vertices, optional
            Left handles. The default is None.
        rights : array of vertices, optional
            Right handles. The default is None.

        Returns
        -------
        None.
        """
        
        vf = PointsInterpolation(verts, lefts, rights)
        vs, ls, rs = control_points(vf, count)

        self.set_handles(vs, ls, rs)

    # ---------------------------------------------------------------------------
    # Geometry from function

    def from_function(self, count, f, t0=0, t1=1):
        """Create a curve from a function.

        Parameters
        ----------
        count : int
            The number of vertices to create.
        f : function of template f(t) --> vertex
            The function to use to create the curve.
        t0 : float, optional
            Starting value to use to compute the curve. The default is 0.
        t1 : float, optional
            Ending valud to use to compute the curve. The default is 1.

        Returns
        -------
        None.
        """
        
        dt = (t1-t0)/1000
        verts, lefts, rights = control_points(f, count, t0, t1, dt)

        self.set_handles(verts, lefts, rights)
        

# ---------------------------------------------------------------------------
# Nurbs Spline wrapper
        
class WNurbsSpline(WSpline):
    """Nurbs spline wrapper.
    
    Caution: verts are 3-vectors. To set and get the 4-verts, use verts4 property.
    """

    @property
    def verts4(self):
        """The 4-vertices of the curve.

        Returns
        -------
        array of 4-vertices
            The control points of the nurbs.
        """
        
        bpoints = self.wrapped.points
        count   = len(bpoints)
        pts     = np.empty(count*4, np.float)
        bpoints.foreach_get("co", pts)
        return pts.reshape((count, 4))

    @verts4.setter
    def verts4(self, verts):
        nverts = np.array(verts)
        count = len(nverts)

        bpoints = self.wrapped.points
        if len(bpoints) < count:
            bpoints.add(count - len(bpoints))

        if len(bpoints) > count:
            raise RuntimeError(error_title % "Spline.spline_verts" +
                "The number of points to set is not enough\n" +
                f"Splines points: {len(bpoints)}\n" +
                f"Input points:   {count}")

        bpoints.foreach_set("co", np.reshape(nverts, count*4))

        self.mark_update()
        
    @property
    def verts(self):
        """The vertices of the curve.
        
        Note that this property doesn't return the w component of the vertices.

        Returns
        -------
        array of vertices
            The control vertices of the curve.
        """
        
        return self.verts4[:, :3]
    
    @verts.setter
    def verts(self, vs):
        n = np.size(vs)//3
        v4 = np.ones((n, 4), np.float)
        v4[:, :3] =  np.reshape(vs, (n, 3))
        self.verts4 = v4
        
    # ---------------------------------------------------------------------------
    # Save and restore points when changing the number of vertices

    def save(self):
        """Save the vertices within a dictionnary.
        
        The result can be used with the restore function.

        Returns
        -------
        dictionnary {'type', 'verts4'}
            The 4-vertices.
        """
        
        return {"type": 'NURBS',  "verts4": self.verts4}
    
    def restore(data, count=None):
        """Restore the vertices from a dictionnary previously created with save.
        
        This method is typically used with a newly created curve with no point.

        Contrarily to verts4, the number of vertices to restore can be controlled
        with the count argument. The save / restore couple can be used to lower the
        number of control points.
        
        Typical use from WCurve:
            spline = self[index]
            data = spline.save
            self.delete(index)
            spline = self.new('NURBS')
            spline.restore(date, target_count)
            
        Parameters
        ----------
        data : dictionnary {'type', 'verts4'}
            A valid dictionnary of type NURBS.
        count : int, optional
            The number of vertices to restore. The default is None.

        Returns
        -------
        None.
        """

        if count is None:
            count = len(data["verts"])

        self.verts4 = np.resize(data["verts4"], (count, 4))

    # ---------------------------------------------------------------------------
    # Geometry from points

    def from_points(self, count, verts, lefts=None, rights=None):
        """Create a curve from a series of vertices.
        
        This function is similar to set_handles but here the number of control points
        can be controled with the count argument. The control points of the curve are
        computed to match the target number.

        Parameters
        ----------
        count : int
            The number of vertices for the curve.
        verts : array of vertices
            Interpolation vertices.
        lefts : array of vertices, optional
            Left handles. The default is None.
        rights : array of vertices, optional
            Right handles. The default is None.

        Returns
        -------
        None.
        """
        
        vf = PointsInterpolation(verts, lefts, rights)
        vs, ls, rs = control_points(vf, count)

        self.verts = vs

    # ---------------------------------------------------------------------------
    # Geometry from function

    def from_function(self, count, f, t0=0, t1=1):
        """Create a curve from a function.

        Parameters
        ----------
        count : int
            The number of vertices to create.
        f : function of template f(t) --> vertex
            The function to use to create the curve.
        t0 : float, optional
            Starting value to use to compute the curve. The default is 0.
        t1 : float, optional
            Ending valud to use to compute the curve. The default is 1.

        Returns
        -------
        None.
        """
        
        dt = (t1-t0)/1000
        verts, lefts, rights = control_points(f, count, t0, t1, dt)

        self.verts = verts
        
# ---------------------------------------------------------------------------
# Wrap a blender spline
#

def spline_wrapper(spline):
    """Utility function to wrap a BEZIER ot NURBS spline.

    Parameters
    ----------
    spline : Blender spline
        The spline to wrap with a WBezierSpline ot WNurbsSpline wrapper.

    Returns
    -------
    WSpline
        The wrapper of the spline.
    """
    
    return WBezierSpline(spline) if spline.type == 'BEZIER' else WNurbsSpline(spline)

# ---------------------------------------------------------------------------
# Curve wrapper
# wrapped : Curve

class WCurve(WID):
    """Curve data wrapper.
    
    In addition to wrap the Curve class, the wrapper also behaves as an array
    to give easy access to the splines.
    
    The items are wrapper of splines.
    """

    def __init__(self, wrapped, evaluated=False):
        if evaluated:
            super().__init__(wrapped, name=wrapped.name)
        else:
            super().__init__(name=wrapped.name, coll=bpy.data.curves)

    # ---------------------------------------------------------------------------
    # WCurve is a collection of splines

    def __len__(self):
        """Number of splines.

        Returns
        -------
        int
            Number fo splines.
        """
        
        return len(self.wrapped.splines)

    def __getitem__(self, index):
        """The wrapper of the indexed spline.

        Parameters
        ----------
        index : int
            Valid index within the colleciton of spines.

        Returns
        -------
        WSpline
            Wrapper of the indexed spline.
        """
        
        return spline_wrapper(self.wrapped.splines[index])

    # ---------------------------------------------------------------------------
    # Add a spline

    def new(self, spline_type='BEZIER'):
        """Create a new spline of the given type.

        Parameters
        ----------
        spline_type : str, optional
            A valide spline type. The default is 'BEZIER'.

        Returns
        -------
        spline : WSpline
            Wrapper of the newly created spline.
        """
        
        splines = self.wrapped.splines
        spline  = spline_wrapper(splines.new(spline_type))
        self.wrapped.id_data.update_tag()
        return spline

    # ---------------------------------------------------------------------------
    # Delete a spline

    def delete(self, index):
        """Delete the spline at the given index.

        Parameters
        ----------
        index : int
            Index of the spline to delete.

        Returns
        -------
        None.
        """
        
        splines = self.wrapped.splines
        if index <= len(splines)-1:
            splines.remove(splines[index])
        self.wrapped.id_data.update_tag()
        return
    
    # ---------------------------------------------------------------------------
    # Set the number of splines
    
    def set_splines_count(self, count, spline_type='BEZIER'):
        """Set the number of splines within the curve.
        
        The current number fo splines is either reduced or increased to 
        match the target count.

        Parameters
        ----------
        count : int
            Number of splines.
        spline_type : TYPE, optional
            DESCRIPTION. The default is 'BEZIER'.

        Returns
        -------
        None.
        """

        current = len(self)
        
        # Delete the splines which are too numerous
        for i in range(current - count):
            self.delete(len(self)-1)
            
        # Create new splines
        for i in range(current, count):
            self.new(spline_type)
            
        self.wrapped.id_data.update_tag()
        
    # ---------------------------------------------------------------------------
    # Set the number of vertices per spline
    
    def set_verts_count(self, count, index=None):
        """Set the number of vertices for one or all the splines.

        Parameters
        ----------
        count : int
            Number of vertices.
        index : int, optional
            Index of the spline to manage. All the splines if None. The default is None.

        Returns
        -------
        array of ints
            Indices of the modified splines.
        """
        
        # ----- All the splines or only one
        if index is None:
            splines = self
            created = [i for i in range(len(self))]
        else:
            splines = [self[index]]
            create = [index]
        
        # ----- Nothing to do
        ok = True
        for ws in splines:
            if len(ws) != count:
                ok = False
                break
            
        if ok: return created
        
        # ----- Save the existing splines
        saves = [spline.save() for spline in splines]

        # ----- Clear
        if index is None:
            self.wrapped.splines.clear()
        else:
            self.delete(index)

        # ----- Rebuild the splines
        created = []
        for save in saves:
            spline = self.new(save["type"])
            spline.restore(save, count)
            created.append(len(self)-1)

        # ---- OK
        self.wrapped.id_data.update_tag()
        
        return created


# ---------------------------------------------------------------------------
# Text wrapper
# wrapped : TextCurve

class WText(WID):
    """TextCurve wrapper.
    
    Simple wrapper limited to provide the text attribute.
    Other attributes come from the Blender TextCurve class.
    """

    def __init__(self, wrapped, evaluated=False):
        if evaluated:
            super().__init__(wrapped, name=wrapped.name)
        else:
            super().__init__(name=wrapped.name, coll=bpy.data.curves)

    @property
    def text(self):
        """The text displayed by the object.

        Returns
        -------
        str
            Text.
        """
        
        return self.wrapped.body

    @text.setter
    def text(self, value):
        self.wrapped.body = value

# ---------------------------------------------------------------------------
# Object wrapper
# wrapped: Object

class WObject(WID):
    """Blender object wrapper.
    
    Provides the wdata attributes which is the proper wrapper of the data block.
    WObject captures attributes of wdata. The following expressions are equivalent:
        - wobject.wdata.verts
        - wobject.verts
        
    In particular, wrapper of a curve object implements array access to splines wrappers.
    """

    def __init__(self, wrapped):
        
        init = True
        try:
            if wrapped.is_evaluated:
                super().__init__(wrapped)
                init = False
        except:
            pass
            
        if init:
            super().__init__(name=wrapped.name, coll=bpy.data.objects)

    # ---------------------------------------------------------------------------
    # Data

    @property
    def object_type(self):
        """Blender Object type (Mesh, Curve, Text, Empty)    

        Returns
        -------
        str
            Blender object type.
        """

        data = self.wrapped.data
        if data is None:
            return 'Empty'
        else:
            return data.__class__.__name__

    @property
    def is_mesh(self):
        """Type is mesh.

        Returns
        -------
        bool
            True if object type is Mesh.
        """
        
        return self.object_type == 'Mesh'

    @property
    def wdata(self):
        """Returns the wrapper of the data block.
        
        The type of wrapper depends upon the object type.

        Raises
        ------
        RuntimeError
            If the type is not yet supported.

        Returns
        -------
        WID
            Data wrapper.
        """

        # Empty object -> return None
        data = self.wrapped.data
        if data is None:
            return None
        
        # Supported types
        name = data.__class__.__name__
        if name == 'Mesh':
            return WMesh(data, self.wrapped.is_evaluated)
        elif name == 'Curve':
            return WCurve(data, self.wrapped.is_evaluated)
        elif name == 'TextCurve':
            return WText(data, self.wrapped.is_evaluated)
        else:
            raise RuntimeError(
                error_title % "WObject.wdata" +
                "Data class '{name}' not yet supported !"
                )
            
    # ---------------------------------------------------------------------------
    # Catch wdata attributes
            
    def __getattr__(self, name):
        
        try:
            return super().__getattr__(name)
        except:
            pass
        
        try:
            return getattr(self.wdata, name)
        except:
            raise RuntimeError(f"getattr: Attribute '{name}' doesn't exist for {self}")

    def __setattr__(self, name, value):
        try:
            super().__setattr__(name, value)
        except:
            pass
        
        try:
            setattr(self.wdata, name, value)
        except:
            raise RuntimeError(f"setattr: Attribute '{name}' doesn't exist for {self}")


    # ---------------------------------------------------------------------------
    # Array of splines
    
    def __len__(self):
        try:
            return len(self.wdata)
        except:
            pass
        
        raise RuntimeError(f"{self} is not an array.")

    def __getitem__(self, index):
        try:
            return self.wdata[index]
        except:
            pass
        
        raise RuntimeError(f"{self} is not an array.")



    # ---------------------------------------------------------------------------
    # Mesh specific
           

    def origin_to_geometry(self):
        """Utility to set the mesh origin to the geometry center.

        Raises
        ------
        RuntimeError
            If the object is not a mesh.

        Returns
        -------
        None.
        """

        wmesh = self.wdata
        if wmesh.class_name != "Mesh":
            raise RuntimeError(
                error_title % "origin_to_geometry" +
                "origin_to_geometry can only be called with a Mesh objecs"
                )

        verts = wmesh.verts
        origin = np.sum(verts, axis=0)/len(verts)
        wmesh.verts = verts - origin

        self.location = np.array(self.location) + origin

    # ---------------------------------------------------------------------------
    # Location

    @property
    def location(self):
        """Location as np.array.

        Returns
        -------
        numpy array
            The location of the object.
        """
        
        return np.array(self.wrapped.location)

    @location.setter
    def location(self, value):
        self.wrapped.location = to_shape(value, 3)

    @property
    def x(self):
        """x location.
        """
        
        return self.wrapped.location.x

    @x.setter
    def x(self, value):
        self.wrapped.location.x = value

    @property
    def y(self):
        """y location.
        """
        
        return self.wrapped.location.y

    @y.setter
    def y(self, value):
        self.wrapped.location.y = value

    @property
    def z(self):
        """z location.
        """
        
        return self.wrapped.location.z

    @z.setter
    def z(self, value):
        self.wrapped.location.z = value

    # ---------------------------------------------------------------------------
    # Scale

    @property
    def scale(self):
        """Scale as np.array.

        Returns
        -------
        numpy array
            The scale of the object.
        """
        
        return np.array(self.wrapped.scale)

    @scale.setter
    def scale(self, value):
        self.wrapped.scale = to_shape(value, 3)

    @property
    def sx(self):
        """x scale.
        """
        
        return self.wrapped.scale.x

    @sx.setter
    def sx(self, value):
        self.wrapped.scale.x = value

    @property
    def sy(self):
        """y scale.
        """
        
        return self.wrapped.scale.y

    @sy.setter
    def sy(self, value):
        self.wrapped.scale.y = value

    @property
    def sz(self):
        """z scale.
        """
        
        return self.wrapped.scale.z

    @sz.setter
    def sz(self, value):
        self.wrapped.scale.z = value

    # ---------------------------------------------------------------------------
    # Rotation in radians

    @property
    def rotation(self):
        """Euler rotation as np.array.

        Returns
        -------
        numpy array
            The euler rotation of the object.
        """
        
        return np.array(self.wrapped.rotation_euler)

    @rotation.setter
    def rotation(self, value):
        self.wrapped.rotation_euler = to_shape(value, 3)

    @property
    def rx(self):
        """x euler in rtadians.
        """
        
        return self.wrapped.rotation_euler.x

    @rx.setter
    def rx(self, value):
        self.wrapped.rotation_euler.x = value

    @property
    def ry(self):
        """y euler in rtadians.
        """
        
        return self.wrapped.rotation_euler.y

    @ry.setter
    def ry(self, value):
        self.wrapped.rotation_euler.y = value

    @property
    def rz(self):
        """z euler in rtadians.
        """
        
        return self.wrapped.rotation_euler.z

    @rz.setter
    def rz(self, value):
        self.wrapped.rotation_euler.z = value

    # ---------------------------------------------------------------------------
    # Rotation in degrees

    @property
    def rotationd(self):
        """Euler rotation as np.array in degrees.

        Returns
        -------
        numpy array
            The euler rotation of the object.
        """
        
        return np.degrees(self.wrapped.rotation_euler)

    @rotationd.setter
    def rotationd(self, value):
        self.wrapped.rotation_euler = np.radians(to_shape(value, 3))

    @property
    def rxd(self):
        """x euler in degrees.
        """
        
        return degrees(self.wrapped.rotation_euler.x)

    @rxd.setter
    def rxd(self, value):
        self.wrapped.rotation_euler.x = radians(value)

    @property
    def ryd(self):
        """y euler in degrees.
        """
        
        return degrees(self.wrapped.rotation_euler.y)

    @ryd.setter
    def ryd(self, value):
        self.wrapped.rotation_euler.y = radians(value)

    @property
    def rzd(self):
        """z euler in degrees.
        """
        
        return degrees(self.wrapped.rotation_euler.z)

    @rzd.setter
    def rzd(self, value):
        self.wrapped.rotation_euler.z = radians(value)

    # ---------------------------------------------------------------------------
    # Rotation quaternion

    @property
    def rotation_quaternion(self):
        """Quaternion as np.array.

        Returns
        -------
        numpy array
            The quaternion rotation of the object.
        """
        
        return np.array(self.wrapped.rotation_quaternion)

    @rotation_quaternion.setter
    def rotation_quaternion(self, value):
        self.wrapped.rotation_quaternion = Quaternion(value)

    # ---------------------------------------------------------------------------
    # Orientation

    def orient(self, target, axis='Z', up='Y'):
        """Orient an axis of the object along an arbitrary axis.
        
        Axis can be specified either by a vector or by a str.
        Valid axis are: 'x', '+Y', '-Z'...

        Parameters
        ----------
        target : vector or str
            The direction to orient along.
        axis : vector or str, optional
            The object axis to orient along the target axis. The default is 'Z'.
        up : vector or str, optional
            The object axis to orient upwards (along 'Z'). The default is 'Y'.

        Returns
        -------
        None.
        """
        
        # Compute the rotation quaternion
        q = q_tracker(axis, target, up=up, sky='Z', no_up = True)
        
        # Rotates the object
        mode = self.wrapped.rotation_mode
        self.wrapped.rotation_mode = 'QUATERNION'
        self.wrapped.rotation_quaternion = q
        self.wrapped.rotation_mode = mode

    # ---------------------------------------------------------------------------
    # Snapshot

    def snapshot(self, key="Wrap"):
        """Store the matrix basis in a key. 

        Parameters
        ----------
        key : str, optional
            Snapshot key. The default is "Wrap".

        Returns
        -------
        None.
        """
        
        m = np.array(self.wrapped.matrix_basis).reshape(16)
        self.wrapped[key] = m

    def restore_snapshot(self, key, mandatory=False):
        """Restore the snapshot.

        Parameters
        ----------
        key : str
            The sbapshot key.
        mandatory : bool, optional
            Raises an error if the key doesn't exist. The default is False.

        Raises
        ------
        RuntimeError
            If mandatory and the snapshot doesn't exist.

        Returns
        -------
        None.
        """
        
        m = self.wrapped.get(key)
        if m is None:
            if mandatory:
                raise RuntimeError(
                    error_title % "to_snapshot" +
                    f"The snapshot key '{key}' doesn't exist for object '{self.name}'."
                    )
            return

        m = np.reshape(m, (4, 4))
        self.wrapped.matrix_basis = np.transpose(m)

        self.mark_update()
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    # World transformations
    
    @property
    def wlocation(self):
        return bmx_location(self.wrapped.matrix_world)
    
    @property
    def wx(self):
        return bmx_x(self.wrapped.matrix_world)
    @property
    def wy(self):
        return bmx_y(self.wrapped.matrix_world)
    
    @property
    def wz(self):
        return bmx_z(self.wrapped.matrix_world)
    
    @property
    def wscale(self):
        return bmx_scale(self.wrapped.matrix_world)
        
    @property
    def wsx(self):
        return bmx_sx(self.wrapped.matrix_world)
        
    @property
    def wsy(self):
        return bmx_sy(self.wrapped.matrix_world)
        
    @property
    def wsz(self):
        return bmx_sz(self.wrapped.matrix_world)
    
    @property
    def wmatrix(self):
        return bmx_mat(self.wrapped.matrix_world)
    
    @property
    def weuler(self):
        return bmx_euler(self.wrapped.matrix_world, self.wrapped.rotation_euler.order)
    
    @property
    def wrx(self):
        return bmx_rx(self.wrapped.matrix_world, self.wrapped.rotation_euler.order)
    
    @property
    def wry(self):
        return bmx_ry(self.wrapped.matrix_world, self.wrapped.rotation_euler.order)
    
    @property
    def wrz(self):
        return bmx_rz(self.wrapped.matrix_world, self.wrapped.rotation_euler.order)
    
    @property
    def weulerd(self):
        return np.degrees(bmx_euler(self.wrapped.matrix_world, self.wrapped.rotation_euler.order))
    
    @property
    def wrxd(self):
        return np.degrees(bmx_rx(self.wrapped.matrix_world, self.wrapped.rotation_euler.order))
    
    @property
    def wryd(self):
        return np.degrees(bmx_ry(self.wrapped.matrix_world, self.wrapped.rotation_euler.order))
    
    @property
    def wrzd(self):
        return np.degrees(bmx_rz(self.wrapped.matrix_world, self.wrapped.rotation_euler.order))
    
    @property
    def wquat(self):
        return bmx_quat(self.wrapped.matrix_world)

    # -----------------------------------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------------------------------
    # Shape keys management

    # -----------------------------------------------------------------------------------------------------------------------------
    # Indexed shape key name

    @staticmethod
    def sk_name(name, step=None):
        """Stepped shape key name. 

        Parameters
        ----------
        name : str
            Base name of the shape key name.
        step : int, optional
            The step number. The default is None.

        Returns
        -------
        str
            Full name of the shape key.
        """
        
        return WShapekey.sk_name(name, step)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Has been the shape_keys structure created ?

    @property
    def has_sk(self):
        """The object has or not shape keys.
        """
        
        return self.wrapped.data.shape_keys is not None

    @property
    def shape_keys(self):
        """The Blender shapee_keys block.
        """
        
        return self.wrapped.data.shape_keys

    @property
    def sk_len(self):
        """Number of shape keys.
        """
        
        sks = self.shape_keys
        if sks is None:
            return 0
        return len(sks.key_blocks)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Get a shape key
    # Can create it if it doesn't exist

    def get_sk(self, name, step=None, create=True):
        """The a shape key by its name, and possible its step.
        
        Can create the shape key if it doesn't exist.

        Parameters
        ----------
        name : str
            Base name of the shape key.
        step : int, optional
            Step of the shape key if in a series. The default is None.
        create : bool, optional
            Create the shape key if it doesn't exist. The default is True.

        Returns
        -------
        WShapekey
            Wrapper of the shape or None if it doesn't exist.
        """

        fname = WShapekey.sk_name(name, step)
        obj   = self.wrapped
        data  = obj.data

        if data.shape_keys is None:
            if create:
                obj.shape_key_add(name=fname)
                obj.data.shape_keys.use_relative = False
            else:
                return None

        # Does the shapekey exists?

        sk = data.shape_keys.key_blocks.get(fname)

        # No !

        if (sk is None) and create:

            eval_time = data.shape_keys.eval_time

            if step is not None:
                # Ensure the value is correct
                data.shape_keys.eval_time = step*10

            sk = obj.shape_key_add(name=fname)

            # Less impact as possible :-)
            obj.data.shape_keys.eval_time = eval_time

        # Depending upon the data type

        return WShapekey(sk)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Create a shape

    def create_sk(self, name, step=None):
        """Create a shape key.
        
        Equivalent to a call to get_sk with create=True.

        Parameters
        ----------
        name : str
            Base name of the shape key.
        step : int, optional
            Step of the shape key if in a series. The default is None.

        Returns
        -------
        WShapekey
            Wrapper of the shape or None if it doesn't exist.
        """
        
        return self.get_sk(name, step, create=True)

    # -----------------------------------------------------------------------------------------------------------------------------
    # Does a shape key exist?

    def sk_exists(self, name, step):
        """Looks if a shape key exists.

        Parameters
        ----------
        name : str
            Base name of the shape key.
        step : int, optional
            Step of the shape key if in a series. The default is None.

        Returns
        -------
        bool
            True if the shape key exists.
        """
        
        return self.get_sk(name, step, create=False) is not None

    # -----------------------------------------------------------------------------------------------------------------------------
    # Set the eval_time property to the shape key

    def set_on_sk(self, name, step=None):
        """Set the evaluation time of the object on the specified shape key.
        
        The method raises an error if the shape key doesn't exist. Call sk_exists
        before for a safe call.

        Parameters
        ----------
        name : str
            Base name of the shape key.
        step : int, optional
            Step of the shape key if in a series. The default is None.

        Raises
        ------
        RuntimeError
            If the shape key doesn't exist.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """

        sk = self.get_sk(name, step, create=False)
        if sk is None:
            raise RuntimeError(
                error_title % "WObject.set_on_sk" +
                f"The shape key '{self.sk_name(name, step)}' doesn't exist in object '{self.name}'!")

        self.wrapped.data.shape_keys.eval_time = sk.frame
        return self.wrapped.data.shape_keys.eval_time

    # -----------------------------------------------------------------------------------------------------------------------------
    # Delete a shape key

    def delete_sk(self, name=None, step=None):
        """Delete a shape key.
        
        If name is None, all the shape keys are deleted.

        Parameters
        ----------
        name : str, optional
            Base name of the shape key. The default is None.
        step : int, optional
            Step of the shape key if in a series. The default is None.

        Returns
        -------
        None.
        """

        if not self.has_sk:
            return

        if name is None:
            self.wrapped.shape_key_clear()
        else:
            sk = self.get_sk(name, step, create=False)
            if sk is not None:
                self.wrapped.shape_key_remove(sk.wrapped)
                
    # -----------------------------------------------------------------------------------------------------------------------------
    # Shape_key eval time
    
    @property
    def eval_time(self):
        return self.wrapped.data.shape_keys.eval_time
    
    @eval_time.setter
    def eval_time(self, value):
        self.wrapped.data.shape_keys.eval_time = value
        
    # -----------------------------------------------------------------------------------------------------------------------------
    # Get animation from shape keys
    
    def get_sk_animation(self, eval_times):
        memo = self.eval_time
        
        verts = np.empty((len(eval_times), self.verts_count, 3), np.float)
        for i, evt in enumerate(eval_times):
            self.eval_time = evt
            verts[i] = self.evaluated.verts
        
        self.eval_time = memo
        
        return verts
        
                

# ---------------------------------------------------------------------------
# Wrapper

def wrap(name):
    """Wrap an object.
    
    To wrap an object, use this function rather than the direct class instanciation:
        - use: wobj = wrap("Cube")
        - avoid: wobj = WObject("Cube")

    Parameters
    ----------
    name : str or object with name property.
        The Blender object to wrap.

    Raises
    ------
    RuntimeError
        If the object doesn't exist.

    Returns
    -------
    WID
        The wrapper of the object.
    """

    # Nothing to wrap    
    if name is None:
        return None
    
    # The name of an object is given rather than an object instance
    if type(name) is str:
        obj = bpy.data.objects.get(name)
    else:
        obj = name
        
    # If None, it doesn't mean the object with the given name doesn't exist
    if obj is None:
        raise RuntimeError(
            error_title % "wrap" +
            f"Object named '{name}' not found"
            )
        
    # The argument is already a wrapper
    if issubclass(type(obj), WStruct):
        return obj
    
    # Initialize with the proper wrapper depending on the type of the blender ID
    cname = obj.__class__.__name__
    if cname == "Object":
        return WObject(obj)
    elif cname == "Curve":
        return WCurve(obj)
    elif cname == "Mesh":
        return WMesh(obj)
    elif cname == "TextCurve":
        return WText(obj)
    elif cname == 'Spline':
        return WSpline(obj)
    else:
        raise RuntimeError(
            error_title % "wrap" +
            f"Blender class {cname} not yet wrapped !")
