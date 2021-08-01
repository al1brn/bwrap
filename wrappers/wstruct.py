#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 08:46:04 2021

@author: alain
"""

import numpy as np
import bpy

from ..maths.interpolation import BCurve
from ..blender.frames import get_frame

from ..core.commons import WError, get_chained_attr, set_chained_attr


# =============================================================================================================================
# Class structure
#
# The class structure stick to the Blender classes = Struct -> ID -> Object
#
# A wrapper keeps an instance of the wrapped object
# It presents classes and methods which ease the 3D objects programmation
#
# Each wrapper also present all methods and propertiers of the wrapped object
# (python code generator is used)
#
# The Wrapper class structure is the folowing, with the Blender class between parenthesis
#
# WStruct (Struct)
# |
# +--- WSpline (Spline)
# |    |
# |    +--- WBezierSpline (Spline)
# |    |
# |    +--- WNurbsSpline (Spline)
# |    
# +--- WShapekey (Key)
# |
# +--- WID (ID)
#      |
#      +--- WMesh (Mesh)
#      |
#      +--- WCurve (Curve)
#      |
#      +--- WText (TextCurve)
#      |
#      +--- WObject (Object)
#

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
        
        self.wrapped_ = wrapped
        self.name_    = name
        self.coll_    = coll

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
    
    @property
    def name(self):
        try:
            return self.wrapped.name
        except:
            return None
        
    @name.setter
    def name(self, value):
        try:
            self.wrapped.name = value
            if self.name_ is not None:
                self.name_ = self.wrapped.name
        except:
            pass

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
                raise WError(
                    f"{name}: suffix for index must be in (x, y, z, w), not '{name[-1]}'.",
                    Class = "WStruct",
                    Method = "data_path_index",
                    name = name,
                    index = index, 
                    )
            if index >= 0 and idx != index:
                raise WError(f"Suffix of '{name}' gives index {idx} which is different from passed index {index}.",
                    Class = "WStruct",
                    Method = "data_path_index",
                    name = name,
                    index = index, 
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
            raise WError(f"The attribute '{name}' has several animation curves. Can return only one.\n" +
                    "Use dotted syntax such as 'location.x' to specify which curve to get.",
                    Class_name = "WStruct",
                    Method = "bcurve",
                    name = name
                )
            
        if len(acs) == 0:
            raise WError(f"The attribute '{name}' is not animated.",
                    "Use dotted syntax such as 'location.x' to specify which curve to get.",
                    Class_name = "WStruct",
                    Method = "bcurve",
                    name = name
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
            raise WError(
                    f"The number of fcurves to set ({len(acs)}) doesn't match the number of passed fcurves ({len(acurves)}).",
                    Class_name = "WStruct",
                    Methode = "set_acurves",
                    name = name,
                    acurves = acurves,
                    index = index)
            
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

