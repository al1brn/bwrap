#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 10 21:55:49 2022

@author: alain
"""

import bpy
import numpy as np

# ----------------------------------------------------------------------------------------------------
# The root class is a triplet: bone_name, attr_name, index (or array_index)
# This is a Boat for BoneAttribute
# The root class gives birth to:
# - BoatValue   : a single value for a Boat
# - BoatLibrary : a dictionnary of values, the dictionnary keys being the pose names
# - BoatCurve   : a fcurve giving value at each frame
# 
# A BoatCurve can be intialized from a BoatLibrary with an array of couple:
# (time, pose_name).
#
# An armature has a Blender pose library which manages two lists:
# - a list of pose names with an associated frame
# - a list of fcurves
#
# There is one fcurve per "Boat triplet". The fcurve has one keyframe per pose.
# The keyframes are interpreted in the following way:
# - co[0] : frame of the pose
# - co[1] : value to set to Boat
#
# The armature pose library is interpretated to build a PoseLibrary. A PoseLibrary
# is a dictionary of Poses.
#
# A Pose is a list of BoatValues.
#
# = PoseLibrary:
#   - dict[pose_name] of Pose
#     = Pose:
#       - list of BoatValue

# ----------------------------------------------------------------------------------------------------
# A value set to a couple data_path, index

class Boat():
    def __init__(self, armature, bone_name, attr_name, index):
        self.armature  = armature
        self.bone_name = bone_name
        self.attr_name = attr_name
        self.index     = index
        
    def clone(self):
        return Boat(self.armature, self.bone_name, self.attr_name, self.index)
        
    def __repr__(self):
        return f"<Boat: {self.bone_name}.{self.attr_name}[{self.index}]>"

    # ---------------------------------------------------------------------------
    # The data path
        
    @property
    def data_path(self):
        return f"pose.bones[\"{self.bone_name}\"].{self.attr_name}"
    
    @property
    def array_index(self):
        return self.index
    
    # ---------------------------------------------------------------------------
    # Symetric bone name
    
    @property
    def symmetric_name(self):
        if len(self.bone_name) < 2:
            return None
        if self.bone_name[-2:] == '.L':
            return self.bone_name[:-2] + '.R'
        if self.bone_name[-2:] == '.R':
            return self.bone_name[:-2] + '.L'
        return None
    
    # ---------------------------------------------------------------------------
    # The bone name matches another name
    
    def bone_matches(self, names):
        
        if names is None:
            return True
        
        if type(names) is str:
            names = [names]
            
        for name in names:
            if name[-2:] == '.*':
                if self.bone_name[:-2] == name[:-2]:
                    return True
            if self.bone_name == name:
                return True
            
        return False
    
    # ---------------------------------------------------------------------------
    # The attribute name matches another name
    
    def attr_matches(self, names):
        
        if names is None:
            return True
        
        if type(names) is str:
            names = [names]
            
        for name in names:
            if name == 'rotation':
                if self.attr_name in ['rotation_euler', 'rotation_quaternion', 'rotation_axis_angle']:
                    return True
            if self.attr_name == name:
                return True
            
        return False
    
    # ---------------------------------------------------------------------------
    # The attribute name matches another name
    
    def index_matches(self, indices):
        
        if indices is None:
            return True
        
        if type(indices) is int:
            return (indices == self.index) or (indices == -1)
            
        for index in indices:
            if self.index == index:
                return True
            
        return False
    
    # ---------------------------------------------------------------------------
    # Matches a selection
    
    def matches(self, bones=None, attrs=None, indices=None):
        return self.bone_matches(bones) and self.attr_matches(attrs) and self.index_matches(indices)

    # ---------------------------------------------------------------------------
    # Matches another data_path, array_index structure
    
    def matches_other(self, other):
        return (self.data_path == other.data_path) and (self.array_index == other.array_index)
    
    # ---------------------------------------------------------------------------
    # The attribute
    
    @property
    def attribute(self):
        return self.armature.path_resolve(self.data_path)
    
    # ---------------------------------------------------------------------------
    # Action for animation data
    
    @property
    def armature_action(self):
        ad = self.armature.animation_data
        if ad is None:
            self.armature.animation_data_create()
        action = self.armature.animation_data.action
        if action is None:
            action = bpy.data.actions.new(self.armature.name)
            self.armature.animation_data.action = action
            
        return action

    # ---------------------------------------------------------------------------
    # The fcurve corresponding tothe Boat
    
    def armature_fcurve(self, create=True, reset=False):
        
        action = self.armature_action
        
        fc = action.fcurves.find(self.data_path, index=self.index)

        if fc is None:
            if not create:
                return None
        elif reset:
            action.fcurves.remove(fc)
            fc = None
            
        if fc is None:
            fc = action.fcurves.new(self.data_path, index=self.index)
            
        return fc
    
# ----------------------------------------------------------------------------------------------------
# An array of boats

class Boats():
    def __init__(self, name="List of bone attributes"):
        self.boats = []
        self.name  = name
        
    def __repr__(self):
        s = f"<List of {len(self)} Boats:\n"
        for boat in self.boats:
            s += f"{boat}\n"
        return s + ">"
    
    # ---------------------------------------------------------------------------
    # As a list
    
    def __len__(self):
        return len(self.boats)
    
    def __getitem__(self, index):
        return self.boats[index]
    
    def append(self, boat):
        self.boats.append(boat)
        return boat
    
    def extend(self, boats):
        self.boats.extend(boats)
        
    # ---------------------------------------------------------------------------
    # The list of the bones in the pose
        
    def bones(self):
        bones = []
        for boat in self.boats:
            if not boat.bone_name in bones:
                bones.append(boat.bone_name)
        return bones
    
    # ---------------------------------------------------------------------------
    # The list of the attributes for a particular bone
    
    def bone_attrs(self, bone_name):
        attrs = []
        for boat in self.boats:
            if boat.bone_name == bone_name:
                if not boat.attr_name in attrs:
                    attrs.append(boat.attr_name)
        return attrs
    
    # ---------------------------------------------------------------------------
    # The array indices of a bone attribute
    
    def attr_indices(self, bone_name, attr_name, with_vals=False):
        idx  = []
        vals = [] 
        for boat in self.boats:
            if boat.bone_name == bone_name and boat.attr_name == attr_name:
                idx.append(boat.index)
                vals.append(boat.value)
        if with_vals:
            return idx, vals
        else:
            return idx
        
    # ---------------------------------------------------------------------------
    # Selection of bone attributes by bone names, attr names and indices
    # None: all
    
    def selection(self, bones=None, attrs=None, indices=None):
        sel = []
        for i, boat in enumerate(self.boats):
            if boat.matches(bones, attrs, indices):
                sel.append(i)
        return sel
    
    # ---------------------------------------------------------------------------
    # Filter
    
    def filter(self, bones=None, attrs=None, indices=None):
        sel = self.selection(bones, attrs, indices)
        res = type(self)(name=f"Filter of {self.name}")
        for i in sel:
            res.append(self[i])
        return res
    
    # ---------------------------------------------------------------------------
    # Remove a subset of the bone attributes
    
    def remove(self, bones="", attrs="", indices=999):
        
        sel = self.selection(bones, attrs, indices)
        
        old_boats = self.boats
        self.boats = []
        
        for i in range(len(old_boats)):
            if not i in sel:
                self.boats.append(old_boats[i])
        
        del old_boats

    # ---------------------------------------------------------------------------
    # Keep a subset of the bone attributes
        
    def keep(self, bones=None, attrs=None, indices=None):

        sel = self.selection(bones, attrs, indices)
        
        old_boats = self.boats
        self.boats = []
        
        for i in sel:
            self.boats.append(old_boats[i])
        
        del old_boats 
            
# ----------------------------------------------------------------------------------------------------
# A value set to a couple data_path, index

class BoatValue(Boat):
    
    def __init__(self, armature, bone_name, attr_name, index, value):
        super().__init__(armature, bone_name, attr_name, index)
        self.value = value
        
    def __repr__(self):
        return f"<BoatValue: {self.bone_name}.{self.attr_name}[{self.index}]={self.value:.3f}>"

    def clone(self):
        return BoatValue(self.armature, self.bone_name, self.attr_name, self.index, self.value)
    
    # ---------------------------------------------------------------------------
    # Symmetric
    
    def symmetric(self, inverse=False):
        name = self.symmetric_name
        if name is None:
            return None
        factor = -1 if inverse else 1
        return BoatValue(self.armature, name, self.attr_name, self.index, self.value*factor)

    # ---------------------------------------------------------------------------
    # From a fcurve
    
    @classmethod
    def FromFCurve(cls, armature, fcurve, keyframe):
        return cls(armature, fcurve.group.name, fcurve.data_path.split(".")[-1], fcurve.array_index, keyframe.co[1])
        
    # ---------------------------------------------------------------------------
    # Apply the pose
        
    def apply(self):
        if self.value is None:
            return 
        self.attribute[self.index] = self.value
        
# ----------------------------------------------------------------------------------------------------
# A pose libray manages values for bones attributes

class BoatLibrary(Boat):
    def __init__(self, armature, bone_name, attr_name, index):
        super().__init__(armature, bone_name, attr_name, index)
        self.pose_values = {}
        
    @classmethod
    def FromPoseLib(cls, poselib, bone_name, attr_name, index):
        boat =  cls(poselib.armature, bone_name, attr_name, index)
        for name, pose in poselib.items():
            sel = pose.selection(bone_name, attr_name, index)
            if len(sel) == 1:
                boat.pose_values[name] = pose[sel[0]].value
                
        return boat
    
    def __len__(self):
        return len(self.pose_values)
    
    def __getitem__(self, index):
        return self.pose_values.get(index)
    
    def keys(self):
        return self.pose_values.keys()
    
    def values(self):
        return self.pose_values.values()
    
    def items(self):
        return self.pose_values.items()
    
    def __repr__(self):
        s = f"<BoatLibrary {self.bone_name}.{self.attr_name}[{self.index}] with {len(self)} poses (constant: {self.constant_value}):\n"
        for name, val in self.items():
            s += f"    {name:20s}: {val:.3f}\n"
        return s
    
    @property
    def is_constant(self):
        if len(self) == 0:
            return True
        
        v = None
        for name, value in self.items():
            if v is None:
                v = value
            else:
                if v != value:
                    return False
        return True
    
    @property
    def constant_value(self):
        if len(self) == 0:
            return None
        
        v = None
        for name, value in self.items():
            if v is None:
                v = value
            else:
                if v != value:
                    return None
        return v
    
    def symmetric(self, inverse=False):
        bone_name = self.symmetric_name
        if bone_name is None:
            return None
        
        bl = BoatLibrary(self.armature, bone_name, self.attr_name, self.index)
        factor = -1 if inverse else 1
        for pose_name, value in self.pose_values:
            bl.pose_values[pose_name] = factor*value
                
        return bl

# ----------------------------------------------------------------------------------------------------
# A function driving a bone attribute

class BoatCurve(Boat):
    
    def __init__(self, armature, bone_name, attr_name, index):
        super().__init__(armature, bone_name, attr_name, index)
        self.points = None
        
    def __repr__(self):
        spts = ""
        sep = ""
        for pt in self.points:
            spts += sep + f"({pt[0]:.1f} {pt[1]:.3f})"
            sep = ", "
             
        return f"<BoatFunction: {self.bone_name}.{self.attr_name}[{self.array_index}] <- [{spts}]>"

    # ---------------------------------------------------------------------------
    # Initialize from (times, pose) points in a BoatLibrary
    
    @classmethod
    def FromBoatLibrary(cls, boatlib, times, factor=1.):
        bc = cls(boatlib.armature, boatlib.bone_name, boatlib.attr_name, boatlib.index)
        points = []
        for time, name in times:
            bv = boatlib[name]
            if bv is not None:
                points.append([time*factor, bv])
            
        bc.points = points
        bc.fcurve(update=True)
        
        return bc

    # ---------------------------------------------------------------------------
    # The BWrap action containing the fcurves
        
    @property
    def action(self):
        action_name = "BWrap poses"
        a = bpy.data.actions.get(action_name)
        if a is None:
            return bpy.data.actions.new(action_name)
        else:
            return a
        
    # ---------------------------------------------------------------------------
    # Access to the fcurve
        
    def fcurve(self, update=False):
        
        # ----- No points : no fcurve
        if self.points is None:
            return None
        
        # ----- Read the existing fcurve
        fc = self.action.fcurves.find(self.data_path, index=self.index)
        
        # ----- Reset or doesn' exist: must be created
        if update or (fc is None):
            if fc is not None:
                self.action.fcurves.remove(fc)
                
            fc = self.action.fcurves.new(self.data_path, index=self.index)
            
            n = len(self.points)
            fc.keyframe_points.add(count=n)
            fc.keyframe_points.foreach_set("co", np.array(self.points, float).reshape(n*2))
            fc.update()        
        
        # ----- Let's return the fcurve
            
        return fc
    
    # ---------------------------------------------------------------------------
    # Set values
    
    def add_point(self, time, value):
        if self.points is None:
            self.points = [[time, value]]
        else:
            self.points.append([time, value])
        self.fcurve(update=True)
    
    # ---------------------------------------------------------------------------
    # Apply the pose
        
    def apply(self, t):
        fc = self.fcurve()
        if fc is None:
            return
        self.attribute[self.array_index] = fc.evaluate(t)
        
    # ---------------------------------------------------------------------------
    # Set as keyframes
    
    def set_keyframes(self, start_time=0., repeat=1, factor=1.):
        
        repeat = max(1, repeat)

        # ----- Source fcurve
        fc = self.fcurve()
        
        # ----- Number fo keyframes
        n = len(fc.keyframe_points)
        if n == 0:
            return
        elif n == 1:
            repeat = 1
        count = repeat*(n-1) + 1
        
        # ----- Time interval of the source fcurve
        t0 = fc.keyframe_points[0].co[0]
        t1 = fc.keyframe_points[n-1].co[0]
        duration = (t1 - t0) * factor
        
        # ----- The target fcurve
        arm_fc = self.armature_fcurve(reset=True)
        
        # ----- Create the points
        arm_fc.keyframe_points.add(count)
        
        # ----- Loop on the repetitions
        arm_index = 0
        for rep in range(repeat):
            for ikf, kf in enumerate(fc.keyframe_points):

                go = True
                if ikf == 0:
                    go = rep == 0
                    
                if go:
                    time = start_time + duration*rep + t0 + (kf.co[0] - t0)*factor
                    arm_fc.keyframe_points[arm_index].co = (time, kf.co[1])
                    arm_index += 1

        arm_fc.update()
        
    # ---------------------------------------------------------------------------
    # Add a noise to the armature curve
        
    def add_noise(self, strength=1., scale=0., phase=0.):
        
        arm_fc = self.armature_fcurve()
        if arm_fc is None:
            return
        
        noise = arm_fc.modifiers.new(type='NOISE')
        
        noise.blend_type = 'MULTIPLY'
        noise.depth = 0
        noise.phase = phase
        noise.scale = scale
        noise.strength = strength

        
# ----------------------------------------------------------------------------------------------------
# A pose is basically a set of BoatValues

class Pose(Boats):
    
    def __init__(self, armature, name="Pose"):
        super().__init__()
        self.armature  = armature
        
    @classmethod
    def FromLibrary(cls, armature, pose_name, pose_frame, fcurves):

        pose = cls(armature, pose_name)
        for fc in fcurves:
            for kf in fc.keyframe_points:
                if kf.co[0] == pose_frame:
                    pose.boats.append(BoatValue.FromFCurve(pose.armature, fc, kf))
                    break
                
        return pose
    
    def clone(self, name="Clone"):
        pose = Pose(self.armature, pose_name=name)
        for boat in self.boats:
            pose.boats.append(boat.clone())
        return pose
                
    def __repr__(self):
        bones = self.bones()
        s = f"<Pose {self.name} with {len(self.boats)} bones attributes:\n"
        for bone in bones:
            s += f"    {bone}\n"
            attrs = self.bone_attrs(bone)
            for attr in attrs:
                idx, vals = self.attr_indices(bone, attr, with_vals=True)
                sidx = f"{idx}"
                svals = [f"{v:.3f}" for v in vals] 
                s += f"        {attr:20s}{sidx:12s} <- [{', '.join(svals)}]\n"
        s += ">"
        return s
    
    # ---------------------------------------------------------------------------
    # Append a value
    
    def append_value(self, bone_name, attr_name, index, value):
        bv = BoatValue(self.armature, bone_name, attr_name, index, value)
        self.boats.append(bv)
        return bv
        
    # ---------------------------------------------------------------------------
    # Symmetric
    
    def symmetric(self, inverse_bones=[]):
        if len(self.name) < 2:
            return None
        pose_name = None
        if self.name[-2:] == ".L":
            pose_name = self.name[:-2] + ".R"
        elif self.name[-2:] == ".R":
            pose_name = self.name[:-2] + ".L"
        if pose_name is None:
            return None
        
        sym = Pose(self.armature, pose_name=pose_name)
        for bv in self.boats:
            bvs = bv.symmetric(inverse=bv.bone_name in inverse_bones)
            if bvs is not None:
                sym.boats.append(bvs)
                
        return sym

    # ---------------------------------------------------------------------------
    # Symmetrize
    
    def symmetrize(self, inverse_bones=[]):
        new_boats = []
        for bv in self.boats:
            bvs = bv.symmetric(inverse=bv.bone_name in inverse_bones)
            if bvs is not None:
                new_boats.append(bvs)
                
        for bv in new_boats:
            self.boats.append(bv)
            
    # ---------------------------------------------------------------------------
    # Apply the pose values to the bone attributes
        
    def apply(self):
        for boat in self.boats:
            boat.apply()
            
# ----------------------------------------------------------------------------------------------------
# Animation is a list of BoatCurves

class Animation(Boats):
    
    def __init__(self, name="Animation"):
        
        super().__init__(name)
        
    @classmethod
    def FromLib(cls, boats, times, factor=1.):
        anim = cls(name=f"Animation of {boats.name}")
        
        for boat in boats:
            anim.append(BoatCurve.FromBoatLibrary(boat, times, factor=factor))
            
        return anim
            
    # ---------------------------------------------------------------------------
    # Apply at time t
            
    def apply(self, t):
        for boat in self.boats:
            boat.apply(t)
            
    # ---------------------------------------------------------------------------
    # Set keyframes
    
    def set_keyframes(self, times, start_time=0., repeat=1, factor=1):
        for boat in self.boats:
            boat.set_keyframes(start_time=start_time, repeat=repeat, factor=factor)

    # ---------------------------------------------------------------------------
    # Add noise to the armature fcurves
    
    def add_noise(self, strength=1., scale=0., phase=0.):
        for boat in self.boats:
            boat.add_noise(strength=strength, scale=scale, phase=phase)
            
            
# ----------------------------------------------------------------------------------------------------
# A pose libray: a dictionary of Poses

class PoseLib():
    def __init__(self, armature):
        self.armature = armature
        self.poses = {}
        self.constants = Pose(self.armature, "CONSTANTS")
        
    @classmethod
    def FromArmature(cls, armature):
        
        plib = cls(armature)
    
        # ----- The armature object
        try:
            pose_lib = armature.pose_library
        except:
            print(f"CAUTION: no pose lib for armature '{armature.name}'")
            return
        
        # ----- Load the poses
        for pm in pose_lib.pose_markers:
            plib.poses[pm.name] = Pose.FromLibrary(plib.armature, pm.name, pm.frame, pose_lib.fcurves)
            
        return plib
    
    @property
    def name(self):
        if self.armature is None:
            return "No armature"
        else:
            return self.armature.name
            
    def __len__(self):
        return len(self.poses)
    
    def __getitem__(self, index):
        return self.poses[index]
    
    def keys(self):
        return self.poses.keys()
    
    def values(self):
        return self.poses.values()
    
    def items(self):
        return self.poses.items()
    
    @property
    def pose_names(self):
        return self.poses.keys()
    
    def __repr__(self):
        return "<" + self.by_poses() + ">"
    
    # ---------------------------------------------------------------------------
    # Symmetrize
    
    def symmetrize(self):
        new_poses = []
        for pose in self.values():
            spose = pose.symmetric()
            if spose is not None:
                if spose.name not in self.keys():
                    new_poses.append(spose)
                    
        for spose in new_poses:
            self.poses[spose.name] = spose 
    
    # ---------------------------------------------------------------------------
    # The pose for a particular bone attribute
    
    def sub_library(self, bones=None, attrs=None):
        plib = PoseLib(self.armature)
        for name, pose in self.items():
            clone = pose.clone()
            clone.keep(bones=bones, attrs=attrs)
            if len(clone) > 0:
                plib.poses[name] = clone
        return plib
    
    # ---------------------------------------------------------------------------
    # Animated bones
    
    def bones(self):
        bones = []
        for pose in self.values():
            bns = pose.bones()
            for bn in bns:
                if not bn in bones:
                    bones.append(bn)
        return bones
    
    # ---------------------------------------------------------------------------
    # Attributes animated for a bone
    
    def bone_attrs(self, bone):
        attrs = []
        for pose in self.values():
            ats = pose.bone_attrs(bone)
            for at in ats:
                if not at in attrs:
                    attrs.append(at)
        return attrs

    # ---------------------------------------------------------------------------
    # Original
    
    def dump_original(self, filter=None):
        
        print()
        print('-'*80)
        print(f"Dump pose library of armature '{self.armature.name}'")
        print()

        pose_lib = self.armature.pose_library
        
        bones = None
        poses = None
        attrs = None
        inds  = None
        
        if filter is not None:
            bones = filter.get("bones")
            poses = filter.get("poses")
            attrs = filter.get("attrs")
            inds  = filter.get("indices")

        def get_pose(frame):
            for pm in pose_lib.pose_markers:
                if pm.frame == frame:
                    return pm.name
            return "NONE"

        for fc in pose_lib.fcurves:
            bone_name = fc.group.name
            attr_name = fc.data_path.split('.')[-1]
            
            ok = True
            if bones is not None:
                ok = ok and bone_name in bones
            if attrs is not None:
                ok = ok and attr_name in attrs
            if inds is not None:
                ok = ok and fc.array_index in inds
                
            if ok:
                print(f"fcurve {bone_name:12s} {attr_name:13s}[{fc.array_index}]")
                for ikf, kf in enumerate(fc.keyframe_points):
                    pose_name = get_pose(kf.co[0])
                    
                    ok = poses is None
                    if not ok:
                        ok = pose_name in poses
                    if ok:
                        print(f"    {ikf}: {pose_name:15s} = {kf.co[1]:.3f}")

        print('-'*20)
        print()
        
    # ---------------------------------------------------------------------------
    # Display by bone attributes : Pose > Bone > Attr
    
    def by_poses(self):
        s = f"PoseLib for '{self.name}' with {len(self)} poses:\n"
        for name, pose in self.poses.items():
            #s += f"pose '{name}:\n"
            s += f"{pose}\n"
        return s
    
    # ---------------------------------------------------------------------------
    # Display by bone attributes : Bone > Attr > Pose
    
    def by_bones(self, bones=None, attrs=None):
        
        if bones is None:
            bones = self.bones()
            
        if type(bones) is str:
            bones = [bones]
        
        s = '-'*30
        s += f"\nPosLib '{self.name}' by bones:\n\n"
        for bone in bones:
            if attrs is None:
                battrs = self.bone_attrs(bone)
            else:
                battrs = attrs
                
            s += f"Bone: {bone}:\n"
            for attr in battrs:
                s += f"    Attr: {attr}\n"
                for name, pose in self.items():
                    sel = pose.selection(bone, attr)
                    if len(sel) > 0:
                        idx = []
                        vals = []
                        for i in sel:
                            idx.append(pose.boats[i].index)
                            vals.append(pose.boats[i].value)
                            
                        sidx = f"{idx}"
                        svals = [f"{v:.3f}" for v in vals] 
                        s += f"        {name:20s}{sidx:12s} <- [{', '.join(svals)}]\n"
                
            s += "\n"
            
        return s
    
    # ---------------------------------------------------------------------------
    # Get the pose lib per bone attribute
    
    def build_bone_lib(self, bones=None, attrs=None, index=None):
        
        if bones is None:
            bones = self.bones()
            
        if type(bones) is str:
            bones = [bones]
            
        boats = Boats()
        for bone in bones:
            if attrs is None:
                battrs = self.bone_attrs(bone)
            else:
                battrs = attrs
                
            for attr in battrs:
                if index is None:
                    aidx = [0, 1, 2, 3]
                else:
                    aidx = index
                    
                for idx in aidx:
                    boatl = BoatLibrary.FromPoseLib(self, bone, attr, idx)
                    if len(boatl) > 0:
                        boats.append(boatl)
                        
        return boats
    
    # ---------------------------------------------------------------------------
    # Clean the constant poses
    # All the constant values are regrouped in one boat value
    
    def clean_constants(self):
        blib   = self.build_bone_lib()
        
        for bl in blib:
            if bl.is_constant:
                self.constants.append_value(bl.bone_name, bl.attr_name, bl.index, bl.constant_value)
                for pose in self.values():
                    pose.remove(bl.bone_name, bl.attr_name, bl.index)
                    
    # ---------------------------------------------------------------------------
    # Apply the constants
    
    def apply_constants(self):
        self.constants.apply()
        
    # ---------------------------------------------------------------------------
    # Create animation fcurves
    
    def build_animation(self, times, factor=1.):
        return Animation.FromLib(self.build_bone_lib(), times, factor)
    
 
 