import numpy as np

try:
    from ..core.commons import WError
    
except:
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



# -----------------------------------------------------------------------------------------------------------------------------
# Get the full shape of an object based on:
# - the main shape
# - the shape of the items
#
# main: ()        , item: ()         --> full: ()             
# main: ()        , item: 1          --> full: (1,)           
# main: ()        , item: 3          --> full: (3,)           
# main: ()        , item: (3, 3)     --> full: (3, 3)         
# main: 1         , item: ()         --> full: (1,)           
# main: 1         , item: 1          --> full: (1, 1)         
# main: 1         , item: 3          --> full: (1, 3)         
# main: 1         , item: (3, 3)     --> full: (1, 3, 3)      
# main: 10        , item: ()         --> full: (10,)          
# main: 10        , item: 1          --> full: (10, 1)        
# main: 10        , item: 3          --> full: (10, 3)        
# main: 10        , item: (3, 3)     --> full: (10, 3, 3)     
# main: (10, 5)   , item: ()         --> full: (10, 5)        
# main: (10, 5)   , item: 1          --> full: (10, 5, 1)     
# main: (10, 5)   , item: 3          --> full: (10, 5, 3)     
# main: (10, 5)   , item: (3, 3)     --> full: (10, 5, 3, 3)  

# ---------------------------------------------------------------------------
# Get the full shape from the main and item shapes

def get_full_shape(main_shape, item_shape):
    
    if main_shape == ():
        if item_shape == ():
            return ()
        
        elif hasattr(item_shape, '__len__'):
            return tuple(item_shape)
        
        else:
            return (item_shape,)
        
    elif item_shape == ():
        if hasattr(main_shape, '__len__'):
            return tuple(main_shape)
        else:
            return (main_shape,)
        
    else:
        msh = list(main_shape) if hasattr(main_shape, '__len__') else [main_shape]
        ish = list(item_shape) if hasattr(item_shape, '__len__') else [item_shape]
        msh.extend(ish)
        
        return tuple(msh)
    
# ---------------------------------------------------------------------------
# Get the main shape from the full and item shapes
    
def get_main_shape(full_shape, item_shape):
    
    if item_shape == ():
        return full_shape
    
    fsh = full_shape if hasattr(full_shape, '__len__') else (full_shape,)
    ish = item_shape if hasattr(item_shape, '__len__') else (item_shape,)
    if len(ish) > len(fsh):
        return ()
        raise WError("The shapes are not compatible with main and item shapes management.",
                Function = "get_main_shape",
                full_shape = full_shape,
                item_shape = item_shape,
                error = "len(item_shape) > len(full_shape)"
                     )
    
    return tuple([fsh[i] for i in range(len(fsh)-len(ish))])

# ---------------------------------------------------------------------------
# Reshape an array with main and item shapes

def reshape_array(array, main_shape, item_shape):
    shape = get_full_shape(main_shape, item_shape)
    if shape == ():
        if hasattr(array, '__len__'):
            return array[0]
        else:
            return array
    else:
        return array.reshape(shape)
    
# ---------------------------------------------------------------------------
# Test if shapes can be bradcasted to a target shape
# Raise an error if a message is passed
#
# (2, 4, 8) <-- ()  =  True
# (2, 4, 8) <-- 1  =  True
# (2, 4, 8) <-- 2  =  False
# (2, 4, 8) <-- 8  =  True
# (2, 4, 8) <-- (1, 1)  =  True
# (2, 4, 8) <-- (4, 1)  =  True
# (2, 4, 8) <-- (1, 8)  =  True
# (2, 4, 8) <-- (4, 8)  =  True
# (2, 4, 8) <-- (1, 1, 1)  =  True
# (2, 4, 8) <-- (2, 1, 1)  =  True
# (2, 4, 8) <-- (1, 4, 1)  =  True
# (2, 4, 8) <-- (2, 4, 1)  =  True
# (2, 4, 8) <-- (1, 1, 8)  =  True
# (2, 4, 8) <-- (2, 1, 8)  =  True
# (2, 4, 8) <-- (1, 4, 8)  =  True
# (2, 4, 8) <-- (2, 4, 8)  =  True

def broadcastable(shape, *args, error=None):
    
    if not hasattr(shape, '__len__'):
        shape = (shape,)
        
    status = {}
        
    for sh in args:
        if hasattr(sh, '__len__'):
            if len(sh) > len(shape):
                status[f"error_{len(status)+1}"] = f"length of shape {sh} is greater than the target shape"
            else:
                for i in range(1, len(sh)+1):
                    if (sh[-i] != 1) and (sh[-i] != shape[-i]):
                        status[f"error_{len(status)+1}"] =   f"dimension {len(sh)-i} of shape {sh} can't be broadcasted"
        else:
            if len(shape) == 0:
                status[f"error_{len(status)+1}"] = f"target shape is null but {sh} isn't."
            else:
                if (sh != 1) and (sh != shape[-1]):
                    status[f"error_{len(status)+1}"] = f"shape {sh} can't be broadcasted to the last dimension of the target shape."
                    
    if len(status) == 0:
        return True
    
    if error is not None:
        s = ""
        for sh in args:
            s += f"{sh} " 
        raise WError("Impossible to broadcast array shapes",
                Context = error(),
                target_shape = shape,
                shapes = s,
                **status)
    
    return False
    
# ---------------------------------------------------------------------------
# Combine shaps to get the shape for which each one can be broadcasted
# 
# If error message is not None, the "broadcastibility" of the shapes is tested.
#
# (1, 3) (3, 4, 5) --> (3, 4, 5) but broadcast will fail
#
# Use broadcastable function for test
#
# () ()  -->  ()
# 1 3  -->  (3,)
# () 3  -->  (3,)
# (1, 1) 3  -->  (1, 3)
# (1, 3) (5, 1)  -->  (5, 3)
# (1, 3) (5, 1) (10, 11, 1)  -->  (10, 11, 3)

def broadcast_shape(*args, error = None):
    
    # ----- Special algo for only two arguments
    
    if len(args) == 2:
        
        # ----- The two are equal
        
        if args[0] == args[1]:
            return args[0]
        
        # ----- Buld the list with the max length and the max items
        
        if hasattr(args[0], '__len__'):
            if hasattr(args[1], '__len__'):
                
                # two tuples
                
                if len(args[0]) >= len(args[1]):
                    shape = list(args[0])
                    other = args[1]
                else:
                    shape = list(args[1])
                    other = args[0]
                    
                for i in range(1, len(other)+1):
                    shape[-i] = max(shape[-i], other[-i])
            else:
                
                # tuple and int
                
                shape = list(args[0])
                if len(shape) == 0:
                    return (args[1],)
                
                shape[-1] = max(shape[-1], args[1])
        else:
            if hasattr(args[1], '__len__'):
                
                # int and tuple
                
                shape = list(args[1])
                if len(shape) == 0:
                    return (args[1],)
                
                shape[-1] = max(shape[-1], args[0])
            else:
                
                # two ints
                
                shape = (max(args[0], args[1]),)
        
    else:
        
        # ----- Algorithm for more than two shapes
            
        shape = []
        for sh in args:
            
            if hasattr(sh, '__len__'):
                new_shape = list(sh)
            else:
                new_shape = [sh]
                
            if len(new_shape) > len(shape):
                for i in range(1, len(shape)+1):
                    new_shape[-i] = max(new_shape[-i], shape[-i])
                shape = new_shape
            else:
                for i in range(1, len(new_shape)+1):
                    shape[-i] = max(new_shape[-i], shape[-i])
                    
    # Broadcastable ?
                    
    if error is None:
        return tuple(shape)
    else:
        shape = tuple(shape)
        broadcastable(shape, *args, error=error)
        return shape
    

            
