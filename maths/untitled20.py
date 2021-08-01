import numpy as np

# -----------------------------------------------------------------------------------------------------------------------------
# Get the full shape of an object based on:
# - the main shape
# - the shape of the items
#
# 
#
# Exemples:
# ----- item_shape = 1
#    - 9           full_shape = ()       main_shape = ()
#    - [9]         full_shape = (1, )    main_shape = (1, )
#    - [1, 2, 3]   full_shape = (3, )    main_shape = (3, )
#
# --- item_shape = (1, )
#    - 9 --> error
#    - [9]         full_shape = (1, )    main_shape = ()
#    - [[1], [2], [3]] full_shape = (3, 1) main_shape = (3, 1)



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
    
def get_main_shape(shape, item_shape):
    
    if item_shape == ():
        return shape
    
    fsh = shape if hasattr(shape, '__len__') else (shape,)
    ish = item_shape if hasattr(item_shape, '__len__') else (item_shape,)
    if len(ish) > len(fsh):
        print("ERROR", fsh, ish)
        return None
    
    return tuple([fsh[i] for i in range(len(fsh)-len(ish))])

def reshape_main(array, main_shape, item_shape):
    shape = get_full_shape(main_shape, item_shape)
    if shape == ():
        if hasattr(array, '__len__'):
            return array[0]
        else:
            return array
    else:
        return array.reshape(shape)
    
def samples():
    shapes = [(), 1, (1, ), 3, (3, ), 10, (3, 3)]
    for msh in shapes:
        for ish in shapes:
            fsh = get_full_shape(msh, ish)
            mshs = f"{msh}"
            ishs = f"{ish}"
            fshs = f"{fsh}"
            print(f"# main: {mshs:10s}, item: {ishs:10s} --> full: {fshs:15s}")

samples()    
    
    
def tests():    
    shapes = [(), 1, (1, ), 2, (2, ), (1, 2), (3, 3)]
    for msh in shapes:
        for ish in shapes:
            fsh = get_full_shape(msh, ish)
            mshr = get_main_shape(fsh, ish)
            print(f"{msh} + {ish} --> {get_full_shape(msh, ish)} --> {mshr}{msh}")
            a = reshape_main(np.arange(np.product(fsh)), msh, ish)
            if np.product(fsh) < 10:
                print("     ", a.shape, a)
            else:
                print("   ", a.shape)
