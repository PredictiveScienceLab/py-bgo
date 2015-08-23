"""
The function `colorAlpha_to_rgb` returns a list of RGB color that mimic 
a RGBA on a given background.
The code implements the algorithm from 
[this stackoverflow post](http://stackoverflow.com/questions/2049230/convert-rgba-color-to-rgb?rq=1y%)

The code has not been much tested. 
A by eye comparison between a pdf image with alpha channel and 
a few values for the input color/alpha on white background shows that this approach is 'good enough'.

Dependances:
    Numpy
    Matplotlib

Licence:
    This a free sofware and come without any warranty.
    It can be freely used, modified and redistributed, 
    with the only requirement that the author (Francesco Montesano) 
    and the inspiration for the algorithm
    (http://stackoverflow.com/questions/2049230/convert-rgba-color-to-rgb?rq=1)
    are aknowledged.

"""

from matplotlib.colors import colorConverter as cC
import numpy as np

__version__ = "0.21"
__author__ = "Francesco Montesano (franz.bergesund@gmail.com)"

__all__ = ["colorAlpha_to_rgb"]

def _to_rgb(c):
    """
    Convert color *c* to a numpy array of *RGB* handling exeption
    Parameters
    ----------
    c: Matplotlib color
        same as *color* in *colorAlpha_to_rgb*
    output
    ------
    rgbs: list of numpy array
        list of c converted to *RGB* array
    """

    if(getattr(c, '__iter__', False) == False):  #if1: if c is a single element (number of string)
        rgbs = [np.array(cC.to_rgb(c)),]  #list with 1 RGB numpy array

    else:  #if1, else: if is more that one element

        try:   #try1: check if c is numberic or not
            np.array(c) + 1

        except (TypeError, ValueError):  #try1: if not numerics is not (only) RGB or RGBA colors
            #convert the list/tuble/array of colors into a list of numpy arrays of RGB
            rgbs = [np.array( cC.to_rgb(i)) for i in c]

        except Exception as e:  #try1: if any other exception raised
            print("Unexpected error: {}".format(e))
            raise e #raise it

        else:  #try1: if the colors are all numberics

            arrc = np.array(c)  #convert c to a numpy array
            arrcsh = arrc.shape  #shape of the array 

            if len(arrcsh)==1:  #if2: if 1D array given 
                if(arrcsh[0]==3 or arrcsh[0]==4):  #if3: if RGB or RBGA
                    rgbs = [np.array(cC.to_rgb(c)),]  #list with 1 RGB numpy array
                else:   #if3, else: the color cannot be RBG or RGBA
                    raise ValueError('Invalid rgb arg "{}"'.format(c))
                #end if3
            elif len(arrcsh)==2:  #if2, else: if 2D array
                if(arrcsh[1]==3 or arrcsh[1]==4):  #if4: if RGB or RBGA
                    rgbs = [np.array(cC.to_rgb(i)) for i in c]  #list with RGB numpy array
                else:   #if4, else: the color cannot be RBG or RGBA
                    raise ValueError('Invalid list or array of rgb')
                #end if4
            else:  #if2, else: if more dimention
                raise ValueError('The rgb or rgba values must be contained in a 1D or 2D list or array')
            #end if2
        #end try1
    #end if1

    return rgbs

def _is_number(s):
    """
    Check if *c* is a number (from
    http://stackoverflow.com/questions/354038/how-do-i-check-if-a-string-is-a-number-in-python)
    Parameters
    ----------
    c: variable
    output
    ------
    true if c is a number
    false otherwise
    """
    try:
        float(s) # for int, long and float
    except ValueError:
        return False
    return True

def _check_alpha(alpha, n):
    """
    Check if alpha has one or n elements and if they are numberics and between 0 and 1
    Parameters
    ----------
    alpha: number or list/tuple/numpy array of numbers
        values to check
    output
    ------
    alpha: list of numbers 
        if all elements numberics and between 0 and 1
    """
    alpha = np.array(alpha).flatten()  #convert alpha to a flattened array
    if(alpha.size == 1):  #if1: alpha is one element
        if(_is_number(alpha) == False or alpha < 0 or alpha > 1):
            raise ValueError("'alpha' must be a float with value between 0 and 1, included") 
        else:
            alpha = [alpha for i in range(n)]  #replicate the alphas len(colors) times
    elif(alpha.size==n):  #if1, else: if alpha is composed of len(colors) elements
        try:  #check if all alphas are numbers
            alpha+1 
        except TypeError:
            raise ValueError("All elements of alpha must be a float with value between 0 and 1, included") 
        else:
            if((alpha < 0).any() or (alpha > 1).any()):
                raise ValueError("'alpha' must be a float with value between 0 and 1, included") 
    else:  #if1, else: if none of the previous cases
        raise ValueError("Alpha must have either one element or as many as 'colors'")
    #end if1
    return alpha

def colorAlpha_to_rgb(colors, alpha, bg='w'):
    """
    Given a Matplotlib color and a value of alpha, it returns 
    a RGB color which mimic the RGBA colors on the given background

    Parameters
    ----------
    colors: Matplotlib color (documentation from matplotlib.colors.colorConverter.to_rgb), 
        list/tuple/numpy array of colors
        Can be an *RGB* or *RGBA* sequence or a string in any of
        several forms:
        1) a letter from the set 'rgbcmykw'
        2) a hex color string, like '#00FFFF'
        3) a standard name, like 'aqua'
        4) a float, like '0.4', indicating gray on a 0-1 scale
        if *color* is *RGBA*, the *A* will simply be discarded.
    alpha: float [0,1] or list/tuple/numpy array with len(colors) elements
        Value of alpha to mimic. 
    bg: Matplotlib color (optional, default='w')
        Color of the background. Can be of any type shown in *color*

    output
    ------
    rgb: *RGB* color 

    example
    -------

    import mimic_alpha as ma

    print(ma.colorAlpha_to_rgb('r', 0.5))
    >>> [array([ 1. ,  0.5,  0.5])]
    print(ma.colorAlpha_to_rgb(['r', 'g'], 0.5)) 
    >>> [array([ 1. ,  0.5,  0.5]), array([ 0.5 ,  0.75,  0.5 ])]
    print(ma.colorAlpha_to_rgb(['r', 'g'], [0.5, 0.3])) 
    >>> [array([ 1. ,  0.5,  0.5]), array([ 0.7 ,  0.85,  0.7 ])]
    print(ma.colorAlpha_to_rgb(['r', [1,0,0]], 0.5)) 
    >>> [array([ 1. ,  0.5,  0.5]), array([ 1. ,  0.5,  0.5])]
    print( ma.colorAlpha_to_rgb([[0,1,1], [1,0,0]], 0.5) ) 
    >>> [array([ 0.5,  1. ,  1. ]), array([ 1. ,  0.5,  0.5])]
    print(ma.colorAlpha_to_rgb(np.array([[0,1,1], [1,0,0]]), 0.5)) 
    >>> [array([ 0.5,  1. ,  1. ]), array([ 1. ,  0.5,  0.5])]
    print(ma.colorAlpha_to_rgb(np.array([[0,1,1], [1,0,0]]), 0.5, bg='0.5')) 
    >>> [array([ 0.25,  0.75,  0.75]), array([ 0.75,  0.25,  0.25])]
    """

    colors = _to_rgb(colors)  #convert the color and save in a list of np arrays
    bg = np.array(cC.to_rgb(bg))   #convert the background

    #check if alpha has 1 or len(colors) elements and return a list of len(color) alpha 
    alpha = _check_alpha(alpha, len(colors))  
    #interpolate between background and color 
    rgb = [(1.-a) * bg + a*c for c,a in zip(colors, alpha)]

    return rgb

if __name__ == "__main__":
    print(colorAlpha_to_rgb('r', 0.5))
    print(">>> [array([ 1. ,  0.5,  0.5])]")
    print(colorAlpha_to_rgb(['r', 'g'], 0.5)) 
    print(">>> [array([ 1. ,  0.5,  0.5]), array([ 0.5 ,  0.75,  0.5 ])]")
    print(colorAlpha_to_rgb(['r', 'g'], [0.5, 0.3])) 
    print(">>> [array([ 1. ,  0.5,  0.5]), array([ 0.7 ,  0.85,  0.7 ])]")
    print(colorAlpha_to_rgb(['r', [1,0,0]], 0.5)) 
    print(">>> [array([ 1. ,  0.5,  0.5]), array([ 1. ,  0.5,  0.5])]")
    print(colorAlpha_to_rgb([[0,1,1], [1,0,0]], 0.5) ) 
    print(">>> [array([ 0.5,  1. ,  1. ]), array([ 1. ,  0.5,  0.5])]")
    print(colorAlpha_to_rgb(np.array([[0,1,1], [1,0,0]]), 0.5)) 
    print(">>> [array([ 0.5,  1. ,  1. ]), array([ 1. ,  0.5,  0.5])]")
    print(colorAlpha_to_rgb(np.array([[0,1,1], [1,0,0]]), 0.5, bg='0.5')) 
    print(">>> [array([ 0.25,  0.75,  0.75]), array([ 0.75,  0.25,  0.25])]")
    
    exit()

