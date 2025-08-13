import os
import sys

curfile = __file__
proj_root = os.path.dirname(os.path.dirname(curfile))
if proj_root not in sys.path:
    sys.path.append(proj_root)
