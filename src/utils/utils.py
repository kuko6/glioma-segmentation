import numpy as np
from matplotlib.colors import ListedColormap

# -------------------------------------------------------------------------------- #
# Additional helper functions.
# Includes: 
#   - custom cmap used for mask visualization
# -------------------------------------------------------------------------------- #

def get_custom_cmap():
    # black, green, yellow, blue
    # colorarray = [
    #     [0/256, 0/256, 0/256, 1], # Background
    #     [105/256, 173/256, 212/256, 1], # Necrotic
    #     [114/256, 195/256, 116/256, 1], # Edema
    #     [254/256, 249/256, 9/256, 1], # Enhancing
    # ]

    # black, blue, yellow, green
    # colorarray = [
    #   [0/256, 0/256, 0/256, 1], # Background
    #   [0/256, 224/256, 119/256, 1], # Necrotic
    #   [0/256, 132/256, 251/256, 1], # Edema
    #   [252/256, 239/256, 2/256, 1], # Enhancing
    # ]

    # black, blue, orange, green
    # colorarray = [
    #   [0/256, 0/256, 0/256, 1], # Background
    #   [0/256, 250/256, 131/256, 1], # Necrotic
    #   [6/256, 127/256, 239/256, 1], # Edema 
    #   [250/256, 129/256, 42/256, 1], # Enhancing
    # ]

    # black, blue, red, green
    colorarray = [
        [0/256, 0/256, 0/256, 1], # Background
        [0/256, 250/256, 131/256, 1], # Necrotic
        [0/256, 132/256, 251/256, 1], # Edema
        [252/256, 56/256, 56/256, 1], # Enhancing
    ]
    cmap = ListedColormap(colorarray)

    return cmap