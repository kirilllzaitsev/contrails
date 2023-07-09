from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np


def plot_contrails(false_color, human_pixel_mask, N_TIMES_BEFORE=4):
    img = false_color[..., N_TIMES_BEFORE]

    plt.figure(figsize=(18, 6))
    ax = plt.subplot(1, 3, 1)
    ax.imshow(img)
    ax.set_title("False color image")

    ax = plt.subplot(1, 3, 2)
    ax.imshow(human_pixel_mask, interpolation="none")
    ax.set_title("Ground truth contrail mask")

    ax = plt.subplot(1, 3, 3)
    ax.imshow(img)
    ax.imshow(human_pixel_mask, cmap="Reds", alpha=0.4, interpolation="none")
    ax.set_title("Contrail mask on false color image")
    # return ax


# Individual human masks
def plot_individual_human_masks(human_individual_mask):
    n = human_individual_mask.shape[-1]
    fig = plt.figure(figsize=(16, 4))
    for i in range(n):
        plt.subplot(1, n, i + 1)
        plt.imshow(human_individual_mask[..., i], interpolation="none")
    # return fig


# Animation
def plot_animation(false_color):
    fig = plt.figure(figsize=(6, 6))
    im = plt.imshow(false_color[..., 0])

    def draw(i):
        im.set_array(false_color[..., i])
        return [im]

    anim = animation.FuncAnimation(
        fig, draw, frames=false_color.shape[-1], interval=500, blit=True
    )
    plt.close()
    # display.HTML(anim.to_jshtml())
    return anim
