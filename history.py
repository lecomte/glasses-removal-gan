import numpy as np


class History:
    def __init__(self, size):
        """ Initalizes history of refined images.

        Arguments:
            size (int): fixed size of the history"""
        self.size = size
        self.history = np.empty([0, 128, 128, 1])

    def update_history(self, images):
        """ Updates history of refined images.

        Arguments:
            images (np.array): images to be added to the history"""
        excess = len(self.history) + len(images) - self.size
        excess = max(excess, 0)
        np.random.shuffle(self.history)
        self.history = np.delete(self.history, range(excess), axis=0)
        self.history = np.concatenate((self.history, images))

    def get_images(self, ammount):
        """ Get images from the history.

        Arguments:
            ammount (int): number of images to be returned"""
        assert ammount <= len(
            self.history
        ), "There aren't enough images in the history to fulfill the request"
        np.random.shuffle(self.history)
        return self.history[:ammount]
