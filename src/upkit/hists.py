# -*- coding: utf-8 -*-
#%%
"""
Created on Tue Jul 16 14:28:12 2024

@author: mleon
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import matplotlib as mpl
import tools
from fit import Fit


np.seterr(over='ignore', invalid='ignore', divide='ignore')


def set_plot_style():
    params = {
        "figure.figsize": (12, 9),  # Larger figure size
        "axes.titlesize": 35,  # Title font size
        "legend.fontsize": 25,  # Legend font size
        "lines.linewidth": 3,  # Line width
        "lines.markersize": 8,  # Marker size
        "axes.grid": False,  # Show grid by default
        "grid.alpha": 0.75,  # Grid transparency
        "grid.color": "#cccccc",  # Grid color
        'axes.linewidth': 2.,
        'axes.labelsize': 28,
        'xtick.labelsize': 28,
        'ytick.labelsize': 28,
        'xtick.major.width': 1.,
        'xtick.major.size': 6,
        'ytick.major.width': 1.,
        'ytick.major.size': 6,
        'xtick.minor.width': 1.,
        'xtick.minor.size': 3,
        'ytick.minor.width': 1.,
        'ytick.minor.size': 3,
        'xtick.major.pad': 2,
        'ytick.major.pad': 2,
        'text.usetex': False,
        'font.size': 20
    }

    plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \usepackage{amssymb} \usepackage{amsfonts}\usepackage{bm}"
    plt.rcParams.update(params)
    plt.rc("font", family="Times New Roman")


def close_all():
    plt.close('all')


class Histo:
    def __init__(self, data=None, bin_edges=None, bin_content=None, bin_error=None, bins=None, range=None, **kwargs):
        ### If data is given to plot ###
        self.data = data
        self.bin_content = bin_content
        self.bin_edges = bin_edges
        self.bin_error = bin_error
        self.ax = None
        self.range = range
        if data is not None:
            ### If data is given to plot ###
            if (bins is None) and (range is None):
                self.plot(**kwargs)

            elif (bins is not None) and (range is None):
                self.plot(bins=bins, **kwargs)

            elif (bins is None) and (range is not None):
                self.plot(range=range, **kwargs)

            else:
                self.plot(bins=bins, range=range, **kwargs)

        elif bin_edges is not None and bin_content is not None:
            ### If histogram values are given to plot ###
            self.plot(bins=bins, range=range, **kwargs)

        else:
            raise ValueError('No data given!')


    def plot(self, ax=None, **kwargs):
        """

        Parameters
        ----------
        x : TYPE
            DESCRIPTION.
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if ax is None:
            fig, ax = plt.subplots()
        
        self.ax = ax

        data_range = kwargs.get('range', None)

        if data_range is not None and self.data is not None:
            min_val, max_val = data_range
            filtered_data = self.data[(self.data >= min_val) & (self.data <= max_val)]
        
        else:
            filtered_data = self.data

        label = kwargs.get('label', None)
        self.label = label

        if kwargs.get('label', None) is None:
            if filtered_data is not None:
                kwargs['label'] = f"Events: {len(filtered_data)}"
            else:
                kwargs['label'] = "Events: {:.0f}".format(np.sum(self.bin_content))

        if filtered_data is not None:
            h, bin_edges, _ = ax.hist(
                filtered_data, **kwargs)
            errors = self.bin_error if self.bin_error is not None else np.sqrt(abs(h))

        else:
            h, bin_edges, _ = ax.hist(
                self.bin_edges[:-1], weights=self.bin_content, **kwargs)
            errors = self.bin_error if self.bin_error is not None else np.sqrt(abs(h))

        self.res = {"bin_content": h,
                    "bin_edges": bin_edges, "bin_errors": errors}
    
        plt.tight_layout()

    def plot_exp(self, ax=None, **kwargs):
        """

        Parameters
        ----------
        ax : Axes
            The axes to plot on.

        Returns
        -------
        None.

        """

        if ax is None:
            ax = self.ax

        bin_centers = (self.res['bin_edges'][:-1] +
                       self.res['bin_edges'][1:]) / 2

        self.bin_centers = bin_centers
        if self.bin_error is not None:
            y_err = self.bin_error
        elif 'bin_errors' in self.res:
            y_err = self.res['bin_errors']
        else:
            y_err = np.sqrt(self.res['bin_content'])

        ax.errorbar(bin_centers, self.res['bin_content'], yerr=y_err, **kwargs)

    def add_histo(self, h1: "Histo", c1: float, **kwargs):
        """

        Parameters
        ----------
        h1 : Histo
            The first histogram to add.
        c1 : float
            The coefficient for the first histogram.

        Returns
        -------
        Histo
            The resulting histogram after addition.

        """

        if np.array_equal(self.res['bin_edges'], h1.res['bin_edges']):

            tot_content = c1 * h1.res["bin_content"] + self.res['bin_content']
            tot_edges = h1.res["bin_edges"]
            bin_err = np.sqrt(self.res['bin_errors']**2 + (c1*h1.res['bin_errors'])**2)
            return Histo(data = None, bin_edges=tot_edges, bin_content=tot_content, bin_error=bin_err, **kwargs)

        else: 
            raise ValueError('Incompatible histogram bin edges!')


    def __add__(self, other):
        if not isinstance(other, Histo):
            raise ValueError("Incompatible histogram types!")
        
        elif isinstance(other, Histo):
            return self.add_histo(other, 1)

        else:
            raise ValueError('Incompatible histogram bin edges!')

    def __sub__(self, other):
        if not isinstance(other, Histo):
            raise ValueError("Incompatible histogram types!")

        elif isinstance(other, Histo):
            return self.add_histo(other, -1)

        else:
            raise ValueError('Incompatible histogram bin edges!')
        
    def __mul__(self, factor):
        if isinstance(factor, (int, float)):
            return self.scale(factor)
        else:
            raise ValueError("Invalid factor type!")

    def __rmul__(self, factor):
        return self.__mul__(factor)

    def __truediv__(self, factor):
        if isinstance(factor, (int, float)):
            return self.scale(1 / factor)
        else:
            raise ValueError("Invalid factor type!")

    def scale(self, factor):
        scaled_content = factor * self.res['bin_content']
        scaled_errors = factor * self.res['bin_errors']
        return Histo(data=None, bin_edges=self.res['bin_edges'], bin_content=scaled_content, bin_error=scaled_errors)

    def normalize(self, norm_factor=1.0):
        total = np.sum(self.res['bin_content'])
        if total == 0:
            raise ValueError("Cannot normalize histogram with zero total content.")

        factor = norm_factor / total

        return self.scale(factor)

    # def rebin(self, bins):
    #     """
    #     Rebin the histogram to a new number of bins.
    #     If original data is available, rebin from data.
    #     Otherwise, rebin from binned contents.
    #     """
    #     new_edges = np.linspace(self.res['bin_edges'][0], self.res['bin_edges'][-1], bins + 1)
    #     if self.data is not None:
    #         # Rebin from original data
    #         new_content, _ = np.histogram(self.data, bins=new_edges)
    #         new_errors = np.sqrt(new_content)
    #     else:
    #         # Rebin from binned contents
    #         # Expand the original bin contents into data points at bin centers
    #         bin_centers = (self.res['bin_edges'][:-1] + self.res['bin_edges'][1:]) / 2
    #         expanded = np.repeat(bin_centers, self.res['bin_content'].astype(int))
    #         new_content, _ = np.histogram(expanded, bins=new_edges)
    #         new_errors = np.sqrt(new_content)
    #     return Histo(data=None, bin_edges=new_edges, bin_content=new_content, bin_error=new_errors, range=self.range)

    def show_hists(self, xlabel=None, ylabel=None, title=None, save_name=None, legend=None, ax=None):
        """

        Parameters
        ----------
        xlabel : TYPE, optional
            DESCRIPTION. The default is None.
        ylabel : TYPE, optional
            DESCRIPTION. The default is None.
        title : TYPE, optional
            DESCRIPTION. The default is None.
        save_name : TYPE, optional
            DESCRIPTION. The default is None.
        legend : TYPE, optional
            DESCRIPTION. The default is None.
        ax : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        if ax is None:
            ax = self.ax

        if legend is not None:
            ax.legend(legend)
        else:
            ax.legend() 

        ### Titles and Legend ###
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.tight_layout()

        ### Save Figure ###
        if save_name is not None:
            plt.savefig(save_name)

        plt.show()

    def save_hist(self, file_name:str):
        """
        Save histogram data to a .npz file.

        Parameters
        ----------
        file_name : str
            The name of the output file (should end with .npz).

        Returns
        -------
        None.

        """
        if not file_name.endswith('.npz'):
            raise ValueError("File name must end with .npz")

        np.savez(file_name, bin_edges=self.res['bin_edges'], bin_content=self.res['bin_content'], bin_errors=self.res['bin_errors'])

    @classmethod
    def load_hist(cls, file_name: str):
        """
        Load histogram data from a .npz file.

        Parameters
        ----------
        file_name : str
            The name of the input file (should end with .npz).

        Returns
        -------
        Histo
            An instance of the Histo class with loaded data.

        """
        if not file_name.endswith('.npz'):
            raise ValueError("File name must end with .npz")

        data = np.load(file_name)
        return cls(
            bin_edges=data['bin_edges'],
            bin_content=data['bin_content'],
            bin_errors=data['bin_errors']
        )


class Histo2D:
    def __init__(self, xdata=None, ydata=None, x_edges=None, x_content=None, y_edges=None, y_content=None, bins=None, range=None, ax=None, plot=True, **kwargs):
        self.ax = ax
        self.bins = bins
        self.range = range

        if x_edges is not None and x_content is not None:
            x_centers = (x_edges[:-1] + x_edges[1:]) / 2
            xdata_bins = np.repeat(x_centers, x_content.astype(int))
        else:
            xdata_bins = None

        if y_edges is not None and y_content is not None:
            y_centers = (y_edges[:-1] + y_edges[1:]) / 2
            ydata_bins = np.repeat(y_centers, y_content.astype(int))
        else:
            ydata_bins = None

        self.xdata = xdata if xdata is not None else xdata_bins
        self.ydata = ydata if ydata is not None else ydata_bins

        if self.xdata is not None and self.ydata is not None:
            xlen, ylen = len(self.xdata), len(self.ydata)
            if xlen != ylen:
                raise ValueError(f"xdata and ydata must have the same length ({xlen} != {ylen})")
            
        if plot and self.xdata is not None and self.ydata is not None:
            self.plot(bins=self.bins, range=self.range, **kwargs)

    def plot(self, ax=None, **kwargs):
        """


        Parameters
        ----------
        xdata : TYPE
            DESCRIPTION.
        ydata : TYPE
            DESCRIPTION.
        ax : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """

        if self.ax is None:
            fig, ax = plt.subplots()

        self.ax = ax

        if self.ydata is not None and self.xdata is not None:
            h, x_edges, y_edges, img = ax.hist2d(np.array(self.xdata), np.array(self.ydata),
                                                 label=f"x Entries = {len(self.xdata)}\n y Entries = {len(self.ydata)}", **kwargs)

            plt.colorbar(img, ax=ax)
            plt.tight_layout()

            self.res = {"bin_content": h.T, "x_edges": x_edges, "y_edges": y_edges}

    def profile_function(self, bins=None, range: tuple = None):
        """
        Creates a profile of the 2D histogram along the x-axis.
        Returns the mean and standard deviation of y-values in each x-bin.

        Parameters
        ----------
        bins : int or sequence, optional
            Number of bins or bin edges along x-axis. Defaults to self.bins.
        range : tuple, optional
            x-axis range for binning. Defaults to self.range.

        Returns
        -------
        x_centers : np.ndarray
            Centers of the x bins.
        y_means : np.ndarray
            Mean y value in each x bin.
        y_std : np.ndarray
            Standard deviation of y values in each x bin.
        """
        if bins is None:
            bins = self.bins
        if range is None:
            range = self.range

        if isinstance(self.bins, list):
            bins = self.bins[1]

        else:
            bins = self.bins

        if isinstance(self.range, list):
            range = self.range[1]
        else:
            range = self.range

        # Compute bin edges
        x_edges = np.linspace(range[0], range[1], bins + 1) if bins is not None else self.res['x_edges']
        x_centers = (x_edges[:-1] + x_edges[1:]) / 2

        y_means, y_std = [], []

        for i in range(len(x_edges) - 1):
            y_slice = self.ydata[(self.xdata >= x_edges[i]) & (self.xdata < x_edges[i + 1])]
            if len(y_slice) > 0:
                y_means.append(np.mean(y_slice))
                y_std.append(np.std(y_slice))
            else:
                y_means.append(np.nan)
                y_std.append(np.nan)

        y_means = np.array(y_means)
        y_std = np.array(y_std)

        return x_centers, y_means, y_std
    
    def show_hists(self, xlabel=None, ylabel=None, title=None, save_name=None, legend=None, ax=None):
        """


        Parameters
        ----------
        xlabel : TYPE, optional
            DESCRIPTION. The default is None.
        ylabel : TYPE, optional
            DESCRIPTION. The default is None.
        title : TYPE, optional
            DESCRIPTION. The default is None.
        save_name : TYPE, optional
            DESCRIPTION. The default is None.
        legend : TYPE, optional
            DESCRIPTION. The default is None.
        ax : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        """

        if ax is None:
            ax = self.ax

        if legend is None:
            ax.legend()

        else:
            ax.legend(legend)

        ### Titles and Legend ###
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.tight_layout()

        ### Save Figure ###
        if save_name is not None:
            ax.figure.savefig(save_name)

        plt.show()

    def plot_surface(self, **kwargs):
        """


        Returns
        -------
        None.

        """

        # Create 2D histogram
        if self.res["bin_content"] is None:
            self.plot(self.xdata, self.ydata)

        # Convert bin edges to bin centers
        x_centers = (self.res["x_edges"][:-1] + self.res["x_edges"][1:]) / 2
        y_centers = (self.res["y_edges"][:-1] + self.res["y_edges"][1:]) / 2
        x_centers, y_centers = np.meshgrid(x_centers, y_centers)

        # Create the 3D surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        # Transpose the histogram data to match the coordinates
        hist = self.res["bin_content"].T

        # Plot surface
        surf = ax.plot_surface(x_centers, y_centers, hist, **kwargs)
        self.surf = surf

        # Add a color bar which maps values to colors
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        cbar.set_label("Frequency")

    def plot_bar3d(self, xlabel=None, ylabel=None, zlabel=None, **kwargs):
        """
        Plots a 3D histogram as a bar plot with a colormap and color bar.

        Parameters
        ----------
        ax : matplotlib.axes._subplots.Axes3DSubplot, optional
            The axis to plot on. If None, a new figure and axis are created.
        **kwargs : dict, optional
            Additional keyword arguments to pass to the bar3d plotting function.

        Returns
        -------
        None
        """

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


        if self.res["bin_content"] is None:
            self.plot()  # Ensure bin contents are generated

        # Flatten the bin centers and histogram values for bar3d plotting
        x_centers = (self.res["x_edges"][:-1] + self.res["x_edges"][1:]) / 2
        y_centers = (self.res["y_edges"][:-1] + self.res["y_edges"][1:]) / 2
        x_pos, y_pos = np.meshgrid(x_centers, y_centers)
        self.x_pos = x_pos
        self.y_pos = y_pos
        z_pos = np.zeros_like(x_pos)

        # The size of each bar is determined by the bin width and the histogram value
        dx = dy = np.ones_like(z_pos) * (
            self.res["x_edges"].ravel()[1] - self.res["x_edges"].ravel()[0]
        )
        dz = self.res["bin_content"].ravel()

        # Normalize dz for colormap and create color mapping
        norm = mpl.colors.Normalize(vmin=dz.min(), vmax=dz.max())
        colors = plt.cm.viridis(norm(dz))

        # Plot bars with color mapping
        bars = ax.bar3d(
            x_pos.ravel(),
            y_pos.ravel(),
            z_pos.ravel(),
            dx.ravel(),
            dy.ravel(),
            dz.ravel(),
            color=colors,
            **kwargs
        )

        # Set labels
        if xlabel is not None:
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_zlabel(zlabel)

        # Add a color bar based on the colormap and normalization
        mappable = plt.cm.ScalarMappable(norm=norm, cmap='viridis')
        mappable.set_array(dz)
        fig.colorbar(mappable, ax=ax, shrink=0.5, aspect=10, label='Frequency')

    def save(self, file_name:str):
        """
        Save the 3D histogram plot to a file.

        Parameters
        ----------
        file_name : str
            The name of the file to save the plot to. Should end with .npz

        Returns
        -------
        None
        """
        if not file_name.endswith('.npz'):
           raise ValueError("File name must end with .npz")

        np.savez(file_name, x_edges=self.res['x_edges'], y_edges=self.res['y_edges'], bin_content=self.res['bin_content'])

# %%
