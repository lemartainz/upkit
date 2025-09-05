# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 19:06:47 2024

@author: mleon
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import scipy as sp
import tools


class Fit:
    """
        Flexible fitting class for histograms or data arrays.

        Parameters
        ----------
        function : callable
            The function to fit.
        parameters : list or np.ndarray
            Initial guess for fit parameters.
        bounds : 2-tuple of arrays
            Lower and upper bounds for parameters.
        xdata, ydata : array-like, optional
            Raw data to fit.
        bin_content, bin_center : array-like, optional
            Histogram values and centers.
        histo : Histogram object, optional
            Pass a Histogram instance from my_package.
        y_err : array-like, optional
            Uncertainties for ydata.
        print_results : bool, default=True
            Whether to print fitted parameter values.
        kwargs : dict
            Additional fit options.
    """
    def __init__(self, function, parameters, bounds, xdata=None, ydata=None, y_err=None, bin_content=None, bin_center=None, histo=None, print_results=True, **kwargs):
        if function is None:
            raise ValueError("No function to fit!")

        self.print_results = print_results
        self.func = function
        self.parameters = parameters
        self.bounds = bounds
        self.y_err = y_err

        # Determine whether to use histogram data or regular data points
        if bin_content is not None and bin_center is not None:
            self.x = bin_center
            self.y = bin_content
        elif xdata is not None and ydata is not None:
            self.x = xdata
            self.y = ydata
        elif ydata is not None:
            self.x = np.arange(ydata.shape[0])
            self.y = ydata
        elif histo is not None:
            bin_edges = histo.res['bin_edges']
            bin_content = histo.res['bin_content']
            bin_errors = histo.res['bin_errors']
            self.x = bin_edges[:-1] + np.diff(bin_edges)/2
            self.y = bin_content
            self.y_err = bin_errors
        else:
            raise ValueError('No values to fit')

        # Run fit if data is available
        self.fit(**kwargs)

    def fit(self, ax=None, signal=None, background=None, **kwargs):
        """
        Fit a single function to the data.

        Parameters:
        ----------
        ax : matplotlib axis, optional
            Axis on which to plot the fit results.
        
        signal : Function to fit the signal region
        
        background : Function to fit the background function
        """
        if ax is None:
            ax = plt.gca()

        # Use the provided function as the fit function
        fit_func = self.func
        data_x, data_y = self.x, self.y
        
        self.signal = signal
        data_range = kwargs.get('range', None)
        
        if isinstance(data_range[0], tuple):
            x_min, x_max = data_range[0][0], data_range[0][1]
        
        else:
            x_min, x_max = data_range[0], data_range[1]
        # Residuals function for least_squares
        def residuals(params):
            mask = (data_x >= x_min) & (data_x <= x_max)
            if self.y_err is not None:
                # y_err = np.maximum(self.y_err, 10)  # Avoid division by zero
                return (data_y[mask] - fit_func(data_x[mask], *params)) / self.y_err[mask]
            else:
                return (data_y[mask] - fit_func(data_x[mask], *params))

        # Perform the fit
        result = sp.optimize.least_squares(residuals, self.parameters, bounds = self.bounds)

        # Extract fitted parameters and calculate the Jacobian
        fit_params = result.x
        jacobian = result.jac
        # Calculate residuals and their variance
        res = residuals(fit_params)
        residual_variance = np.var(res)
        # Calculate the covariance matrix
        cov = residual_variance * np.linalg.inv(jacobian.T.dot(jacobian))

        param_errors = np.sqrt(np.diag(cov))

        self.fit_params = fit_params
        self.jacobian = jacobian  # Store the Jacobian if needed
        self.cov = cov
        
        fit_info = (
            f"A = {fit_params[0]:.2f} $\pm$ {param_errors[0]:.2f}\n"
            f"$\mu$ = {fit_params[1]:.4f} $\pm$ {param_errors[1]:.4f}\n"
            f"$\sigma$ = {fit_params[2]:.4f} $\pm$ {param_errors[2]:.4f}\n"
        )

        if signal is not None:
            yield_value, yield_uncertainty, yield_var = self.integrate(fit_params[1] - 3 * fit_params[2],
                                                                       fit_params[1] + 3 * fit_params[2], signal, **kwargs)
            self.yield_value = yield_value
            self.yield_uncertainty = yield_uncertainty 

            fit_info += (f"Yield = {yield_value:.0f} $\pm$ {yield_uncertainty:.0f}")

        # Plot the fit and data points
        x_fit = np.linspace(x_min, x_max, 10000)
        self.yfit = fit_func(data_x, *fit_params)
        ax.plot(x_fit, fit_func(x_fit, *fit_params), label='Fit', color='red')
        if signal is not None and background is not None:
            ax.plot(x_fit, signal(x_fit, *fit_params[:3]), label='Signal', color='magenta')
            ax.plot(x_fit, background(x_fit, *fit_params[3:]),
                    label='Background', color='orange', linestyle='--')

        handles, labels = ax.get_legend_handles_labels()
        handles.append(mlines.Line2D([], [], color='None', label=fit_info))
        labels.append(fit_info)
        self.labels = labels
        self.handles = handles

        ax.legend(handles=handles, labels=labels, loc='best')
        self.x_fit = x_fit
        # Print results
        if self.print_results:
            print("Fit Parameters:", fit_params)

        return fit_params

    def integrate(self, a, b, signal, bins=None, range=None, **kwargs):
        """


        Parameters
        ----------
        a : TYPE
            DESCRIPTION.
        b : TYPE
            DESCRIPTION.
        bins : TYPE, optional
            DESCRIPTION. The default is None.
        range : TYPE, optional
            DESCRIPTION. The default is None.
        **kwargs : TYPE
            DESCRIPTION.

        Yields
        ------
        yield_value : TYPE
            DESCRIPTION.
        yield_uncertainty : TYPE
            DESCRIPTION.

        """


        ### Implement auto differentiation for any signal function
        if bins is not None and range is not None:
            bin_width = (range[1] - range[0]) / bins
            yield_value, _ = sp.integrate.quad(
                signal, a, b, args=tuple(self.fit_params[:3]))
            yield_value /= bin_width
            self.bin_width = bin_width

        else:
            yield_value, _ = sp.integrate.quad(signal, a, b, args=tuple(self.fit_params[:3]))

        # Partial derivatives of the yield with respect to A and sigma
        A, mean, sigma = self.fit_params[:3]
        
        if self.signal is tools.gauss_fit:
            # Covariance matrix from curve_fit
            sigma_A2 = self.cov[0, 0]       # Variance of A
            sigma_sigma2 = self.cov[2, 2]   # Variance of sigma
            cov_A_sigma = self.cov[0, 2]    # Covariance of A and sigma
    
            # Partial derivatives
            dY_dA = np.sqrt(2 * np.pi) * sigma
            dY_dsigma = A * np.sqrt(2 * np.pi)
    
            # Uncertainty in yield
            if bins is not None and range is not None:
                yield_uncertainty_cov = np.sqrt((dY_dA ** 2) * sigma_A2 + (dY_dsigma ** 2)
                                                * sigma_sigma2 + 2 * dY_dA * dY_dsigma * cov_A_sigma) / bin_width
                yield_uncertainty_var = np.sqrt((dY_dA ** 2) * sigma_A2 + (dY_dsigma ** 2)
                                                * sigma_sigma2) / bin_width
    
            else:
                yield_uncertainty_cov = np.sqrt((dY_dA ** 2) * sigma_A2 + (dY_dsigma ** 2)
                                                * sigma_sigma2 + 2 * dY_dA * dY_dsigma * cov_A_sigma)
    
            return yield_value, yield_uncertainty_cov, yield_uncertainty_var
        
        elif self.signal is tools.lorentz_fit:
            # Covariance matrix from curve_fit
            sigma_A2 = self.cov[0, 0]       # Variance of A
            sigma_gamma2 = self.cov[2, 2]   # Variance of gamma
            sigma_x02 = self.cov[1, 1]      # Variance of x0
            cov_A_gamma = self.cov[0, 2]    # Covariance of A and gamma
            cov_A_x0 = self.cov[0, 1]       # Covariance of A and x0
            cov_gamma_x0 = self.cov[1, 2]   # Covariance of gamma and x0
        
            # Parameters
            A = self.fit_params[0]
            x0 = self.fit_params[1]
            gamma = self.fit_params[2]
            x1, x2 = a, b
        
            # Partial derivatives
            dY_dA = (1 / np.pi) * (np.arctan((x2 - x0) / gamma) - np.arctan((x1 - x0) / gamma))
            dY_dx0 = (A / (np.pi * gamma)) * (
                1 / (1 + ((x2 - x0) / gamma) ** 2) - 1 / (1 + ((x1 - x0) / gamma) ** 2)
            )
            dY_dgamma = (A / np.pi) * (
                ((x2 - x0) / gamma**2) / (1 + ((x2 - x0) / gamma) ** 2)
                - ((x1 - x0) / gamma**2) / (1 + ((x1 - x0) / gamma) ** 2)
            )
            if bins is not None and range is not None:
                yield_value = (A / np.pi) * (np.arctan((x2 - x0) / gamma) - np.arctan((x1 - x0) / gamma))/bin_width
                # Error propagation
                sigma_Y2_cov = (
                    dY_dA**2 * sigma_A2
                    + dY_dx0**2 * sigma_x02
                    + dY_dgamma**2 * sigma_gamma2
                    + 2 * dY_dA * dY_dx0 * cov_A_x0
                    + 2 * dY_dA * dY_dgamma * cov_A_gamma
                    + 2 * dY_dx0 * dY_dgamma * cov_gamma_x0
                ) 
                yield_uncertainty_cov = np.sqrt(sigma_Y2_cov)/bin_width
                
                sigma_Y2 = (
                    dY_dA**2 * sigma_A2
                    + dY_dx0**2 * sigma_x02
                    + dY_dgamma**2 * sigma_gamma2
                )
                yield_uncertainty = np.sqrt(sigma_Y2)/bin_width
                
                return yield_value, yield_uncertainty_cov, yield_uncertainty
            else:
                yield_value = (A / np.pi) * (np.arctan((x2 - x0) / gamma) - np.arctan((x1 - x0) / gamma))
                sigma_Y2_cov = (
                    dY_dA**2 * sigma_A2
                    + dY_dx0**2 * sigma_x02
                    + dY_dgamma**2 * sigma_gamma2
                    + 2 * dY_dA * dY_dx0 * cov_A_x0
                    + 2 * dY_dA * dY_dgamma * cov_A_gamma
                    + 2 * dY_dx0 * dY_dgamma * cov_gamma_x0
                ) 
                yield_uncertainty_cov = np.sqrt(sigma_Y2_cov)
                sigma_Y2 = (
                    dY_dA**2 * sigma_A2
                    + dY_dx0**2 * sigma_x02
                    + dY_dgamma**2 * sigma_gamma2
                )
                yield_uncertainty = np.sqrt(sigma_Y2)
        
                return yield_value, yield_uncertainty_cov, yield_uncertainty
            
        else:
            print("Signal function not recognized. Cannot compute yield.")

        

    def fit_quality(self):
        """


        Returns
        -------
        chi2 : TYPE
            DESCRIPTION.
        reduced_chi2 : TYPE
            DESCRIPTION.
        aic : TYPE
            DESCRIPTION.

        """

        # Calculate the mean value of the observed data

        # Calculate chi-squared statistic
        chi2 = np.sum((self.y - self.yfit)**2 / self.y_err **
                      2) if self.y_err is not None else np.sum((self.y - self.yfit)**2)

        # Calculate degrees of freedom
        dof = len(self.y) - len(self.fit_params)  # Corrected for number of fit parameters
        reduced_chi2 = chi2 / dof

        # Calculate residuals
        residuals = self.y - self.func(self.x, *self.fit_params)
        n = len(self.y)

        if self.y_err is not None:
            log_likelihood = -0.5 * np.sum(((self.y - self.yfit) / self.y_err)**2 + np.log(2 * np.pi * self.y_err**2))
        else:
            residual_variance = np.var(residuals)
            log_likelihood = -0.5 * n * (np.log(2 * np.pi * residual_variance) + np.sum(residuals**2) / residual_variance / n)

        # Number of parameters in the model
        k = len(self.fit_params)

        # Calculate AIC
        aic = 2 * k - 2 * log_likelihood

        return chi2, reduced_chi2, aic
