def plot_binned_mixed_events(self, particles, pdf_path):
        """


        Parameters
        ----------
        params : TYPE
            DESCRIPTION.
        bounds : TYPE
            DESCRIPTION.
        pdf_path : TYPE
            DESCRIPTION.

        Returns
        -------
        x_mean_arr : TYPE
            DESCRIPTION.
        x_sigma_arr : TYPE
            DESCRIPTION.
        x_array : TYPE
            DESCRIPTION.

        """
        xe = self.res['x_edges']
        bin_size = xe[1] - xe[0]

        with PdfPages(pdf_path) as pdf:

            fig, axs = plt.subplots(3, 2, figsize = (12, 9))  # 3x2 grid per page
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            axs = axs.flatten()  # Flatten to make indexing easier
            plot_count = 0  # Track the number of subplots on the current page

            signal_yield = []
            back_yield = []
            for i in range(xe.size - 1):
                print(f'Bins: [{xe[i]}:{xe[i+1]}]')
                mask = np.where((self.xdata > xe[i]) & (self.xdata < xe[i + 1]))[0]
                y_vals = np.array(self.ydata[mask])
                indices = np.where((self.ydata[mask] <= 0.85) | (self.ydata[mask] >= 1.02))[0]
                # indices = np.where((self.ydata[mask] >= 1.0))[0]
                # indices = np.where((self.ydata[mask] >= 0.85))[0]

                if y_vals.size > 0:  # Skip empty slices
                    ax = axs[plot_count]
                    if isinstance(self.bins, list):
                        bins = self.bins[1]

                    else:
                        bins = self.bins

                    p_p1_bin = particles['p_p1'][mask][indices]
                    p_p2_bin = particles['p_p2'][mask][indices]

                    px_total = []
                    py_total = []
                    pz_total = []
                    E_total = []


                    for _ in range(500):  # Repeat mixing 10 times
                        p_j = np.random.permutation(len(indices))

                        q2_orig = self.xdata[mask][indices]
                        q2_mixed = self.xdata[mask][indices][p_j]

                        mag_orig = p_p2_bin.mag
                        mag_mixed = p_p2_bin[p_j].mag

                        # Apply 5% Q² and proton mag constraint
                        valid_mask = (np.abs(q2_mixed - q2_orig) / q2_orig <= .01) & \
                            (np.abs(mag_mixed - mag_orig) / mag_orig <= .1)

                        # else:
                        # continue

                        if np.any(valid_mask):
                            p_mixed = (
                                particles['p_beam'] + particles['p_target'] -
                                (particles['p_e'][mask][indices][valid_mask] +
                                p_p1_bin[valid_mask] + p_p2_bin[p_j][valid_mask])
                            )

                            px_total.append(p_mixed.px)
                            py_total.append(p_mixed.py)
                            pz_total.append(p_mixed.pz)
                            E_total.append(p_mixed.E)

                    if len(px_total) > 0:  # Check that at least one mix worked
                        p_mixed = vec.array({
                            "px": np.concatenate(px_total),
                            "py": np.concatenate(py_total),
                            "pz": np.concatenate(pz_total),
                            "E": np.concatenate(E_total)
                        })


                    # Histogram of actual events in the bin
                    h_actual = histo(y_vals, bins=bins, range=self.range[1], color='white', ax=ax)
                    plt.sca(ax)
                    h_actual.plot_exp(color='black', fmt='.', markersize = 2, elinewidth = 1, ax=ax)
                    
                    # Histogram of mixed events in the bin
                    h_mixed = histo(p_mixed.M, bins=bins, range=self.range[1], color='green')
                    
                    bin_edges = h_actual.res['bin_edges']
                    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # Compute bin centers
                    
                    try:
                        scale_factor = np.sum(h_actual.res['bin_content'][((bin_centers <= 1.5) & (bin_centers >= 1.1)) | ((bin_centers >= .45) & ((bin_centers <= .85)))]) / \
                            np.sum(h_mixed.res['bin_content'][((bin_centers <= 1.5) & (bin_centers >= 1.1)) | ((bin_centers >= .45) & ((bin_centers <= .85)))])

                        # Plot scaled mixed events on the same axis
                        h_mixed = histo(
                            bin_edges=h_mixed.res['bin_edges'],
                            bin_content=h_mixed.res['bin_content'] * scale_factor,
                            bins=bins,
                            range=self.range[1],
                            histtype='step',
                            color='magenta',
                            alpha=0.5,
                            ax=ax,
                        )
                        
                        sig_yield = np.sum(h_actual.res['bin_content'][(bin_centers <= 1.1) & ((bin_centers >= .85))])
                        b_yield = np.sum(h_mixed.res['bin_content'][(bin_centers <= 1.1) & ((bin_centers >= .85))])

                        # sig_yield -= b_yield
                        signal_yield.append(sig_yield)
                        back_yield.append(b_yield)
                        # Customize plot
                        ax.tick_params(axis='x', labelsize=8)
                        ax.tick_params(axis='y', labelsize=8)
                        ax.legend(fontsize=5)
                        ax.set_title(f"[{xe[i]:.2f}-{xe[i + 1]:.2f}]", fontsize=10)
                        plot_count += 1

                        # If 20 subplots are filled, save the page and reset
                        if plot_count == 6:
                            pdf.savefig(fig)  # Save the current figure (page) to PDF
                            plt.close(fig)  # Close the figure to free memory
                            fig, axs = plt.subplots(3, 2, figsize = (12, 9))  # Start a new page
                            fig.set_layout_engine('tight')
                            fig.subplots_adjust(hspace=0.3, wspace=0.3)
                            axs = axs.flatten()
                            plot_count = 0
                    except (RuntimeError, np.linalg.LinAlgError):
                        print('Singular Matrix')

                    close_all()  # Make sure to close figures related to histograms

            # Save the remaining plots on the last page if there are any
            if plot_count > 0:
                pdf.savefig(fig)
                plt.close(fig)
        return signal_yield, back_yield

    def calculate_phi_mix(self, particles, c, d, pdf_path, h5_path, angle1=40, angle2=150):
        """


        Parameters
        ----------
        params : TYPE
            DESCRIPTION.
        bounds : TYPE
            DESCRIPTION.
        pdf_path : TYPE
            DESCRIPTION.

        Returns
        -------
        x_mean_arr : TYPE
            DESCRIPTION.
        x_sigma_arr : TYPE
            DESCRIPTION.
        x_array : TYPE
            DESCRIPTION.

        """
        from numba import njit, prange
        @njit(parallel=True)
        def mix_phi_randomized(pt1, pz1, phi1, pt2, pz2, phi2,
                        px_base, py_base, pz_base, E_base, mass1, mass2, angle1, angle2):
            n_particles = pt1.shape[0]
            n_mix = 1
            MM = np.empty((n_mix, n_particles))
            px1_all = np.empty((n_mix, n_particles))
            py1_all = np.empty((n_mix, n_particles))
            pz1_all = np.empty((n_mix, n_particles))
            E1_all  = np.empty((n_mix, n_particles))

            px2_all = np.empty((n_mix, n_particles))
            py2_all = np.empty((n_mix, n_particles))
            pz2_all = np.empty((n_mix, n_particles))
            E2_all  = np.empty((n_mix, n_particles))

            px = np.empty((n_mix, n_particles))
            py = np.empty((n_mix, n_particles))
            pz = np.empty((n_mix, n_particles))
            E  = np.empty((n_mix, n_particles))

            angle_rad1 = angle1 * (np.pi / 180.0)
            angle_rad2 = angle2 * (np.pi / 180.0)

            for i in prange(n_mix):
                for j in range(n_particles):
                    while True:
                        dphi1 = np.random.uniform(-np.pi, np.pi)
                        dphi2 = np.random.uniform(-np.pi, np.pi)

                        phi1_new = phi1[j] + dphi1
                        phi2_new = phi2[j] + dphi2

                        dphi_new = np.abs(np.arctan2(np.sin(phi1_new - phi2_new), np.cos(phi1_new - phi2_new)))
                        # Check if the new angle is within the specified range
                        if (dphi_new > angle_rad1) & (dphi_new < angle_rad2):
                            break

                    px1 = pt1[j] * np.cos(phi1_new)
                    py1 = pt1[j] * np.sin(phi1_new)
                    px2 = pt2[j] * np.cos(phi2_new)
                    py2 = pt2[j] * np.sin(phi2_new)
                    
                    p1_sq = px1**2 + py1**2 + pz1[j]**2
                    p2_sq = px2**2 + py2**2 + pz2[j]**2

                    E1 = np.sqrt(p1_sq + mass1**2)
                    E2 = np.sqrt(p2_sq + mass2**2)

                    px_tot = px_base[j] - px1 - px2
                    py_tot = py_base[j] - py1 - py2
                    pz_tot = pz_base[j] - pz1[j] - pz2[j]
                    E_tot  = E_base[j] - E1 - E2

                    M2 = E_tot**2 - (px_tot**2 + py_tot**2 + pz_tot**2)
                    MM[i, j] = np.sqrt(max(M2, 0))

                    E_new = np.sqrt(px_tot**2 + py_tot**2 + pz_tot**2 + mass1**2)

                    # Save 4-momenta
                    px1_all[i, j] = px1
                    py1_all[i, j] = py1
                    pz1_all[i, j] = pz1[j]
                    E1_all[i, j]  = E1

                    px2_all[i, j] = px2
                    py2_all[i, j] = py2
                    pz2_all[i, j] = pz2[j]
                    E2_all[i, j]  = E2

                    px[i,j] = px_tot
                    py[i,j] = py_tot
                    pz[i,j] = pz_tot
                    E[i,j]  = E_new

            return MM, (px1_all, py1_all, pz1_all, E1_all), (px2_all, py2_all, pz2_all, E2_all), (px, py, pz, E)


        # Outside the compiled function
        def phi_mix_fast(particle1, particle2, cuts, angle1, angle2):
            pt1 = particle1.pt
            pt2 = particle2.pt
            pz1 = particle1.pz
            pz2 = particle2.pz
            phi1 = particle1.phi
            phi2 = particle2.phi

            # Fixed masses
            m1 = particle1.M[0]
            m2 = particle2.M[0]
            e1 = np.sqrt(pt1**2 + pz1**2 + m1**2)
            e2 = np.sqrt(pt2**2 + pz2**2 + m2**2)

            # Get beam + target - p_e[cuts] as NumPy arrays
            p_total_base = (particles['p_beam'] + particles['p_target'] - particles['p_e'][cuts])
            px_base = p_total_base.px
            py_base = p_total_base.py
            pz_base = p_total_base.pz
            E_base  = p_total_base.E

            return mix_phi_randomized(pt1, pz1, phi1, pt2, pz2, phi2, 
                                    px_base, py_base, pz_base, E_base, m1, m2, angle1, angle2)
        import h5py
        h5f = h5py.File(h5_path, 'w')

        with PdfPages(pdf_path) as pdf:

            fig, axs = plt.subplots(3, 2)  # 4x5 grid per page
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            axs = axs.flatten()  # Flatten to make indexing easier
            plot_count = 0  # Track the number of subplots on the current page
            # ax = axs[plot_count]
            if isinstance(self.bins, list):
                bins = self.bins[1]

            else:
                bins = self.bins

            try:
                signal_mask = (self.ydata >= 0.85) & (self.ydata <= 1.05)  # or your own criteria
                Q2_signal = self.xdata[(self.xdata != -999) & signal_mask]

                # Step 2: define bins based on quantiles of signal events
                n_bins = 10
                Q2_cut = np.quantile(Q2_signal, np.linspace(0, 1, n_bins + 1))
                Q2 = self.xdata
                h_MM_cuts_all = []
                h_phi_mix_all = []
                
                # Loop over each shift_right value
                for i in range(len(Q2_cut) - 1):
                    Q2_min, Q2_max = Q2_cut[i], Q2_cut[i+1]
                    ax = axs[plot_count]
                    print(r"Q2 range: [{:.3f}, {:.3f}]".format(Q2_cut[i], Q2_cut[i + 1]))
                    
                    all_phi_distributions = []
                    fig1, a = plt.subplots()

                    # Plot the real MM distribution first
                    h_MM_cuts = histo(self.ydata[((Q2 > Q2_cut[i]) & (Q2 < Q2_cut[i + 1]))],
                                            bins=bins, range=(0.65, 1.25), color='white', ax=a)
                    # h_MM_cuts.plot_exp(color='black', fmt='.', ax=a)
                    # h_MM_cuts.show_hists(xlabel=r'MM $[GeV/c^{2}]$', ylabel='Counts / 10 MeV', ax=a)

                    params = [1000, 0.938, .1, 1, 1, 1, 1, 1]
                    bounds = [(0, 0.8, 0, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf), (10000, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf)]
                    fit_MM_test = Fit(tools.lorentz_poly4_fit, params, bounds, histo=h_MM_cuts,
                            signal=tools.lorentz_fit, background=tools.poly4_fit, bins=bins, range=(0.65, 1.25), print_results=False)
                    
                    sigma = fit_MM_test.fit_params[2]
                    mu = fit_MM_test.fit_params[1]  
                    shift_right = np.linspace(mu + c * sigma, mu + d *sigma, 11)  # Example with 10 values for shift_right
                    shift_left = np.linspace(mu - d *sigma, mu - c*sigma, 11)
                    all_shift_pairs = [(left, right) for left, right in product(shift_left, shift_right) if left <= right]
                    plt.close()
                    q2_grp_name  = f"Q2_bin_{i}"
                    if q2_grp_name in h5f:
                        del h5f[q2_grp_name]          # overwrite if rerun
                    grp_q2 = h5f.create_group(q2_grp_name)
                    grp_q2.attrs["Q2_min"] = Q2_min
                    grp_q2.attrs["Q2_max"] = Q2_max

                    for shift_right_value in shift_right:
                        fig1, ax1 = plt.subplots()
                        h_MM_cuts = histo(self.ydata[((Q2 > Q2_cut[i]) & (Q2 < Q2_cut[i + 1]))],
                                            bins=bins, range=(0.65, 1.25), color='white', ax=ax1)
                        h_MM_cuts.plot_exp(color='black', fmt='.', ax=ax)
                        # h_MM_cuts.show_hists(xlabel=r'MM $[GeV/c^{2}]$', ylabel='Counts / 10 MeV', ax=ax1)
                        phi_distributions = []

                        for left_edge, right_edge in filter(lambda pair: pair[1] == shift_right_value, all_shift_pairs):
                            # print(f"Excluding MM in range [{left_edge:.3f}, {right_edge:.3f}]")

                            range_cut = (self.ydata >= right_edge) | (self.ydata <= left_edge)
                            full_cut = range_cut & (Q2 > Q2_cut[i]) & (Q2 < Q2_cut[i+1])
                            # phi_mix function call
                            MM_bg, p_1, p_2, p_pb = phi_mix_fast(
                                particle1=particles['p_p1'][full_cut],
                                particle2=particles['p_p2'][full_cut],
                                cuts=full_cut, 
                                angle1=angle1, angle2=angle2
                            )

                            bg_signal_mask = (MM_bg >= mu - 3*sigma) & (MM_bg <= mu + 3*sigma)
                            p1x_new = p_1[0][bg_signal_mask]
                            p1y_new = p_1[1][bg_signal_mask]
                            p1z_new = p_1[2][bg_signal_mask]
                            p1E_new = p_1[3][bg_signal_mask]
                            p2x_new = p_2[0][bg_signal_mask]
                            p2y_new = p_2[1][bg_signal_mask]
                            p2z_new = p_2[2][bg_signal_mask]
                            p2E_new = p_2[3][bg_signal_mask]
                            pbx_new = p_pb[0][bg_signal_mask]
                            pby_new = p_pb[1][bg_signal_mask]
                            pbz_new = p_pb[2][bg_signal_mask]
                            pbE_new = p_pb[3][bg_signal_mask]
                            
                            p1_new = np.stack((p1x_new, p1y_new, p1z_new, p1E_new), axis=-1).astype("float32")
                            p2_new = np.stack((p2x_new, p2y_new, p2z_new, p2E_new), axis=-1).astype("float32")   
                            p_pb_new = np.stack((pbx_new, pby_new, pbz_new, pbE_new), axis=-1).astype("float32")   

                            # ---------- save to HDF5 ----------
                            excl_name = f"excl_{left_edge:.3f}_{right_edge:.3f}"
                            sub = grp_q2.create_group(excl_name)
                            sub.create_dataset("mix_p1", data=p1_new, compression="gzip", dtype='float32')
                            sub.create_dataset("mix_p2", data=p2_new, compression="gzip", dtype='float32')
                            sub.create_dataset("mix_pb", data=p_pb_new, compression="gzip", dtype='float32')

                            # add any attributes if useful:
                            # sub.attrs["left_MM_edge"]  = left_edge
                            # sub.attrs["right_MM_edge"] = right_edge

                            h_phi_mix = histo(MM_bg.flatten(), bins=bins, range=(0.65, 1.25), alpha=0.3)
                            plt.close()
                            bin_centers = (h_phi_mix.res['bin_edges'][:-1] + h_phi_mix.res['bin_edges'][1:])/2
                            signal_mask = ((bin_centers <= left_edge) & (bin_centers >= .5)) | \
                                        ((bin_centers >= right_edge) & (bin_centers <= 1.5))

                            scale = np.sum(h_MM_cuts.res['bin_content'][signal_mask]) / \
                                    np.sum(h_phi_mix.res['bin_content'][signal_mask])

                            h_phi_mix.res['bin_content'] *= scale
                            # print(h_phi_mix.res['bin_content'])
                            phi_distributions.append(h_phi_mix.res['bin_content'])

                            all_phi_distributions.append(h_phi_mix.res['bin_content'])

                            # Plot the histogram
                            h_phi_mix = histo(bin_edges = h_phi_mix.res['bin_edges'],
                                                    bin_content=h_phi_mix.res['bin_content'],
                                                    bins=bins,
                                                    range=(0.65, 1.25),
                                                    histtype='step',
                                                    label = 'Range: [{:.3f}, {:.3f}]'.format(left_edge, right_edge),
                                                    ax = ax1
                                                )
                            
                            # print(h_phi_mix.res['bin_content'])
                            # Plot each phi distribution for the given shift_right value
                            
                        # Calculate the averaged histogram
                        avg_phi_distribution = np.mean(np.array(phi_distributions), axis=0)

                        # Plot the averaged histogram
                        h_phi_mix_avg = histo(
                            bin_edges=h_phi_mix.res['bin_edges'],
                            bin_content=avg_phi_distribution,
                            bins=bins,
                            range=(0.65, 1.25),
                            color='red',
                            histtype='step',
                            ax=ax1
                        )
                        plt.close()

                        # Show the figure for each shift_right
                        # fig.tight_layout()
                        # plt.legend(fontsize = 12)
                        # plt.show()

                    # Plot the real MM distribution first
                    raw_data = self.ydata[((Q2 >= Q2_cut[i]) & (Q2 <= Q2_cut[i+1]))]
                    h_MM_cuts = histo(raw_data, bins=bins, range=(0.65, 1.25), color='white', ax=ax)
                    plt.sca(ax)
                    h_MM_cuts.plot_exp(color='black', fmt='.', ax=ax)
                    h_MM_cuts.show_hists(xlabel=r'MM $[GeV/c^{2}]$', ylabel='Counts / 6 MeV', ax=ax)

                    avg_all_phi_distribution = np.mean(np.array(all_phi_distributions), axis=0)
                    sigma_all_phi_distribution = np.std(np.array(all_phi_distributions), axis=0)
                    # print(sigma_all_phi_distribution)
                    h_MM_cuts_all.append(raw_data)
                    # Plot the averaged histogram
                    h_all_phi_mix_avg = histo(
                        bin_edges=h_MM_cuts.res['bin_edges'],
                        bin_content=avg_all_phi_distribution,
                        bin_error=sigma_all_phi_distribution,
                        bins=bins,
                        range=(0.65, 1.25),
                        color='red',
                        histtype='step',
                        ax=ax,
                        label=r'$\phi$ Randomization'
                    )
                    h_phi_mix_all.append(h_all_phi_mix_avg)
                    bin_centers = (h_all_phi_mix_avg.res['bin_edges'][:-1] +
                       h_all_phi_mix_avg.res['bin_edges'][1:]) / 2

                    # h_all_phi_mix_avg.plot_exp(color='red', ls = '-', linewidth = 2, elinewidth = 1, alpha = .5, ax=ax)
                    plt.fill_between(
                        bin_centers,
                        avg_all_phi_distribution - sigma_all_phi_distribution,
                        avg_all_phi_distribution + sigma_all_phi_distribution,
                        color='red',
                        alpha=0.5
                    )
                    # Show the figure for each shift_right
                    fig.tight_layout()
                    plt.show()
                    # Customize plot
                    ax.tick_params(axis='x', labelsize=8)
                    ax.tick_params(axis='y', labelsize=8)
                    ax.legend(fontsize=5)
                    ax.xaxis.label.set_fontsize(10)
                    ax.yaxis.label.set_fontsize(10)
                    ax.set_title(r"$Q^2$ range: [{:.3f}, {:.3f}]".format(Q2_cut[i], Q2_cut[i + 1]), fontsize=10)
                    plot_count += 1

                    # If 20 subplots are filled, save the page and reset
                    if plot_count == 6:
                        pdf.savefig(fig)  # Save the current figure (page) to PDF
                        plt.close(fig)  # Close the figure to free memory
                        fig, axs = plt.subplots(3, 2)  # Start a new page
                        fig.subplots_adjust(hspace=0.3, wspace=0.3)
                        axs = axs.flatten()
                        plot_count = 0
            except (RuntimeError, np.linalg.LinAlgError):
                print('Singular Matrix')

            close_all()  # Make sure to close figures related to histograms

            # Save the remaining plots on the last page if there are any
            if plot_count > 0:
                pdf.savefig(fig)
                plt.close(fig)

        h5f.close()  # Ensure the HDF5 file is closed properly
        # Return a dictionary with Q2 ranges and corresponding histograms

        return {(q2min, q2max): { "MM": MM, "phi": phi} for i, ((q2min, q2max), MM, phi) in enumerate(zip(zip(Q2_cut[:-1], Q2_cut[1:]), h_MM_cuts_all, h_phi_mix_all))}
def plot_binned(self, params, bounds, pdf_path):
        """


        Parameters
        ----------
        params : TYPE
            DESCRIPTION.
        bounds : TYPE
            DESCRIPTION.
        pdf_path : TYPE
            DESCRIPTION.

        Returns
        -------
        x_mean_arr : TYPE
            DESCRIPTION.
        x_sigma_arr : TYPE
            DESCRIPTION.
        x_array : TYPE
            DESCRIPTION.

        """
        yields, yield_uncertainty = [], []

        with PdfPages(pdf_path) as pdf:
            xe = self.res['x_edges']
            bin_size = xe[1] - xe[0]

            x_mean_l, x_sigma_l, x_array_l = [], [], []

            fig, axs = plt.subplots(4, 5)  # 4x5 grid per page
            fig.subplots_adjust(hspace=0.3, wspace=0.3)
            axs = axs.flatten()  # Flatten to make indexing easier
            plot_count = 0  # Track the number of subplots on the current page

            for i in range(xe.size - 1):
                self.yvals = self.ydata[(self.xdata > xe[i]) & (self.xdata <= xe[i + 1])]
                y_vals = np.array(self.yvals)

                if y_vals.size > 0:  # Skip empty slices
                    if isinstance(self.bins, list):
                        bins = self.bins[1]

                    else:
                        bins = self.bins

                    h_t = histo(y_vals, bins=bins, range=self.range[1])
                    try:
                        ax = axs[plot_count]

                        fit_t = Fit(tools.lorentz_poly4_fit, params, bounds, histo=h_t,
                                    signal=tools.lorentz_fit, background=tools.poly4_fit, bins=bins, range=self.range[1], ax=ax)

                        yields.append(fit_t.yield_value)
                        yield_uncertainty.append(fit_t.yield_uncertainty)

                        ax.legend(handles=fit_t.handles, labels=fit_t.labels, fontsize=5)
                        h_t.plot(ax=ax, bins=bins, range=self.range[1])
                        ax.tick_params(axis='x', labelsize=8)
                        ax.tick_params(axis='y', labelsize=8)
                        ax.set_title(f"[{xe[i]:.2f}-{xe[i+1]:.2f}]", fontsize=10)
                        plot_count += 1

                        # If 20 subplots are filled, save the page and reset
                        if plot_count == 20:
                            pdf.savefig(fig)  # Save the current figure (page) to PDF
                            plt.close(fig)  # Close the figure to free memory
                            fig, axs = plt.subplots(4, 5)  # Start a new page
                            fig.subplots_adjust(hspace=0.3, wspace=0.3)
                            axs = axs.flatten()
                            plot_count = 0
                    except (RuntimeError, np.linalg.LinAlgError):
                        print('Singular Matrix')

                    close_all()  # Make sure to close figures related to histograms

            # Save the remaining plots on the last page if there are any
            if plot_count > 0:
                pdf.savefig(fig)
                plt.close(fig)

        self.yields = yields
        self.yield_uncertainty = yield_uncertainty


def profile_function(self, params, bounds, pdf_path): 
    """ Parameters ----------
    params : TYPE DESCRIPTION.
    bounds : TYPE DESCRIPTION.
    pdf_path : TYPE DESCRIPTION.

    Returns -------
    x_mean_arr : TYPE DESCRIPTION.
    x_sigma_arr : TYPE DESCRIPTION.
    x_array : TYPE DESCRIPTION.
    """
    with PdfPages(pdf_path) as pdf:
        xe = self.res['x_edges']
        bin_size = xe[1] - xe[0]
        x_mean_l, x_sigma_l, x_array_l = [], [], []
        fig, axs = plt.subplots(4, 5)  # 4x5 grid per page
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        axs = axs.flatten()  # Flatten to make indexing easier
        plot_count = 0  # Track the number of subplots on the current page

        for i in range(xe.size - 1):
            self.yvals = self.ydata[(self.xdata > xe[i]) & (self.xdata <= xe[i + 1])]
            y_vals = np.array(self.yvals)

            if y_vals.size > 0:  # Skip empty slices
                h_t = histo(self.yvals, bins=100, range=self.range[1])
                try:
                    ax = axs[plot_count]
                    fit_t = Fit(tools.gauss_poly2_fit, params, bounds, histo=h_t, range=self.range[1], ax=ax)
                    x_mean, x_sigma = fit_t.fit_params[1], fit_t.fit_params[2]
                    ax.legend(handles=fit_t.handles, labels=fit_t.labels, fontsize=5)
                    x_array_l.append(xe[i] + bin_size / 2)
                    x_mean_l.append(x_mean)
                    x_sigma_l.append(x_sigma)
                    h_t.plot(ax=ax, bins=100, range=self.range[1])
                    ax.tick_params(axis='x', labelsize=8)
                    ax.tick_params(axis='y', labelsize=8)
                    ax.set_title(f"[{xe[i]:.2f}-{xe[i+1]:.2f}]", fontsize=10)
                    plot_count += 1

                    # If 20 subplots are filled, save the page and reset
                    if plot_count == 20:
                        pdf.savefig(fig)  # Save the current figure (page) to PDF
                        plt.close(fig)  # Close the figure to free memory
                        fig, axs = plt.subplots(4, 5)  # Start a new page
                        fig.subplots_adjust(hspace=0.3, wspace=0.3)
                        axs = axs.flatten()
                        plot_count = 0
                except (RuntimeError, np.linalg.LinAlgError):
                    print('Singular Matrix')

                close_all()  # Make sure to close figures related to histograms

        # Save the remaining plots on the last page if there are any
        if plot_count > 0:
            pdf.savefig(fig)
            plt.close(fig)

        x_mean_arr = np.array(x_mean_l)
        x_sigma_arr = np.array(x_sigma_l)
        x_array = np.array(x_array_l)

        self.plot(bins=self.bins, range=self.range)
        plt.errorbar(x_array, x_mean_arr, yerr=x_sigma_arr, fmt='_', color='black')

        return x_mean_arr, x_sigma_arr, x_array