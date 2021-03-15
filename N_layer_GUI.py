import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import Layout
from scipy import interpolate
# Import radiative transfer utils box
import utils

class n_layer_gui:

    def __init__(self):
        """
        
        Collect scalable physical parameters to be used for the N-Layer radiative model.
        
        """
        ###############################
        self.N_slider = widgets.IntSlider(min=3, max=20, value=3, description='N-Layers: ')
        widgets.interact(self.get_n_slider, N_slider=self.N_slider)
        ###############################
        self.albedo_slider = widgets.FloatSlider(min=0, max=1, value=0.3, description='Albedo: ')
        widgets.interact(self.get_albedo_slider, albedo_slider=self.albedo_slider)
        ###############################
        self.S0_slider = widgets.IntSlider(min=10, max=10000, value=1368, description='$S_{0}$ ($W/m^2$): ')
        widgets.interact(self.get_S0_slider, S0_slider=self.S0_slider)
        ###############################
        style = {'description_width': 'initial'}
        self.SW_slider = widgets.IntSlider(min=0, max=100, value=3, style=style, 
                                           layout=Layout(width='45%', height='20px'),
                                           description='SW absorbed in Stratosphere (%): ')
        widgets.interact(self.get_SW_slider, SW_slider=self.SW_slider)
        ###############################
        self.heatflux_slider = widgets.IntSlider(min=0, max=500, value=0, style=style,
                                                 layout=Layout(width='35%', height='20px'),
                                                 description='Surface Heat Flux ($W/m^2$): ')
        widgets.interact(self.get_heatflux_slider, heatflux_slider=self.heatflux_slider)
        ###############################
        self.pertub_button = widgets.ToggleButton(value=False, description='Perturbation?', disabled=False, 
                                                  button_style='info', tooltip='Description')
        widgets.interact(self.get_perturbation, pertub_button=self.pertub_button)
        ###############################
        self.calc_button = widgets.ToggleButton(value=False, description='Radiate', disabled=False, button_style='danger',
                                                tooltip='Description')
        widgets.interact(self.get_temps, calc_button=self.calc_button)
    
    

    def get_n_slider(self, N_slider):
        self.N = N_slider
    def get_albedo_slider(self, albedo_slider):
        self.albedo = albedo_slider
    def get_S0_slider(self, S0_slider):
        self.S0 = S0_slider
    def get_SW_slider(self, SW_slider):
        self.SW_strat_absorption = SW_slider
    def get_heatflux_slider(self, heatflux_slider):
        self.heatflux = heatflux_slider
    def get_perturbation(self, pertub_button):
        self.pertub_button = pertub_button
        if self.pertub_button is True:
            self.layer_perturb = widgets.IntSlider(min=1, max=self.N, value=1, description='Which layer?: ')
            widgets.interact(self.get_perturb_location, layer_perturb=self.layer_perturb)
            
            self.perturb_magnitude = widgets.FloatText(value=0.0, description='% emissivity:', disabled=False)
            widgets.interact(self.get_perturb_magnitude, perturb_magnitude=self.perturb_magnitude)
            
        elif self.pertub_button is False:
            self.layer_perturb = 0
            self.perturb_magnitude = 0
    def get_perturb_location(self, layer_perturb):
        self.layer_perturb = layer_perturb   
    def get_perturb_magnitude(self, perturb_magnitude):
        self.perturb_magnitude = perturb_magnitude  
        
    
    def get_temps(self, calc_button):
        """
        Query whether track button has been toggled, and calculate the radiation matrix before plotting.
        
        """
        
        ## separate N into troposphere, tropopause, and stratosphere
        ## subtract 1 to allow for tropopause
        N_troposphere = int(0.85*self.N) - 1
        N_tpause = 1
        N_tpause_location = int(0.85*self.N) # layer index for tropopause (surface is zero index)
        N_stratosphere = self.N - N_troposphere - N_tpause
        
        # Load Tropical Profile 
        era5_profile = np.load('tropical_profile.npy')
        
        ## find tropopause index in the ERA5 data (take the first inversion, polar upper atmos is complex)
        era5_N_of_tpause = np.where(np.r_[True, era5_profile[1:] < era5_profile[:-1]] & 
                                np.r_[era5_profile[:-1] < era5_profile[1:], True])[0][0]
        
        ## dummy values for interpolation to our reduced integer layers
        era5_layers = np.linspace(0, self.N, 32)
        era5_layers_to_tpause = np.linspace(0, N_tpause_location, era5_N_of_tpause)
        
        ## create separate interpolations up to tropopause and for stratosphere
        ## --this is necessary to retain the structure, otherwise we can average out the tropopause
        interp_troposphere = interpolate.interp1d(era5_layers_to_tpause, era5_profile[:era5_N_of_tpause])
        profile_for_emissivities_trop = interp_troposphere(np.arange(N_tpause_location+1))

        interp_stratosphere = interpolate.interp1d(era5_layers[era5_N_of_tpause:], era5_profile[era5_N_of_tpause:])
        profile_for_emissivities_strat = interp_stratosphere(np.arange(N_stratosphere) + N_tpause_location + 1)
        
        profile_for_emissivities_all = np.append(profile_for_emissivities_trop, profile_for_emissivities_strat)
        
        
        
        if self.calc_button.value is True:
            print('running...')
            
            #### MONTE CARLO ####
            ### M realizations of possible emmisivity profiles for mr. monte carlo
            M = 10000

            ### emissivity profile for each layer, M realizations   
            ## surface is blackbody (emissivity = 1)
            emissivity_surface_M = np.tile(1,M)

            ## monotonic decreasing emissivities in troposphere
            ## this selects a random first layer emissivity,
            ## then each higher layer is random, but a lesser emissivity than the layer below it
            emissivity_troposphere_M = np.zeros((M,N_troposphere))
            emissivity_troposphere_M[:,0] = np.random.uniform(0.2,1,M)
            for i in range(N_troposphere-1):
                emissivity_troposphere_M[:,i+1] = np.random.uniform(0.001,
                                                                    emissivity_troposphere_M[:,i],M)

            ## tropopause
            emissivity_tpause_M = np.random.uniform(0.0001,emissivity_troposphere_M[:,-1],M)

            ## stratosphere
            emissivity_stratosphere_M = np.zeros((M,N_stratosphere))
            emissivity_stratosphere_M[:,0] = np.random.uniform(0.00001,emissivity_tpause_M,M)
            for i in range(N_stratosphere-1):
                emissivity_stratosphere_M[:,i+1] = np.random.uniform(0,emissivity_stratosphere_M[:,i],M)

            emissivity_Nlayers_M = np.vstack([emissivity_surface_M,
                                              emissivity_troposphere_M.T,
                                              emissivity_tpause_M,
                                              emissivity_stratosphere_M.T])


            ## load profiles that don't need to be repeated in the monte carlo loop
            insolation = (self.S0/4)*(1 - self.albedo)
            upward_heatflux = utils.vertical_heat_flux_profile(self.N, self.heatflux, 'tanh')           
            forcings = utils.forcings_vector(self.N, insolation, upward_heatflux, 
                                                 self.SW_strat_absorption)
            
            ## array to hold Monte Carlo temperature values
            temperature_M = np.zeros([M,self.N+1]) # arrays to hold Monte Carlo values
            
            
            ## THE SLOW PART, solving for the temperatures M times. Maybe can vectorize this?
            for i,val in enumerate(emissivity_Nlayers_M.T):
                R_up_matrix = utils.R_up_matrix(self.N, val)
                total_emissivity_matrix = utils.emissivity_matrix(R_up_matrix, val)

                # Find the temperature vector using the emissivity matrix and the forcings
                temperature_vector = utils.temperature(total_emissivity_matrix, forcings)
                temperature_M[i] = temperature_vector
            
            
            ### Check which emissivity profile gave most realistic temperature profile
            
            ## First, restrict to accurate surface temperature:
            ## This is a dummy thing that places -1000 as surface temp for anything that is
            ## more than a few degrees off the surface temperature.
            temperature_M_temp = temperature_M.copy()
            surface_bound = 3
            temperature_M_temp[:,0] = np.where(
                np.abs(profile_for_emissivities_trop[0] - temperature_M[:,0]) < surface_bound, 
                temperature_M[:,0], -1000)

            ## Then find profile that minimizes the squared error relative to era5 up to tropopause
            error = ((temperature_M_temp[:,:N_tpause_location+1] - profile_for_emissivities_trop)**2).sum(axis=1)
            best_index = np.where(error == error.min())
            
            ### Additional, separate monte carlo for stratosphere ###
            ## start with best fit for tropopause
            emissivity_Nlayers_M = np.repeat(emissivity_Nlayers_M[:,best_index[0]],M,axis=1)

            emissivity_tpause = emissivity_Nlayers_M[:,best_index[0]][N_tpause_location]

            ## stratosphere: resample possible strat values
            emissivity_stratosphere_M = np.zeros((M,N_stratosphere))
            emissivity_stratosphere_M[:,0] = np.random.uniform(0.000001,0.1,M)
            for i in range(N_stratosphere-1):
                emissivity_stratosphere_M[:,i+1] = np.random.uniform(0,0.1,M)

            emissivity_Nlayers_M = np.vstack([emissivity_Nlayers_M[:N_tpause_location+1],
                                              emissivity_stratosphere_M.T])

            # arrays to hold Monte Carlo values
            temperature_M = np.zeros([M,self.N+1]) # arrays to hold Monte Carlo values
            
            ## THE SLOW PART, solving for the temperatures M times. Maybe can vectorize this?
            for i,val in enumerate(emissivity_Nlayers_M.T):
                R_up_matrix = utils.R_up_matrix(self.N, val)
                total_emissivity_matrix = utils.emissivity_matrix(R_up_matrix, val)

                # Find the temperature vector using the emissivity matrix and the forcings
                temperature_vector = utils.temperature(total_emissivity_matrix, forcings)
                temperature_M[i] = temperature_vector
                
                
            ### Second round: check which emissivity profile gave the right strat inversion ###
            
            ## First, repeat restriction to accurate surface temperature (probably does nothing)
            ## This is a dummy thing that places -1000 as surface temp for anything that is
            ## more than a few degrees off the surface temperature.
            temperature_M_temp = temperature_M.copy()
            temperature_M_temp[:,0] = np.where(
                np.abs(profile_for_emissivities_trop[0] - temperature_M[:,0]) < surface_bound, 
                temperature_M[:,0], -1000)

            ## Then find profile that minimizes the squared error of the inversion profile
            error = 0.
            for i in range(N_stratosphere):
                j = self.N - N_stratosphere + i
                error += ((temperature_M_temp[:,j+1] - temperature_M_temp[:,j]) - (
                    (profile_for_emissivities_all[j+1] - profile_for_emissivities_all[j])))**2
            best_index_all = np.where(error == error.min()) 
            
            
            ### Additional solution without convection ###
            zero_heatflux = np.zeros(len(upward_heatflux))
            forcings_no_convection = utils.forcings_vector(self.N, insolation, zero_heatflux, 
                                                 self.SW_strat_absorption)
            temperature_vector_no_convection = utils.temperature(total_emissivity_matrix, forcings_no_convection)
            
            ### Additional solution without SW absorb ###
            zero_SW = 0
            forcings_no_SW = utils.forcings_vector(self.N, insolation, upward_heatflux, zero_SW)
            temperature_vector_no_SW = utils.temperature(total_emissivity_matrix, forcings_no_SW)
            
            ### Additional solution for the perturbation ###
            emmisivities = emissivity_Nlayers_M[:,best_index_all].squeeze()
            emmisivities_perturb = emmisivities.copy()
            emmisivities_perturb[self.layer_perturb] = emmisivities[self.layer_perturb]*(1+self.perturb_magnitude/100)
            
            R_up_matrix = utils.R_up_matrix(self.N, emmisivities_perturb)
            
            total_emissivity_matrix = utils.emissivity_matrix(R_up_matrix, emmisivities_perturb)
            
            temperature_vector_perturb = utils.temperature(total_emissivity_matrix, forcings)
            
            if self.pertub_button == True:
                R_up_matrix_reference = utils.R_up_matrix(self.N, emmisivities)
                emissivity_matrix_reference = utils.emissivity_matrix(R_up_matrix_reference, emmisivities)
                
                TOA_instantaneous_forcing = utils.perturb_forcing(total_emissivity_matrix,
                                                                  emissivity_matrix_reference,
                                                                  temperature_M[best_index_all].squeeze())
                print('Instantaneous Radiative Forcing of Perturbation:', 
                      np.around(TOA_instantaneous_forcing, 3),  'W/m\N{SUPERSCRIPT TWO}')
                
                deltaT = temperature_vector_perturb - (temperature_M[best_index_all]).squeeze()
                dT_pert = str(np.round(deltaT[0],2))
                print('Change in Surface Temperature due to Perturbation = ' + dT_pert)

                

            fig, axs = plt.subplots(1,3, figsize=(11,6))

            ## Perturbation plots
            if self.pertub_button is True:
                axs[0].plot(emmisivities_perturb[1:], range(1, self.N+1), 
                            marker='o',mfc='white',mec='k', color='k',lw=1,alpha=0.9,ls='--',label='Perturbation')
                axs[2].plot(temperature_vector_perturb, range(0, self.N+1), 
                            marker='o',mfc='white',mec='firebrick', lw=1,color='firebrick',alpha=0.9,ls='--',label='Perturbation')
                
            ### Control plot
            axs[0].plot(emissivity_Nlayers_M[:,best_index_all].squeeze()[1:], range(1, self.N+1), 
                        marker='o', color='black',label='Control')

            axs[0].set_ylabel('Layer number (n)')
            axs[0].set_xlabel('Emissivity')
            axs[0].set_ylim(0,self.N+0.2)
            axs[0].set_yticks(np.arange(self.N+1))
            axs[0].legend()
            axs[0].grid(alpha=0.3)

            axs[1].set_xlabel('Net Heat Flux (W/$m^2$)')
            axs[1].plot(upward_heatflux, range(0, self.N+1), 
                        marker='o', color='dodgerblue',label='Convection\nParameterization')
            axs[1].set_ylim(0,self.N+0.2)
            axs[1].set_yticks(np.arange(self.N+1))
            axs[1].axvline(0,c='0.5',lw=1)
            axs[1].legend()
            axs[1].grid(alpha=0.3)

            axs[2].plot(temperature_M[best_index_all].squeeze(), range(0, self.N+1), 
                        color='firebrick', marker='o', lw=3,label='Control')
            axs[2].set_xlabel('Temperature ($\degree$K)')
            axs[2].set_ylim(0,self.N+0.2)
            axs[2].set_yticks(np.arange(self.N+1))
            axs[2].grid(alpha=0.3)
            
            ## add ERA-5 plot, dont have to keep it
            axs[2].plot(profile_for_emissivities_all,
                        np.arange(len(profile_for_emissivities_all)),
                        c='k',alpha=0.7,lw=1,label='ERA-5')
            axs[2].legend()
            
            plt.tight_layout()
            plt.show()
            
            #######################################################
            ### additional plots, can comment out whole section ###
#             fig, axs = plt.subplots(1,3, figsize=(11,6))

#             ## no SW plot 
#             if (self.SW_strat_absorption > 0):               
#                 axs[0].plot(temperature_M[best_index_all].squeeze(), range(0, self.N+1), 
#                             color='firebrick', marker='o', lw=3,label='Control')
#                 axs[0].plot(temperature_vector_no_SW, range(0, self.N+1), 
#                             marker='o',ls='--',lw=2,color='darkorange',alpha=0.8,label='No SW absorbed\nin stratosphere')
#                 axs[0].set_xlabel('Temperature ($\degree$K)')
#                 axs[0].set_ylim(0,self.N+0.2)
#                 axs[0].set_yticks(np.arange(self.N+1))
#                 axs[0].grid(alpha=0.3)
#                 axs[0].set_ylabel('Layer number (n)')
#                 axs[0].legend()
            
#             ## no convection plots
#             if (self.heatflux > 0):
#                 axs[1].plot(temperature_M[best_index_all].squeeze(), range(0, self.N+1), 
#                             color='firebrick', marker='o', lw=3,label='Control')
#                 axs[1].plot(temperature_vector_no_convection, range(0, self.N+1), 
#                             marker='o',lw=2,color='dodgerblue',alpha=0.8,ls='--',label='No convection')
#                 axs[1].set_xlabel('Temperature ($\degree$K)')
#                 axs[1].set_ylim(0,self.N+0.2)
#                 axs[1].set_yticks(np.arange(self.N+1))
#                 axs[1].grid(alpha=0.3)
#                 axs[1].set_ylim(0,self.N+0.2)
#                 axs[1].set_yticks(np.arange(self.N+1))
#                 axs[1].legend()
#                 axs[1].grid(alpha=0.3)

#             ## delta T perturbation plot    
#             if self.pertub_button is True:
#                 axs[2].plot(deltaT, range(0, self.N+1), 
#                             marker='o',mfc='white',mec='firebrick',
#                             color='firebrick',lw=2,alpha=1,ls='--',label='Perturbation\n$\Delta$T vs. Control')
#                 axs[2].set_xlabel('$\Delta$T ($\degree$K)')
#                 axs[2].axvline(0,lw=1,c='k',alpha=0.5)
#                 axs[2].set_ylim(0,self.N+0.2)
#                 axs[2].set_yticks(np.arange(self.N+1))
#                 axs[2].grid(alpha=0.3)
#                 axs[2].set_ylim(0,self.N+0.2)
#                 axs[2].set_yticks(np.arange(self.N+1))
#                 axs[2].legend()
#                 axs[2].grid(alpha=0.3)
                
#             plt.tight_layout()
#             plt.show()
            
#             control_Ts = (temperature_M[best_index_all].squeeze())[0]
#             if (self.SW_strat_absorption > 0):
#                 dT_sw = str(np.round(temperature_vector_no_SW[0] - control_Ts,1))
#                 print('No SW: ΔTAS = ' + dT_sw)
#             if (self.heatflux > 0):
#                 dT_conv = str(np.round(temperature_vector_no_convection[0] - control_Ts,1))
#                 print('No convection: ΔTAS = ' + dT_conv)
#             if self.pertub_button is True:
#                 dT_pert = str(np.round(deltaT[0],2))
#                 print('Emissivity perturbation: ΔTAS = ' + dT_pert)
            
            ################ end additional plots #################
            #######################################################
            
            self.calc_button.value = False
