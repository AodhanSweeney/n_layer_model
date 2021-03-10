import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from scipy import interpolate
# Import radiative transfer utils box
import utils

class n_layer_gui:

    def __init__(self):
        """
        
        Collect scalable physical parameters to be used for the N-Layer radiative model.
        
        """
        ###############################
        self.N_slider = widgets.IntSlider(min=1, max=100, value=2, description='N-Layers: ')
        widgets.interact(self.get_n_slider, N_slider=self.N_slider)
        ###############################
        self.albedo_slider = widgets.FloatSlider(min=0, max=1, value=0.3, description='Albedo: ')
        widgets.interact(self.get_albedo_slider, albedo_slider=self.albedo_slider)
        ###############################
        self.S0_slider = widgets.IntSlider(min=10, max=10000, value=1368, description='$S_{0}$ ($W/m^2$): ')
        widgets.interact(self.get_S0_slider, S0_slider=self.S0_slider)
        ###############################
        self.SW_slider = widgets.IntSlider(min=0, max=100, value=3, 
                                                            description='%SW in Stratosphere: ')
        widgets.interact(self.get_SW_slider, SW_slider=self.SW_slider)
        ###############################
        self.heatflux_slider = widgets.IntSlider(min=0, max=1000, value=110, 
                                                 description='Heat Flux \nTotal ($W/m^2$): ')
        widgets.interact(self.get_heatflux_slider, heatflux_slider=self.heatflux_slider)
        ###############################
        self.togglebutton = widgets.ToggleButtons(options=['Tropics', 'Poles'], description='Location?', disabled=False, 
                                                  button_style='danger')
        widgets.interact(self.get_togglebutton, togglebutton=self.togglebutton)
        ###############################
        self.pertub_button = widgets.ToggleButton(value=False, description='Perturbation?', disabled=False, 
                                                  button_style='success', tooltip='Description')
        widgets.interact(self.get_perturbation, pertub_button=self.pertub_button)
        ###############################
        self.calc_button = widgets.ToggleButton(value=False, description='Radiate', disabled=False, button_style='info',
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
    def get_togglebutton(self, togglebutton):
        self.togglebutton = togglebutton                    
    def get_perturbation(self, pertub_button):
        self.pertub_button = pertub_button
        if self.pertub_button is True:
            self.layer_perturb = widgets.IntSlider(min=1, max=self.N, value=1, description='Which layer?: ')
            widgets.interact(self.get_perturb_location, layer_perturb=self.layer_perturb)
            
            self.perturb_magnitude = widgets.FloatText(value=1.0, description='$W/m^2$:', disabled=False)
            widgets.interact(self.get_perturb_magnitude, perturb_magnitude=self.perturb_magnitude)
    def get_perturb_location(self, layer_perturb):
        self.layer_perturb = layer_perturb   
    def get_perturb_magnitude(self, perturb_magnitude):
        self.perturb_magnitude = perturb_magnitude  
        
    
    def get_temps(self, calc_button):
        """
        Query whether track button has been toggled, and calculate the radiation matrix before plotting.
        
        """
        if self.togglebutton == 'Tropics':
            era5_profile = np.load('tropical_profile.npy')
        elif self.togglebutton == 'Poles':
            era5_profile = np.load('polar_profile.npy')
        
        era5_layers = np.linspace(0, self.N+1, 32)
        cb_interp = interpolate.CubicSpline(era5_layers, era5_profile)
        
        profile_for_emissivities = cb_interp(np.arange(0, self.N+1))
        
        
        """
        
        Vince, here you should do the monte carlo thing, the profile you want to fit to is "profile_for_emissivities"


        """
        
        if self.calc_button.value is True:
            # Create some random emissivities to use, surface emissivity is always one, no emissivity should be zero
            emmisivities = np.repeat(.05,self.N+1)
            emmisivities[0] = 1
            
            # Now get our emissivity Matrix
            R_up_matrix = utils.R_up_matrix(self.N, emmisivities)
            total_emissivity_matrix = utils.emissivity_matrix(R_up_matrix, emmisivities)

            # We also need our forcings vector
            insolation = (self.S0/4)*(1 - self.albedo)
            
            upward_heatflux = utils.vertical_heat_flux_profile(self.N, self.heatflux, 'exponential')
            
            perturbation_vector = utils.perturbation_profile(self.N, self.layer_perturb, self.perturb_magnitude)
            
            forcings = utils.forcings_vector(self.N, insolation, upward_heatflux, self.SW_strat_absorption, perturbation_vector)
            
           

            # Now find the temperature vector using the emissivity matrix and the forcings
            temperature_vector = utils.temperature(total_emissivity_matrix, forcings)


            fig, axs = plt.subplots(1,3, figsize=(12,6))

            # Emissivities plot
            axs[0].plot(emmisivities, range(0, self.N+1), color='black')
            axs[0].set_ylabel('Layer number (n)')
            axs[0].set_xlabel('Emissivity')
            axs[0].set_ylim(0,self.N+1)

            axs[1].set_xlabel('Vertical Heat Flux (W/$m^2$)')
            axs[1].plot(upward_heatflux, range(0, self.N+1), color='dodgerblue')
            axs[1].set_ylim(0,self.N+1)

            axs[2].plot(temperature_vector, range(0, self.N+1), color='firebrick', linewidth=3)
            axs[2].set_xlabel('Temperature ($\degree$K)')
            axs[2].set_ylim(0,self.N+1)
            
            self.calc_button.value = False