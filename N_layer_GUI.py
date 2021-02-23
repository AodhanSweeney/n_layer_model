import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
# Import radiative transfer utils box
import utils

class n_layer_gui:

    def __init__(self):
        """
        
        Collect scalable physical parameters to be used for the N-Layer radiative model.
        
        """
        self.N_slider = widgets.IntSlider(min=1, max=100, value=2, description='N-Layers: ')
        widgets.interact(self.get_n_slider, N_slider=self.N_slider)
        
        self.albedo_slider = widgets.FloatSlider(min=0, max=1, value=0.3, description='Albedo: ')
        widgets.interact(self.get_albedo_slider, albedo_slider=self.albedo_slider)
        
        self.S0_slider = widgets.IntSlider(min=10, max=10000, value=1368, description='$S_{0}$ ($W/m^2$): ')
        widgets.interact(self.get_S0_slider, S0_slider=self.S0_slider)
        
        self.heatflux_slider = widgets.IntSlider(min=10, max=1000, value=110, 
                                                 description='Heat Flux \nTotal ($W/m^2$): ')
        widgets.interact(self.get_heatflux_slider, heatflux_slider=self.heatflux_slider)
        
        #self.pertub_button = widgets.ToggleButton(value=False, description='Perturbation?',
        #                                         disabled=False, button_style='info',
        #                                         tooltip='Description')
        #widgets.interact(self.get_perturbation, calc_button=self.pertub_button)
        
        self.calc_button = widgets.ToggleButton(value=False, description='Radiate',
                                                 disabled=False, button_style='info',
                                                 tooltip='Description')
        widgets.interact(self.get_temps, calc_button=self.calc_button)
    
    

    def get_n_slider(self, N_slider):
        self.N = N_slider
    def get_albedo_slider(self, albedo_slider):
        self.albedo = albedo_slider
    def get_S0_slider(self, S0_slider):
        self.S0 = S0_slider
    def get_heatflux_slider(self, heatflux_slider):
        self.heatflux = heatflux_slider
    #def get_perturbation(self, pertub_button):
        #self.perturbation = perturbation
        #if self.perturb_button is True:
            
        
    def get_temps(self, calc_button):
        """
        Query whether track button has been toggled, and calculate the radiation matrix before plotting.
        
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
            
            forcings = utils.forcings_vector(self.N, insolation, upward_heatflux)
            
           

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