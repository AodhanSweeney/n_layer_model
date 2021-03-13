import numpy as np 
from scipy import linalg

def upwelling_radiation(j, emissivities):
    emissivities_for_each_layer = []
    for l in range(j, -1, -1):
        #emmisivity of layer j
        E_j = emissivities[j] 
        
        # emmisivity of source layer (index j minus l minus one)
        if j-l-1 == -1:
            E_jmlm1 = 0.0 
        else:
            E_jmlm1 = emissivities[j-l-1]

        # For layers that are not j or source, find effective emmisivity after absorption 
        effective_emissivity_of_transmission_layers = []
        for k in range(j-l, j):
            fraction_of_emissivity_transfered = 1 - (emissivities[k])
            effective_emissivity_of_transmission_layers.append(fraction_of_emissivity_transfered)
         
        # Multiply all the effective emissivities of transmission layers, to get one final effective emissivity
        if len(effective_emissivity_of_transmission_layers) == 0:
            product_of_transmission_layers = 1
        elif len(effective_emissivity_of_transmission_layers) > 0:
            product_of_transmission_layers = np.prod(effective_emissivity_of_transmission_layers)
            
        #Now multiply the source and reciving layers by the effective emissivity of the transmission layers
        total_emissivity_product = E_j*E_jmlm1*product_of_transmission_layers
        emissivities_for_each_layer.append(total_emissivity_product)
        
    # return the emissivities of each layer now, but take out first entry which is always zero
    return(emissivities_for_each_layer[1:])


def R_up_matrix(N, emissivities):
    empty_Np1_Np1_array = np.zeros((N+1,N+1))
    # We need to change elements of array based off the row 
    for j in range(0, N+1):
        row_to_change = empty_Np1_Np1_array[j]
        
        # Find emissivities associated with upwelling radiation
        upwelling_terms = upwelling_radiation(j, emissivities)
        
        # Diagonal elements should still remain zero, we will deal with this later
        if len(upwelling_terms) == 0:
            continue
        
        elif len(upwelling_terms) > 0:
            # Here we will change each element in our matrix that corresponds to upwelling radiation
            # In the end, this will only make us have a lower triangular matrix with a diagonal of zero
            for upwelling_idx, upwelling_emissivity in enumerate(upwelling_terms):
                row_to_change[upwelling_idx] = upwelling_emissivity
            row_of_upwelling_radiation = row_to_change
                
        # After the row is changed to the upwelling values, put this row into our matrix
        empty_Np1_Np1_array[j] = row_of_upwelling_radiation
    
    # Now we have an upwelling radiation matrix, which has a zero diagonal and is lower triangular!
    upwelling_radiation_matrix = np.matrix(empty_Np1_Np1_array)
    return(upwelling_radiation_matrix)
    
    
def emissivity_matrix(R_up_matrix, emissivities):
    # downwelling radiation matrix is transpose of upwelling
    R_down_matrix = R_up_matrix.T
    
    # The sum of the upwelling and downwelling radiation is the recieved radiation matrix
    # This matrix does not account for the emitted radiation of each layer, or the forcings
    recieved_radiation_matrix = R_up_matrix + R_down_matrix
    
    # We need to add in the diagonal to the recieved radiation matrix, the diagonal accounts for emitted
    # radiation of each layer, it should have opposite sign than recieved radiation.
    # Unbounded layers eminate both up and down, so all layers besides surface should be multiplied by 2.
    emissivities_diagonal = -2*np.diag(emissivities)
    emissivities_diagonal[0][0] = -1
    
    # total emissivity matrix will be sum of both radiation recieved and radiation emmitted! 
    emissivity_matrix = recieved_radiation_matrix + emissivities_diagonal
    return(emissivity_matrix)

def vertical_heat_flux_profile(N, surf_vertical_heat_flux, profile_type):
    
    # Stratosphere will be only radiatively coupled, so only have heatflux for bottom 85% of layers
    # includes surface; tropopause will only receive heat from below
    N_w_heat_flux = int(N*0.85)
    
    if profile_type == 'linear':
        linearly_decreasing_heat_flux = np.linspace(1, 0, N_w_heat_flux)
        
        ## normalizing rebases to the surface heat flux (SHF has value = 1) (extraneous for linear case)
        normalized_linear_decreasing_heat_flux = linearly_decreasing_heat_flux/linearly_decreasing_heat_flux[0]

        # Now we can create upward heat flux profile which stops below tropopause
        actual_lin_dec_heat_flux_profile = normalized_linear_decreasing_heat_flux*surf_vertical_heat_flux
        total_atmospheric_heatflux_profile = np.zeros(N+1)
        total_atmospheric_heatflux_profile[:N_w_heat_flux] = actual_lin_dec_heat_flux_profile

        ## get net flux by subtracting incoming heat flux from layer below
        total_atmospheric_heatflux_profile[1:N_w_heat_flux+1] -= total_atmospheric_heatflux_profile[:N_w_heat_flux]
        
        return(total_atmospheric_heatflux_profile)
    
    elif profile_type == 'exponential':
        exponentially_decreasing_heat_flux = np.geomspace(1, 1e-5, N_w_heat_flux)
        
        ## normalizing rebases to the surface heat flux (SHF has value = 1) (extraneous for exponential case)
        normalized_exp_decreasing_heat_flux = exponentially_decreasing_heat_flux/exponentially_decreasing_heat_flux[0]

        # Now we can create upward heat flux profile which stops below tropopause
        actual_exp_dec_heat_flux_profile = normalized_exp_decreasing_heat_flux*surf_vertical_heat_flux
        total_atmospheric_heatflux_profile = np.zeros(N+1)
        total_atmospheric_heatflux_profile[:N_w_heat_flux] = actual_exp_dec_heat_flux_profile

        ## get net flux by subtracting incoming heat flux from layer below
        total_atmospheric_heatflux_profile[1:N_w_heat_flux+1] -= total_atmospheric_heatflux_profile[:N_w_heat_flux]
        
        return(total_atmospheric_heatflux_profile)
    
    elif profile_type == 'tanh':
        xx = np.flip(np.linspace(0,np.pi,N_w_heat_flux) - np.pi)
        yy = np.tanh(xx) + 1
        
        ## normalizing rebases to the surface heat flux (SHF has value = 1)
        normalized_tanh_decreasing_heat_flux = yy/yy[0]
        
        # Now we can create the upward heat flux profile which spans the first 85% of layers
        actual_tanh_dec_heat_flux_profile = normalized_tanh_decreasing_heat_flux * surf_vertical_heat_flux
        
        total_atmospheric_heatflux_profile = np.zeros(N+1)
        total_atmospheric_heatflux_profile[:N_w_heat_flux] = actual_tanh_dec_heat_flux_profile

        # get net flux by subtracting incoming heat flux from layer below
        total_atmospheric_heatflux_profile[1:N_w_heat_flux+1] -= total_atmospheric_heatflux_profile[:N_w_heat_flux]
        
        return(total_atmospheric_heatflux_profile)
    

def forcings_vector(N, insolation, heat_flux_profile, SW_strat_absorption):
    ## Initialize the forcings at zero
    forcings = np.zeros(N+1)
    
    # surface forcing is insolation less the stratospheric absorption
    forcings[0] = insolation*-1 + SW_strat_absorption
   
    ## separate atmospheric layers into troposphere and stratosphere
    ## subtract 1 layer from troposphere to allow for tropopause
    N_troposphere = np.floor(0.85*N) - 1 
    
    N_tpause = 1
    N_tpause_location = int(0.85*N) # layer index for the tropopause (layer 0 is surface)
    
    N_stratosphere = N - N_troposphere - N_tpause
    
    # stratospheric forcing will be spread uniformly over each strat layer
    forcings[N_tpause_location+1:] = -SW_strat_absorption / N_stratosphere

    net_forcings = forcings + heat_flux_profile 
    
    # Forcings should also be scaled by 1/sigma
    sigma = 5.67e-8 # W * m^-2 * K^-4
    one_over_sigma = 1/sigma
    return(net_forcings*one_over_sigma)

def temperature(total_emissivity_matrix, forcings):
    """
    We have an matrix equation of (total_emissivity_matrix) X (T_i^4) = F_i
    Where total emissivity matrix is N+1 x N+1, T_i^4 is a column vector of N+1 and 
    forcings are another column vector, scaled by 1/sigma.
    """
    inverse_of_emissivity_matrix = linalg.inv(total_emissivity_matrix)
    temperature_vector = forcings.dot(inverse_of_emissivity_matrix)
    return(np.array(temperature_vector)**(1/4))

def perturb_forcing(perturbed_emissivity_matrix, original_emissivity_matrix, temperature_vector):
    """
    We have an matrix equation of (total_emissivity_matrix) X (T_i^4) = F_i
    Where total emissivity matrix is N+1 x N+1, T_i^4 is a column vector of N+1 and 
    forcings are another column vector, scaled by 1/sigma.
    """
    T_4_vector = temperature_vector**(4)
    
    perturb_forcings = perturbed_emissivity_matrix.dot(T_4_vector)
    original_forcings = original_emissivity_matrix.dot(T_4_vector)
        
    difference = sum(perturb_forcings.tolist()[0]) - sum(original_forcings.tolist()[0])
    
    sigma = 5.67e-8
    
    return(-1*sigma*difference)
