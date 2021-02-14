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

def vertical_heat_flux_profile(N, total_vertical_heat_flux):
    number_of_layers = N+1
    # Stratosphere will be only radiatively coupled, so only have heatflux for bottom 85% of layers
    number_of_layers_w_heat_flux = int(number_of_layers*0.85)
    linearly_decreasing_heat_flux = np.linspace(1, 0, number_of_layers_w_heat_flux)
    
    normalized_linear_decreasing_heat_flux = linearly_decreasing_heat_flux/np.sum(linearly_decreasing_heat_flux)
    
    # Now we can create the heat flux profile which spans the first 85% of layers and sums to total heat flux
    actual_lin_dec_heat_flux_profile = normalized_linear_decreasing_heat_flux*total_vertical_heat_flux
    total_atmospheric_heatflux_profile = np.zeros(number_of_layers)
    
    total_atmospheric_heatflux_profile[:number_of_layers_w_heat_flux] = actual_lin_dec_heat_flux_profile
    return(total_atmospheric_heatflux_profile)

def forcings_vector(N, insolation, heat_flux_profile):
    # Let all the forcings be zero for now, besides the insolation
    forcings = np.zeros(N+1)
    forcings[0] = insolation*-1
    downward_flux = np.insert(heat_flux_profile[:-1], 0, 0)    
    net_forcings = forcings - heat_flux_profile + downward_flux
    
    
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