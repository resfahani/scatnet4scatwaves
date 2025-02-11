import numpy as np

def normalizing(scattering_coefficients, sil = 5):

    ss = scattering_coefficients.copy()

    c1 = ss[0].copy()
    c2 = ss[1].copy()

    c2hat = c2/ (np.nanmedian(c1, -1) + sil + 0* np.nanmax(c1, -1) )[:,:,np.newaxis , np.newaxis] 
    c2hat[np.isnan(scattering_coefficients[1])] == np.nan

    return [c1, c2hat]


def preprocessing_(sc: np.ndarray,
                    scaler, 
                    indx:[int, str] = None, 
                    epsilon: int = 1e-15):

    # sc = normalizing(scatteringcoeff, sil = 1e-1) 
    
    if indx == None:
        order_1 = sc[0][:, :, ::1]
        order_2 = sc[1][:, :, ::, :][:,:,:,::]
    else:
        indx = indx
        order_1 = sc[0][:, indx, :]
        order_2 = sc[1][:, indx, :, :]
        
    order_1 = order_1.reshape(order_1.shape[0] , -1)
    order_2 = order_2.reshape(order_2.shape[0], -1)

    order_1 =  np.log((order_1[:, :]**1) + epsilon)
    order_1 = np.nan_to_num(order_1, 0)

    order_2 =  np.log((order_2[:, :]**1) + epsilon)
    order_2 = np.nan_to_num(order_2, 0)

    if scaler is not None:
        order_1  = scaler[0].fit_transform(order_1)
        order_2  = scaler[1].fit_transform(order_2)
        
        print('Normalization applied')

    coeff = np.hstack((order_1, order_2 ))
    coeff = np.nan_to_num(coeff, 0)

    return coeff


def reader_coeff(path_file, network):

    scm_ = np.load(path_file)
    sc = [scm_['order_1'], scm_['order_2']]

    for i in range(len(network.banks[0].centers)):
        n = network.banks[0].centers[i] <= network.banks[1].centers
        sc[1][:,:, i, n] = np.nan
    return sc
