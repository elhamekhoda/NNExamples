import matplotlib.pyplot as plt
import numpy as np
np.random.seed(12122)
import keras
import keras.backend as K
import h5py
from keras.utils import plot_model


def mixture_density(nb_components, target_dimension=1):

    """ The Mixture Density output layer. Use with the keras functional api:
        inputs = Inputs(...)
        net = ....
        model = Model(input=[inputs], output=[mixture_density(2)(net)])
    """

    def layer(X):
        pi = keras.layers.Dense(nb_components, activation='softmax')(X)
        mu = keras.layers.Dense(nb_components, activation='linear')(X)
        std = keras.layers.Dense(nb_components, activation=K.exp)(X)
        return keras.layers.Merge(mode='concat')([pi,mu,std])

    return layer


def mixture_density_loss(nb_components, target_dimension=1):

    """ Compute the mixture density loss:
        \begin{eqnarray}
          P(Y|X) = \sum_i P(C_i) N(Y|mu_i(X), beta_i(X)) \\
          Loss(Y|X) = - log(P(Y|X))
        \end{eqnarray}                                                                                                                                                                                    
    """

    def loss(y_true, y_pred):

        batch_size = K.shape(y_pred)[0]

        # Each row of y_pred is composed of (in order):
        # 'nb_components' prior probabilities
        # 'nb_components'*'target_dimension' means
        # 'nb_components' sigmas
        priors = y_pred[:,:nb_components]

        m_i0 = nb_components
        m_i1 = m_i0 + nb_components * target_dimension
        means = y_pred[:,m_i0:m_i1]

        p_i0 = m_i1
        p_i1 = p_i0 + nb_components * target_dimension
        std = y_pred[:,p_i0:p_i1]

        # Now, compute the (x - mu) vector. Have to reshape y_true and
        # means such that the subtraction can be broadcasted over
        # 'nb_components'
        means = K.reshape(means, (batch_size , nb_components, target_dimension))
        x = K.reshape(y_true, (batch_size, 1, target_dimension)) - means


        # Compute the dot-product over the target dimensions. There is
        # one dot-product per component per example so reshape the
        # vectors such that a batch_dot product can be carried over
        # the axis of target_dimension
        x = K.reshape(x, (batch_size * nb_components, target_dimension))
        std = K.reshape(std, (batch_size * nb_components, target_dimension))
        InvStdsq = 1/(std*std)

        # reshape the result into the natural structure
        expargs = K.reshape(K.batch_dot(-0.5 * x * InvStdsq, x, axes=1), (batch_size, nb_components))

        # There is also one determinant per component per example
        dets = K.reshape(K.abs(K.prod(std, axis=1)), (batch_size, nb_components))
        norms = 1/K.sqrt(np.power(2*np.pi,target_dimension)*dets) * priors

        # LogSumExp, for enhanced numerical stability
        x_star = K.max(expargs, axis=1, keepdims=True)
        logprob = - x_star - K.log(K.sum(norms * K.exp(expargs - x_star), axis=1))



        return logprob

        '''
        #===========================================================================
        # Loss taken from Gilles's code
        #===========================================================================
        y_true = y_true.ravel()
        
        mu = y_pred[:, :nb_components]
        sigma = y_pred[:, nb_components:2*nb_components]
        pi = y_pred[:, 2*nb_components:]
        
        pdf = pi[:, 0] * ((1. / np.sqrt(2. * np.pi)) / sigma[:, 0] *
            K.exp(-(y_true - mu[:, 0]) ** 2 / (2. * sigma[:, 0] ** 2)))
        
        for c in range(1, nb_components):
            pdf += pi[:, c] * ((1. / np.sqrt(2. * np.pi)) / sigma[:, c] *
                        K.exp(-(y_true - mu[:, c]) ** 2 / (2. * sigma[:, c] ** 2)))
        
        logprob = -K.log(pdf)
        
      '''
    return loss


'''
####################################################################################
## Loss function by L-G #########


def mixture_density_loss(nb_components, target_dimension=1):

    """ Compute the mixture density loss:
        \begin{eqnarray}
          P(Y|X) = \sum_i P(C_i) N(Y|mu_i(X), beta_i(X)) \\
          Loss(Y|X) = - log(P(Y|X))
        \end{eqnarray}
    """

    def loss(y_true, y_pred):

        batch_size = K.shape(y_pred)[0]

        # Each row of y_pred is composed of (in order):
        # 'nb_components' prior probabilities
        # 'nb_components'*'target_dimension' means
        # 'nb_components'*'target_dimension' precisions
        priors = y_pred[:,:nb_components]

        m_i0 = nb_components
        m_i1 = m_i0 + nb_components * target_dimension
        means = y_pred[:,m_i0:m_i1]

        p_i0 = m_i1
        p_i1 = p_i0 + nb_components * target_dimension
        precs = y_pred[:,p_i0:p_i1]

        # Now, compute the (x - mu) vector. Have to reshape y_true and
        # means such that the subtraction can be broadcasted over
        # 'nb_components'
        means = K.reshape(means, (batch_size , nb_components, target_dimension))
        x = K.reshape(y_true, (batch_size, 1, target_dimension)) - means


        # Compute the dot-product over the target dimensions. There is
        # one dot-product per component per example so reshape the
        # vectors such that a batch_dot product can be carried over
        # the axis of target_dimension
        x = K.reshape(x, (batch_size * nb_components, target_dimension))
        precs = K.reshape(precs, (batch_size * nb_components, target_dimension))
        # reshape the result into the natural structure
        expargs = K.reshape(K.batch_dot(-0.5 * x * precs, x, axes=1), (batch_size, nb_components))

        # There is also one determinant per component per example
        dets = K.reshape(K.abs(K.prod(precs, axis=1)), (batch_size, nb_components))
        norms = K.sqrt(dets/np.power(2*np.pi,target_dimension)) * priors

        # LogSumExp, for enhanced numerical stability
        x_star = K.max(expargs, axis=1, keepdims=True)
        logprob = - x_star - K.log(K.sum(norms * K.exp(expargs - x_star), axis=1))

        return logprob

    return loss
    '''
    
####################################################################################

# Sanity test

def gen_data(N):

    """ Generate a 2 component distribution by
        adding noise to sigmoid(x) and (1 - sigmoid(x)) """

    def component_1(N):
        x = np.random.uniform(-10, 10, N)
        y = 1.0 / (1.0 + np.exp(-x))
        #r = np.random.normal(size=N)
        #y = np.float32(np.sin(0.75*x)*7.0 + x*0.5 + r)
        z = np.random.normal(scale=0.05, size=N)
        
        #temp_data = x
        #x = y
        #y = temp_data

        return x, y + z
    def component_2(N):
        x = np.random.uniform(-10, 10, N)
        y = 1 - 1.0 / (1.0 + np.exp(-x))
        #r = np.random.normal(size=N)
        #y = np.float32(np.sin(0.75*x)*7.0 + x*0.5 + r)
        z = np.random.normal(scale=0.05, size=N)
        #temp_data = x
        #x = y
        #y = temp_data
        return x, y + z

    n1 = N / 2#np.random.randint(N+1)
    n2 = N - n1

    x1, y1 = component_1(n1)
    x2, y2 = component_2(n2)

    x = np.concatenate([x1, x2])
    y = np.concatenate([y1, y2])

    ishuffle = np.arange(N)
    np.random.shuffle(ishuffle)

    return x[ishuffle], y[ishuffle]

####################################################################################

def main():

    trainX, trainY = gen_data(10000)
    validX, validY = gen_data(1000)
    testX, testY = gen_data(1000)



    # The target distribution has 2 components, so a 2 component
    # mixture should model it very well
    inputs = keras.layers.Input(shape=(1,))
    h = keras.layers.Dense(300, activation='relu')(inputs)
    model = keras.models.Model(inputs=inputs, outputs=[mixture_density(2)(h)])

    # The gradients can get very large when the estimated precision
    # gets very large (small variance) which makes training
    # unstable. If this happens, look into the "clipvalue" or
    # "clipnorm" parameter of the keras optimizers to limit the size
    # of the gradients
    model.compile(
        optimizer=keras.optimizers.Adam(lr=0.001),
        loss=mixture_density_loss(nb_components=2),
    )

    plot_model(model, to_file='model.png')
    print(model.summary())

    history = model.fit(
        x=trainX,
        y=trainY,
        batch_size=32,
        epochs=100,
        validation_data=(validX, validY),
        callbacks=[
            keras.callbacks.ModelCheckpoint('mdn_mod.h5', verbose=1, save_best_only=True)
        ],
        verbose=2
    )
    np.savetxt('training_loss.txt', history.history['loss'])
    np.savetxt('validation_loss.txt', history.history['val_loss'])

    if keras.__version__.split('.')[0] == '1':
        saved = h5py.File('mdn_mod.h5', 'r+')
        if 'optimizer_weights' in saved.keys():
            del saved['optimizer_weights']
        saved.close()

    keras.activations.exp = K.exp
    model = keras.models.load_model(
        'mdn_mod.h5',
        custom_objects={
            'loss': mixture_density_loss(nb_components=2)
        }
    )

    nb_components = 2

    y_pred = model.predict(testX)
    y_smp = np.zeros(y_pred.shape[0])
    
    for i in range(y_pred.shape[0]):
        priors = y_pred[i,:nb_components]
        means = y_pred[i,nb_components:2*nb_components]
        std = y_pred[i,2*nb_components:3*nb_components]
               # Sample a component of the mixture according to the priors
        #cpn = np.random.choice([0, 1, 2, 3,4,5,6,7,8,9], p=priors)
        cpn = np.random.choice([0,1], p=priors)
        # Sample a data point for the chosen mixture
        y_smp[i] = np.random.normal(loc=means[cpn], scale=1.0/np.sqrt(std[cpn]))

    plt.scatter(testX, testY,c='red', label='True data')
    plt.scatter(testX, y_smp, label='Generated data')
    plt.legend(loc='best')
    # The distributions should match very well!
    plt.show()

if __name__ == '__main__':
    main()
