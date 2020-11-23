################################################################################
# Author: Maleakhi, Faidon, Harry, Jamie
# Date: 14/November/2019
# Description: A neural network mini library for Part 1 (Introduction to 
# Machine Learning) using Numpy.
################################################################################
import numpy as np
import pickle


def xavier_init(size, gain=1.0):
    """
    Xavier initialization of network weights.
    """
    low = -gain * np.sqrt(6.0 / np.sum(size))
    high = gain * np.sqrt(6.0 / np.sum(size))
    return np.random.uniform(low=low, high=high, size=size)


class Layer:
    """
    Abstract layer class.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError()

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def backward(self, *args, **kwargs):
        raise NotImplementedError()

    def update_params(self, *args, **kwargs):
        pass


class MSELossLayer(Layer):
    """
    MSELossLayer: Computes mean-squared error between y_pred and y_target.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def _mse(y_pred, y_target):
        return np.mean((y_pred - y_target) ** 2)

    @staticmethod
    def _mse_grad(y_pred, y_target):
        return 2 * (y_pred - y_target) / len(y_pred)

    def forward(self, y_pred, y_target):
        self._cache_current = y_pred, y_target
        return self._mse(y_pred, y_target)

    def backward(self):
        return self._mse_grad(*self._cache_current)


class CrossEntropyLossLayer(Layer):
    """
    CrossEntropyLossLayer: Computes the softmax followed by the negative log-
    likelihood loss.
    """

    def __init__(self):
        self._cache_current = None

    @staticmethod
    def softmax(x):
        numer = np.exp(x - x.max(axis=1, keepdims=True))
        denom = numer.sum(axis=1, keepdims=True)
        return numer / denom

    def forward(self, inputs, y_target):
        assert len(inputs) == len(y_target)
        n_obs = len(y_target)
        probs = self.softmax(inputs)
        self._cache_current = y_target, probs

        out = -1 / n_obs * np.sum(y_target * np.log(probs))
        return out

    def backward(self):
        y_target, probs = self._cache_current
        n_obs = len(y_target)
        return -1 / n_obs * (y_target - probs)


class SigmoidLayer(Layer):
    """
    SigmoidLayer: Applies sigmoid function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        """
        Return element wise transformation of inputs using sigmoid activation function.

        Arguments:
            x {np.ndarray} -- to be applied activation function element wise

        Returns:
            {np.ndarray} -- transformed x
        """
        self._cache_current = x # for backward pass previous layers

        # Sigmoid formula
        return 1/(1+np.exp(-x))

    def backward(self, grad_z):
        """
        Compute and return gradient of function with respect to inputs of the layer.

        Arguments:
            grad_z {np.ndarray} -- dloss/dz from previous layers
        
        Returns:
            {np.ndarray} -- dloss/dx
        """
        # f'(x)= f(x)(1-f(x)) for the sigmoid function
        f_prime = self.forward(self._cache_current)*(1-self.forward(self._cache_current))
        
        grad_x = np.multiply(grad_z, f_prime) # Hadamard product/ element-wise multiplication
        return grad_x


class ReluLayer(Layer):
    """
    ReluLayer: Applies Relu function elementwise.
    """

    def __init__(self):
        self._cache_current = None

    def forward(self, x):
        """
        Return elementwise transformation of x.

        Arguments:
            x {np.ndarray} -- input
        
        Returns:
            {np.ndarray} -- transformed input
        """
        self._cache_current = x
        
        # Apply relu
        return np.maximum(x,0)

    def backward(self, grad_z):
        """
        Compute and return gradient of function with respect to inputs of the layer.

        Inputs:
            grad_z {np.ndarray} -- dloss/dz

        Outputs:
            {np.ndarray} -- dloss/dx
        """
        x = self._cache_current

        # Here two steps are achieved in one.
        # first the gradients of z which correspond to x values less than 0 are set to 0
        # then the gradients of z which correspond to x values greater than 0 are set to 1 but
        # key in the same step are muliplied (in the chain rule) by grad_z. Therefore the return
        # grad loss, for these indices is simply the corresponding values of the grad_z, unchanged.
        grad_x = grad_z
        grad_x[x<=0] = 0

        return grad_x


class LinearLayer(Layer):
    """
    LinearLayer: Performs affine transformation of input.
    """

    def __init__(self, n_in, n_out):
        """
        Constructor.

        Arguments:
            n_in {int} -- Number (or dimension) of inputs.
            n_out {int} -- Number (or dimension) of outputs.
        """
        self.n_in = n_in
        self.n_out = n_out

        # The xavier_init function is used to initialize network weights based on
        # Xavier Glorot and Yoshua Bengio. We will first initialise the weight to
        # a n_in x n_out matrix.
        self._W = xavier_init((n_in, n_out))
        self._b = np.zeros((1, n_out)) # after stacking, bias will be of size batch_size x n_out

        self._cache_current = None
        self._grad_W_current = None
        self._grad_b_current = None

    def forward(self, x):
        """
        Performs forward pass through the layer (i.e. returns XW + B).

        Logs information needed to compute gradient at a later stage in
        `_cache_current`.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, n_in).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size, n_out)
        """
        # Stack b and get the dimension ready
        batch_size, _ = x.shape
        b = self._b.ravel()
        B = np.stack([b]*batch_size, axis=0) # stack b to become of size batch_size x n_out

        # Now perform matrix calculation XW + b
        output = x @ self._W + B
        self._cache_current = x # store data that is useful for computing back propagation (x)

        return output

    def backward(self, grad_z):
        """
        Given `grad_z`, the gradient of some scalar (e.g. loss) with respect to
        the output of this layer, performs back pass through the layer (i.e.
        computes gradients of loss with respect to parameters of layer and
        inputs of layer).

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (batch_size, n_out).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, n_in).
        """
        # Recall from lecture slide (dloss/dW = X.T @ dloss/dz)
        self._grad_W_current = self._cache_current.T @ grad_z

        # (dloss/db = 1^T @ dloss/dz)
        # ones is a row vector of size (1, batch_size)
        self._grad_b_current = np.ones((1, grad_z.shape[0])) @ grad_z

        # (dloss/dX = dloss/dz @ W.T)
        grad_x = grad_z @ self._W.T # to be pass to layer before
        return grad_x

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        layer's parameters using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        # W <- W - alpha @ dloss/dW
        self._W = self._W - (learning_rate * self._grad_W_current)


class MultiLayerNetwork(object):
    """
    MultiLayerNetwork: A network consisting of stacked linear layers and
    activation functions.
    """

    def __init__(self, input_dim, neurons, activations):
        """
        Constructor.

        Arguments:
            input_dim {int} -- Dimension of input (excluding batch dimension).
            neurons {list} -- Number of neurons in each layer represented as a
                list (the length of the list determines the number of layers).
            activations {list} -- List of the activation function to use for
                each layer.
        """
        self.input_dim = input_dim
        self.neurons = neurons
        self.activations = activations

        # Initialise layers
        self._layers = []
        # Initialise input dimension
        previous_dim = self.input_dim

        # Input layer has no activation function:
        layer = LinearLayer(previous_dim, previous_dim)
        self._layers.append(layer)

        # Loop through inputs to initialise layers
        for counter, neuron in enumerate(neurons):
            layer = LinearLayer(previous_dim, neuron)
            self._layers.append(layer)

            # Add activation function to the layer if "identity" is not chosen
            if activations[counter] == "relu":
                activation = ReluLayer()
                self._layers.append(activation)
            elif activations[counter]=="sigmoid":
                activation = SigmoidLayer()
                self._layers.append(activation)

            # Set input dim to next layer to be output dim of current
            previous_dim = neuron

    def forward(self, x):
        """
        Performs forward pass through the network.

        Arguments:
            x {np.ndarray} -- Input array of shape (batch_size, input_dim).

        Returns:
            {np.ndarray} -- Output array of shape (batch_size,
                #_neurons_in_final_layer)
        """
        # Initialise input to first layer
        inputs = x

        # Iterate every linear layer, activation and generate output
        for layer in self._layers:
            # Output of layer becomes input for next layer
            inputs = layer.forward(inputs)

        # Output of last layer is multilayer output
        return inputs

    def __call__(self, x):
        return self.forward(x)

    def backward(self, grad_z):
        """
        Performs backward pass through the network.

        Arguments:
            grad_z {np.ndarray} -- Gradient array of shape (1,
                #_neurons_in_final_layer).

        Returns:
            {np.ndarray} -- Array containing gradient with repect to layer
                input, of shape (batch_size, input_dim).
        """
        # Initialise grad input
        grad_input = grad_z # starting from the back
        
        # Calculate gradient for layer and propagate back
        for layer in reversed(self._layers):
            grad_input = layer.backward(grad_input)

        return grad_input

    def update_params(self, learning_rate):
        """
        Performs one step of gradient descent with given learning rate on the
        parameters of all layers using currently stored gradients.

        Arguments:
            learning_rate {float} -- Learning rate of update step.
        """
        for layer in self._layers:
            layer.update_params(learning_rate)


def save_network(network, fpath):
    """
    Utility function to pickle `network` at file path `fpath`.
    """
    with open(fpath, "wb") as f:
        pickle.dump(network, f)


def load_network(fpath):
    """
    Utility function to load network found at file path `fpath`.
    """
    with open(fpath, "rb") as f:
        network = pickle.load(f)
    return network


class Trainer(object):
    """
    Trainer: Object that manages the training of a neural network.
    """

    def __init__(
        self,
        network,
        batch_size,
        nb_epoch,
        learning_rate,
        loss_fun,
        shuffle_flag,
    ):
        """Constructor.

        Arguments:
            network {MultiLayerNetwork} -- MultiLayerNetwork to be trained.
            batch_size {int} -- Training batch size.
            nb_epoch {int} -- Number of training epochs.
            learning_rate {float} -- SGD learning rate to be used in training.
            loss_fun {str} -- Loss function to be used. Possible values: mse,
                bce.
            shuffle_flag {bool} -- If True, training data is shuffled before
                training.
        """
        self.network = network
        self.batch_size = batch_size
        self.nb_epoch = nb_epoch
        self.learning_rate = learning_rate
        self.loss_fun = loss_fun
        self.shuffle_flag = shuffle_flag

        # Determine loss layer
        if self.loss_fun == 'MSE':
            self._loss_layer = MSELossLayer()
        else:
            self._loss_layer = CrossEntropyLossLayer()

    @staticmethod
    def shuffle(input_dataset, target_dataset):
        """
        Returns shuffled versions of the inputs.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_data_points, ).

        Returns: 2-tuple of np.ndarray: (shuffled inputs, shuffled_targets).
        """
        len_target = target_dataset.shape[0]
        all_indices = np.arange(len_target)

        np.random.shuffle(all_indices)

        return (input_dataset[all_indices], target_dataset[all_indices])

    def train(self, input_dataset, target_dataset):
        """
        Main training loop. Performs the following steps `nb_epoch` times:
            - Shuffles the input data (if `shuffle` is True)
            - Splits the dataset into batches of size `batch_size`.
            - For each batch:
                - Performs forward pass through the network given the current
                batch of inputs.
                - Computes loss.
                - Performs backward pass to compute gradients of loss with
                respect to parameters of network.
                - Performs one step of gradient descent on the network
                parameters.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_training_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_training_data_points, ).
        """
        # Train the data nb_epoch time
        for epoch in range(self.nb_epoch):
            # Data shuffled iff flag is True
            if self.shuffle_flag:
                input_dataset, target_dataset = Trainer.shuffle(input_dataset, target_dataset)

            ## Divide the data into batches
            num_batches = len(input_dataset) / self.batch_size

            # Divide into batches of size self.batch_size
            batch_inputs = np.split(input_dataset, num_batches)
            batch_targets = np.split(target_dataset, num_batches)

            ## Apply minibatch gradient descent for each batch (training)
            for index_batch in range(len(batch_targets)):
                current_batch_input = batch_inputs[index_batch]
                current_batch_target = batch_targets[index_batch]

                # Training (forward + backward + update weights)
                # Forward pass (getting both network output)
                y_pred = self.network.forward(current_batch_input) # (batch_size, # neurons in final layer)
                self._loss_layer.forward(y_pred, current_batch_target)

                # Backward pass (compute the gradient depending on the function)
                grad_z = self._loss_layer.backward()
                self.network.backward(grad_z)

                # Update the weight given parameter
                self.network.update_params(self.learning_rate)

            ## Finish training 1 epoch, continue to another epoch

    def eval_loss(self, input_dataset, target_dataset):
        """
        Function that evaluate the loss function for given data.

        Arguments:
            - input_dataset {np.ndarray} -- Array of input features, of shape
                (#_evaluation_data_points, n_features).
            - target_dataset {np.ndarray} -- Array of corresponding targets, of
                shape (#_evaluation_data_points, ).
        """
        # Using the input, we predict using our network
        y_pred = self.network.forward(input_dataset) # (#_evaluation_data_points, # neurons in final layer)

        # Calculate error
        validation_loss = self._loss_layer.forward(y_pred, target_dataset)
        
        return validation_loss


class Preprocessor(object):
    """
    Preprocessor: Object used to apply "preprocessing" operation to datasets.
    The object can also be used to revert the changes.
    """

    def __init__(self, data):
        """
        Initializes the Preprocessor according to the provided dataset.
        (Does not modify the dataset.)

        Arguments:
            - data {np.ndarray} dataset used to determined the parameters for
            the normalization.
        """
        # Need to calculate the min and max of the data.
        # as we have multiple features we need to calculate the min and max per column.
        # np.amin/max achieve this
        self.data_min = np.amin(data, axis=0)
        self.data_max = np.amax(data, axis=0)

    def apply(self, data):
        """
        Apply the pre-processing operations to the provided dataset.

        Arguments:
            - data {np.ndarray} dataset to be normalized.

        Returns:
            {np.ndarray} normalized dataset.
        """
        # Normalize such that minimum get value of 0 and maximum get value of 1
        normalised_data = (data-self.data_min) / (self.data_max-self.data_min)
        return normalised_data

    def revert(self, data):
        """
        Revert the pre-processing operations to retrieve the original dataset.

        Arguments:
            - data {np.ndarray} dataset for which to revert normalization.

        Returns:
            {np.ndarray} reverted dataset.
        """
        reverted_data = np.multiply(data, self.data_max-self.data_min) + self.data_min
        return reverted_data


def example_main():
    input_dim = 4
    neurons = [16, 32, 32, 3]
    activations = ["relu", "sigmoid", "identity", "identity"]
    net = MultiLayerNetwork(input_dim, neurons, activations)

    dat = np.loadtxt("iris.dat")
    np.random.shuffle(dat)

    # Split data into training and testing
    x = dat[:, :4]
    y = dat[:, 4:]

    split_idx = int(0.8 * len(x))

    x_train = x[:split_idx]
    y_train = y[:split_idx]

    x_val = x[split_idx:]
    y_val = y[split_idx:]

    # Apply normalisation
    prep_input = Preprocessor(x_train)
    x_train_pre = prep_input.apply(x_train)
    x_val_pre = prep_input.apply(x_val)

    # Begin training
    trainer = Trainer(
        network=net,
        batch_size=8,
        nb_epoch=1000,
        learning_rate=0.01,
        loss_fun="cross_entropy",
        shuffle_flag=True,
    )

    trainer.train(x_train_pre, y_train)
    print("Train loss = ", trainer.eval_loss(x_train_pre, y_train))
    print("Validation loss = ", trainer.eval_loss(x_val_pre, y_val))

    preds = net(x_val_pre).argmax(axis=1).squeeze()
    targets = y_val.argmax(axis=1).squeeze()
    accuracy = (preds == targets).mean()
    print("Validation accuracy: {}".format(accuracy))


if __name__ == "__main__":
    example_main()
