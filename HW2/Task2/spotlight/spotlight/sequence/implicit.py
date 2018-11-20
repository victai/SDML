"""
Models for recommending items given a sequence of previous items
a user has interacted with.
"""

import numpy as np

import torch

import torch.optim as optim

from spotlight.helpers import _repr_model
from spotlight.losses import (adaptive_hinge_loss,
                              bpr_loss,
                              hinge_loss,
                              pointwise_loss,
                              my_loss)
from spotlight.sequence.representations import (PADDING_IDX, CNNNet,
                                                LSTMNet,
                                                MixtureLSTMNet,
                                                PoolNet,
                                                GRUNet,
                                                MyLSTMNet)
from spotlight.sampling import sample_items
from spotlight.torch_utils import cpu, gpu, minibatch, set_seed, shuffle
from spotlight.evaluation import AveragePrecision, NewAveragePrecision

import time
import ipdb

class ImplicitSequenceModel(object):
    """
    Model for sequential recommendations using implicit feedback.

    Parameters
    ----------

    loss: string, optional
        The loss function for approximating a softmax with negative sampling.
        One of 'pointwise', 'bpr', 'hinge', 'adaptive_hinge', corresponding
        to losses from :class:`spotlight.losses`.
    representation: string or instance of :class:`spotlight.sequence.representations`, optional
        Sequence representation to use. If string, it must be one
        of 'pooling', 'cnn', 'lstm', 'mixture'; otherwise must be one of the
        representations from :class:`spotlight.sequence.representations`
    embedding_dim: int, optional
        Number of embedding dimensions to use for representing items.
        Overridden if representation is an instance of a representation class.
    n_iter: int, optional
        Number of iterations to run.
    batch_size: int, optional
        Minibatch size.
    l2: float, optional
        L2 loss penalty.
    learning_rate: float, optional
        Initial learning rate.
    optimizer_func: function, optional
        Function that takes in module parameters as the first argument and
        returns an instance of a PyTorch optimizer. Overrides l2 and learning
        rate if supplied. If no optimizer supplied, then use ADAM by default.
    use_cuda: boolean, optional
        Run the model on a GPU.
    sparse: boolean, optional
        Use sparse gradients for embedding layers.
    random_state: instance of numpy.random.RandomState, optional
        Random state to use when fitting.
    num_negative_samples: int, optional
        Number of negative samples to generate for adaptive hinge loss.

    Notes
    -----

    During fitting, the model computes the loss for each timestep of the
    supplied sequence. For example, suppose the following sequences are
    passed to the ``fit`` function:

    .. code-block:: python

       [[1, 2, 3, 4, 5],
        [0, 0, 7, 1, 4]]


    In this case, the loss for the first example will be the mean loss
    of trying to predict ``2`` from ``[1]``, ``3`` from ``[1, 2]``,
    ``4`` from ``[1, 2, 3]`` and so on. This means that explicit padding
    of all subsequences is not necessary (although it is possible by using
    the ``step_size`` parameter of
    :func:`spotlight.interactions.Interactions.to_sequence`.
    """

    def __init__(self,
                 loss='pointwise',
                 representation='pooling',
                 embedding_dim=32,
                 n_iter=10,
                 batch_size=256,
                 l2=0.0,
                 learning_rate=1e-2,
                 optimizer_func=None,
                 use_cuda=False,
                 sparse=False,
                 random_state=None,
                 num_negative_samples=5,
                 test_data=0,
                 neg_prob=0):

        assert loss in ('pointwise',
                        'bpr',
                        'hinge',
                        'adaptive_hinge',
                        'mine')

        if isinstance(representation, str):
            assert representation in ('pooling',
                                      'cnn',
                                      'lstm',
                                      'mixture',
                                      'gru',
                                      'mylstm')

        self._loss = loss
        self._representation = representation
        self._embedding_dim = embedding_dim
        self._n_iter = n_iter
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self._l2 = l2
        self._use_cuda = use_cuda
        self._sparse = sparse
        self._optimizer_func = optimizer_func
        self._random_state = random_state or np.random.RandomState()
        self._num_negative_samples = num_negative_samples

        self._num_items = None
        self._net = None
        self._optimizer = None
        self._loss_func = None

        set_seed(self._random_state.randint(-10**8, 10**8),
                 cuda=self._use_cuda)
        self.test_data = test_data
        self.neg_prob = neg_prob

    def __repr__(self):

        return _repr_model(self)

    @property
    def _initialized(self):
        return self._net is not None

    def _initialize(self, interactions):

        self._num_items = interactions.num_items

        if self._representation == 'pooling':
            self._net = PoolNet(self._num_items,
                                self._embedding_dim,
                                sparse=self._sparse)
        elif self._representation == 'cnn':
            self._net = CNNNet(self._num_items,
                               self._embedding_dim,
                               sparse=self._sparse)
        elif self._representation == 'lstm':
            self._net = LSTMNet(self._num_items,
                                self._embedding_dim,
                                sparse=self._sparse)
        elif self._representation == 'mixture':
            self._net = MixtureLSTMNet(self._num_items,
                                       self._embedding_dim,
                                       sparse=self._sparse)
        elif self._representation == 'gru':
            self._net = GRUNet(self._num_items,
                                self._embedding_dim,
                                sparse=self._sparse)
        elif self._representation == 'mylstm':
            self._net = MyLSTMNet(self._num_items,
                                self._embedding_dim,
                                sparse=self._sparse)
        else:
            self._net = self._representation

        self._net = gpu(self._net, self._use_cuda)

        if self._optimizer_func is None:
            self._optimizer = optim.Adam(
                self._net.parameters(),
                weight_decay=self._l2,
                lr=self._learning_rate
            )
        else:
            self._optimizer = self._optimizer_func(self._net.parameters())

        if self._loss == 'pointwise':
            self._loss_func = pointwise_loss
        elif self._loss == 'bpr':
            self._loss_func = bpr_loss
        elif self._loss == 'hinge':
            self._loss_func = hinge_loss
        elif self._loss == 'mine':
            self._loss_func = my_loss
        else:
            self._loss_func = adaptive_hinge_loss

    def _check_input(self, item_ids):

        if isinstance(item_ids, int):
            item_id_max = item_ids
        else:
            item_id_max = item_ids.max()

        if item_id_max >= self._num_items:
            raise ValueError('Maximum item id greater '
                             'than number of items in model.')

    def fit(self, interactions, verbose=False, calc_map=False, neg_mode="original"):
        """
        Fit the model.

        When called repeatedly, model fitting will resume from
        the point at which training stopped in the previous fit
        call.

        Parameters
        ----------

        interactions: :class:`spotlight.interactions.SequenceInteractions`
            The input sequence dataset.
        """

        sequences = interactions.sequences.astype(np.int64)
        user_ids = interactions.user_ids

        if not self._initialized:
            self._initialize(interactions)

        self._check_input(sequences)
        best_map = 0
        np.random.seed(0)
        random_state = np.random.RandomState()
        for epoch_num in range(self._n_iter):

            start_time = time.time()
            shuffle_indices = np.arange(sequences.shape[0])
            random_state.shuffle(shuffle_indices)
            sequences = sequences[shuffle_indices]
            user_ids = user_ids[shuffle_indices]

            #sequences = shuffle(sequences,
            #                    random_state=self._random_state)


            sequences_tensor = gpu(torch.from_numpy(sequences),
                                   self._use_cuda)

            epoch_loss = 0.0

            #for minibatch_num, batch_sequence in enumerate(minibatch(sequences_tensor,
            #                                                         batch_size=self._batch_size)):
            for minibatch_num in range(sequences.shape[0] // self._batch_size):
                sequence_var = sequences_tensor[minibatch_num * self._batch_size: (minibatch_num + 1) * self._batch_size]
                mini_user_ids = user_ids[minibatch_num * self._batch_size: (minibatch_num + 1) * self._batch_size]

                #sequence_var = batch_sequence

                user_representation, _ = self._net.user_representation(
                    sequence_var
                )

                positive_prediction = self._net(user_representation,
                                                sequence_var)

                if neg_mode == "original":
                    if self._loss == 'adaptive_hinge':
                        negative_prediction = self._get_multiple_negative_predictions(
                            sequence_var.size(),
                            user_representation,
                            n=self._num_negative_samples)
                    else:
                        negative_prediction = self._get_negative_prediction(sequence_var.size(),
                                                                            user_representation)
                elif neg_mode == "mine":
                    '''
                    example:
                        sequence = [1,4,2,48,999,8], userid = 0
                        use neg_prob of user 0 to pick a food 'x'
                        replace the 8 with 'x'
                        negative sequence = [1,4,2,48,999,x]
                    '''
                    if self._loss == 'adaptive_hinge' or self._loss == 'mine':
                        negative_prediction = self.my_get_multiple_negative_predictions2(
                            sequence_var.cpu().data,
                            mini_user_ids,
                            user_representation,
                            n=self._num_negative_samples)
                        #tmp = np.zeros((self._batch_size * self._num_negative_samples, negative_prediction.size(1)))
                        #tmp[:, -1] = 1
                        #tmp = gpu(torch.from_numpy(tmp).float(), self._use_cuda)
                        #negative_prediction *= tmp
                    else:
                        negative_prediction = self.my_get_negative_prediction2(sequence_var.cpu().data, mini_user_ids,
                                                                            user_representation)
                        #tmp = np.zeros((self._batch_size, negative_prediction.size(1)))
                        #tmp[:, -1] = 1
                        #tmp = gpu(torch.from_numpy(tmp).float(), self._use_cuda)
                        #negative_prediction *= tmp

                self._optimizer.zero_grad()

                #loss = self._loss_func(positive_prediction,
                #                       negative_prediction,
                #                       mask=(sequence_var != PADDING_IDX))
                loss = torch.nn.CrossEntropyLoss()(positive_prediction, sequence_var[:, -1])
                epoch_loss += loss.item()

                loss.backward()

                self._optimizer.step()

            epoch_loss /= minibatch_num + 1

            if calc_map: # takes too much time, about 20 mins, so set as an option
                print("calculating map")
                self._net.train(False)
                map_start = time.time()
                ap = NewAveragePrecision(self, self.test_data, 20)
                print("calculate map time", time.time() - map_start)
                self._net.train(True)
            else:
                ap = np.zeros(10)

            if verbose:
                print('Epoch {}: loss {} map: {}; time: {}'.format(epoch_num, epoch_loss, ap.mean(), time.time() - start_time))

            if np.isnan(epoch_loss) or epoch_loss == 0.0:
                raise ValueError('Degenerate epoch loss: {}'
                                 .format(epoch_loss))

    def _get_negative_prediction(self, shape, user_representation):

        negative_items = sample_items(
            self._num_items,
            shape,
            random_state=self._random_state)
        negative_var = gpu(torch.from_numpy(negative_items), self._use_cuda)

        negative_prediction = self._net(user_representation, negative_var)

        return negative_prediction

    def _get_multiple_negative_predictions(self, shape, user_representation,
                                           n=5):
        batch_size, sliding_window = shape
        size = (n,) + (1,) * (user_representation.dim() - 1)
        negative_prediction = self._get_negative_prediction(
            (n * batch_size, sliding_window),
            user_representation.repeat(*size))

        return negative_prediction.view(n, batch_size, sliding_window)

    def my_get_negative_prediction2(self, sequence_var, mini_user_ids, user_representation):
        sampled_neg = np.zeros((mini_user_ids.shape[0], sequence_var.shape[1]))
        for i, u in enumerate(mini_user_ids):
            sampled_neg[i] = np.random.choice(self.neg_prob.shape[1], sequence_var.shape[1], p=self.neg_prob[u])
        sampled_neg = torch.from_numpy(sampled_neg).long()
        #ipdb.set_trace()
        #negative_var = sequence_var.clone()
        #negative_var[:, -1] = sampled_neg
        negative_var = gpu(sampled_neg, self._use_cuda)

        negative_prediction = self._net(user_representation, negative_var)

        return negative_prediction

    def my_get_multiple_negative_predictions2(self, sequence_var, mini_user_ids, user_representation,
                                           n=5):
        sampled_neg = np.zeros((n, mini_user_ids.shape[0], sequence_var.shape[1]))
        for _n in range(n):
            for i, u in enumerate(mini_user_ids):
                sampled_neg[_n, i] = np.random.choice(self.neg_prob.shape[1], sequence_var.shape[1], p=self.neg_prob[u])
        sampled_neg = torch.from_numpy(sampled_neg.reshape(-1, sequence_var.shape[1])).long()
        #sampled_neg = torch.from_numpy(sampled_neg.flatten())
        #negative_var = sequence_var.repeat((n, 1))
        #negative_var[:, -1] = sampled_neg
        negative_var = gpu(sampled_neg, self._use_cuda)
        if self._loss == "mine":
            negative_prediction = self._net(user_representation.repeat((n, 1)), negative_var)
        else:
            negative_prediction = self._net(user_representation.repeat((n, 1, 1)), negative_var)

        #batch_size, sliding_window = shape
        #size = (n,) + (1,) * (user_representation.dim() - 1)
        #negative_prediction = self._get_negative_prediction(
        #    (n * batch_size, sliding_window),
        #    user_representation.repeat(*size))

        return negative_prediction

    def my_get_negative_prediction(self, sequence_var, mini_user_ids, user_representation):
        sampled_neg = np.zeros(mini_user_ids.shape[0])
        for i, u in enumerate(mini_user_ids):
            sampled_neg[i] = np.random.choice(self.neg_prob.shape[1], 1, p=self.neg_prob[u])
        sampled_neg = torch.from_numpy(sampled_neg)
        negative_var = sequence_var.clone()
        negative_var[:, -1] = sampled_neg
        negative_var = gpu(negative_var, self._use_cuda)

        negative_prediction = self._net(user_representation, negative_var)

        return negative_prediction

    def my_get_multiple_negative_predictions(self, sequence_var, mini_user_ids, user_representation,
                                           n=5):
        sampled_neg = np.zeros((n, mini_user_ids.shape[0]))
        for i, u in enumerate(mini_user_ids):
            sampled_neg[:, i] = np.random.choice(self.neg_prob.shape[1], n, p=self.neg_prob[u])
        sampled_neg = torch.from_numpy(sampled_neg.flatten())
        negative_var = sequence_var.repeat((n, 1))
        negative_var[:, -1] = sampled_neg
        negative_var = gpu(negative_var, self._use_cuda)
        if self._loss == "mine":
            negative_prediction = self._net(user_representation.repeat((n, 1)), negative_var)
        else:
            negative_prediction = self._net(user_representation.repeat((n, 1, 1)), negative_var)

        #batch_size, sliding_window = shape
        #size = (n,) + (1,) * (user_representation.dim() - 1)
        #negative_prediction = self._get_negative_prediction(
        #    (n * batch_size, sliding_window),
        #    user_representation.repeat(*size))

        return negative_prediction

    def predict(self, sequences, item_ids=None):
        """
        Make predictions: given a sequence of interactions, predict
        the next item in the sequence.

        Parameters
        ----------

        sequences: array, (1 x max_sequence_length)
            Array containing the indices of the items in the sequence.
        item_ids: array (num_items x 1), optional
            Array containing the item ids for which prediction scores
            are desired. If not supplied, predictions for all items
            will be computed.

        Returns
        -------

        predictions: array
            Predicted scores for all items in item_ids.
        """

        self._net.train(False)

        if self._representation != "mylstm":

            sequences = np.atleast_2d(sequences)

            if item_ids is None:
                item_ids = np.arange(self._num_items).reshape(-1, 1)

            self._check_input(item_ids)
            self._check_input(sequences)

            sequences = torch.from_numpy(sequences.astype(np.int64).reshape(1, -1))
            item_ids = torch.from_numpy(item_ids.astype(np.int64))
            
            sequence_var = gpu(sequences, self._use_cuda)
            item_var = gpu(item_ids, self._use_cuda)

            _, sequence_representations = self._net.user_representation(sequence_var)
            size = (len(item_var),) + sequence_representations.size()[1:]
            out = self._net(sequence_representations.expand(*size).unsqueeze(2),
                            item_var)

            return cpu(out).detach().numpy().flatten()

        else:
            sequences = torch.from_numpy(sequences.astype(np.int64))
            sequence_var = gpu(sequences, self._use_cuda)
            sequence_representations, _ = self._net.user_representation(sequence_var)
            #print("s",sequence_representations.size())
            out = self._net(sequence_representations, 0)
            #print("o",out.size())
            return cpu(out).detach().numpy()
