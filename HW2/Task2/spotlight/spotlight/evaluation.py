import numpy as np

import scipy.stats as st

import ipdb


FLOAT_MAX = np.finfo(np.float32).max


def mrr_score(model, test, train=None):
    """
    Compute mean reciprocal rank (MRR) scores. One score
    is given for every user with interactions in the test
    set, representing the mean reciprocal rank of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will be set to very low values and so not
        affect the MRR.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each user in test.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    mrrs = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            predictions[train[user_id].indices] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[row.indices]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)


def sequence_mrr_score(model, test, exclude_preceding=False):
    """
    Compute mean reciprocal rank (MRR) scores. Each sequence
    in test is split into two parts: the first part, containing
    all but the last elements, is used to predict the last element.

    The reciprocal rank of the last element is returned for each
    sequence.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.SequenceInteractions`
        Test interactions.
    exclude_preceding: boolean, optional
        When true, items already present in the sequence will
        be excluded from evaluation.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each sequence in test.
    """

    sequences = test.sequences[:, :-1]
    targets = test.sequences[:, -1:]

    mrrs = []
    
    for i in range(len(sequences)):

        predictions = -model.predict(sequences[i])

        if exclude_preceding:
            predictions[sequences[i]] = FLOAT_MAX

        mrr = (1.0 / st.rankdata(predictions)[targets[i]]).mean()

        mrrs.append(mrr)

    return np.array(mrrs)

def AveragePrecision(model, test, k=20):
    sequences = test.sequences[:, :-k]
    targets = test.sequences[:, -k:]
    score = 0
    hit = 0
    scores = []
    for i in range(len(sequences)):
        predictions = -model.predict(sequences[i])
        predictions = predictions.argsort()[:k]
        
        score = 0
        hit = 0
        for j, p in enumerate(predictions):
            if p in targets[i]:
                hit += 1
                score += hit / (j+1)
        score /= 20
        scores.append(score)
    
    return np.array(scores)

def NewAveragePrecision(model, test, k=20):
    sequences = test.sequences[:, :-1]
    targets = test.sequences[:, -1]
    user_ids = test.user_ids

    consecutive_idx = np.where((np.diff(sequences, axis=0) == np.zeros(sequences.shape[1])).all(axis=1) == 0)[0]+1

    consecutive_idx = np.hstack((0, consecutive_idx))
    day_targets = []
    for i, j in enumerate(consecutive_idx):
        if i == len(consecutive_idx) - 1:
            day_targets.append(targets[consecutive_idx[i]:])
        else:
            day_targets.append(targets[consecutive_idx[i]: consecutive_idx[i+1]])

    score = 0
    hit = 0
    scores = []
    import time
    #all_pred = []
    #for i, idx in enumerate(consecutive_idx):
    predictions = -model.predict(sequences[consecutive_idx])
    #all_pred.append(predictions)
    #all_pred = np.array(all_pred)
    #print(all_pred.shape)
    predictions = predictions.argsort(axis=1)[:,:20]
    for i, idx in enumerate(consecutive_idx):
        score = 0
        hit = 0
        for j, p in enumerate(predictions[i]):
            if p in day_targets[i]:
                hit += 1
                score += hit / (j+1)
        score /= len(day_targets[i])
        scores.append(score)
    
    return np.array(scores)


def sequence_precision_recall_score(model, test, k=10, exclude_preceding=False):
    """
    Compute sequence precision and recall scores. Each sequence
    in test is split into two parts: the first part, containing
    all but the last k elements, is used to predict the last k
    elements.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.SequenceInteractions`
        Test interactions.
    exclude_preceding: boolean, optional
        When true, items already present in the sequence will
        be excluded from evaluation.

    Returns
    -------

    mrr scores: numpy array of shape (num_users,)
        Array of MRR scores for each sequence in test.
    """
    sequences = test.sequences[:, :-k]
    targets = test.sequences[:, -k:]
    precision_recalls = []
    for i in range(len(sequences)):
        predictions = -model.predict(sequences[i])
        if exclude_preceding:
            predictions[sequences[i]] = FLOAT_MAX

        predictions = predictions.argsort()[:k]
        precision_recall = _get_precision_recall(predictions, targets[i], k)
        precision_recalls.append(precision_recall)

    precision = np.array(precision_recalls)[:, 0]
    recall = np.array(precision_recalls)[:, 1]
    return precision, recall


def _get_precision_recall(predictions, targets, k):

    predictions = predictions[:k]
    num_hit = len(set(predictions).intersection(set(targets)))

    return float(num_hit) / len(predictions), float(num_hit) / len(targets)


def precision_recall_score(model, test, train=None, k=10):
    """
    Compute Precision@k and Recall@k scores. One score
    is given for every user with interactions in the test
    set, representing the Precision@k and Recall@k of all their
    test items.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.
    train: :class:`spotlight.interactions.Interactions`, optional
        Train interactions. If supplied, scores of known
        interactions will not affect the computed metrics.
    k: int or array of int,
        The maximum number of predicted items
    Returns
    -------

    (Precision@k, Recall@k): numpy array of shape (num_users, len(k))
        A tuple of Precisions@k and Recalls@k for each user in test.
        If k is a scalar, will return a tuple of vectors. If k is an
        array, will return a tuple of arrays, where each row corresponds
        to a user and each column corresponds to a value of k.
    """

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    if np.isscalar(k):
        k = np.array([k])

    precision = []
    recall = []

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = FLOAT_MAX

        predictions = predictions.argsort()

        targets = row.indices

        user_precision, user_recall = zip(*[
            _get_precision_recall(predictions, targets, x)
            for x in k
        ])

        precision.append(user_precision)
        recall.append(user_recall)

    precision = np.array(precision).squeeze()
    recall = np.array(recall).squeeze()

    return precision, recall


def rmse_score(model, test):
    """
    Compute RMSE score for test interactions.

    Parameters
    ----------

    model: fitted instance of a recommender model
        The model to evaluate.
    test: :class:`spotlight.interactions.Interactions`
        Test interactions.

    Returns
    -------

    rmse_score: float
        The RMSE score.
    """

    predictions = model.predict(test.user_ids, test.item_ids)

    return np.sqrt(((test.ratings - predictions) ** 2).mean())
