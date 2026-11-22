.. _recommender:

Recommender algorithms module
##############################

Recommender system algorithms and utilities.

Cornac utilities
******************************
.. autoapimodule:: recommenders.models.cornac.cornac_utils
    :members:


DeepRec utilities
******************************
Base model
==============================
.. autoapimodule:: recommenders.models.deeprec.models.base_model
    :members:
    :special-members: __init__

Sequential base model
==============================
.. autoapimodule:: recommenders.models.deeprec.models.sequential.sequential_base_model
    :members:
    :special-members: __init__

Iterators
==============================
.. autoapimodule:: recommenders.models.deeprec.io.iterator
    :members:
    :special-members: __init__
.. autoapimodule:: recommenders.models.deeprec.io.dkn_iterator
    :members:
    :special-members: __init__
.. autoapimodule:: recommenders.models.deeprec.io.dkn_item2item_iterator
    :members:
    :special-members: __init__
.. autoapimodule:: recommenders.models.deeprec.io.nextitnet_iterator
    :members:
    :special-members: __init__
.. autoapimodule:: recommenders.models.deeprec.io.sequential_iterator
    :members:
    :special-members: __init__

Data processing utilities
==============================
.. autoapimodule:: recommenders.models.deeprec.DataModel.ImplicitCF
    :members:
    :special-members: __init__

Utilities
==============================
.. autoapimodule:: recommenders.models.deeprec.deeprec_utils
    :members:
    :special-members: __init__, __repr__


DKN
******************************
.. autoapimodule:: recommenders.models.deeprec.models.dkn
    :members:
    :special-members: __init__


DKN item-to-item
******************************
.. autoapimodule:: recommenders.models.deeprec.models.dkn_item2item
    :members:
    :special-members: __init__


xDeepFM
******************************
.. autoapimodule:: recommenders.models.deeprec.models.xDeepFM
    :members:
    :special-members: __init__


LightGCN
******************************
.. autoapimodule:: recommenders.models.deeprec.models.graphrec.lightgcn
    :members:
    :special-members: __init__


A2SVD
******************************
.. autoapimodule:: recommenders.models.deeprec.models.sequential.asvd
    :members:
    :special-members: __init__


Caser
******************************
.. autoapimodule:: recommenders.models.deeprec.models.sequential.caser
    :members:
    :special-members: __init__


GRU
******************************
.. autoapimodule:: recommenders.models.deeprec.models.sequential.gru
    :members:
    :special-members: __init__


NextItNet
******************************
.. autoapimodule:: recommenders.models.deeprec.models.sequential.nextitnet
    :members:
    :special-members: __init__


RNN Cells
******************************
.. autoapimodule:: recommenders.models.deeprec.models.sequential.rnn_cell_implement
    :members:
    :special-members: __init__


SUM
******************************
.. autoapimodule:: recommenders.models.deeprec.models.sequential.sum
    :members:
    :special-members: __init__
.. autoapimodule:: recommenders.models.deeprec.models.sequential.sum_cells
    :members:
    :special-members: __init__


SLIRec
******************************
.. autoapimodule:: recommenders.models.deeprec.models.sequential.sli_rec
    :members:
    :special-members: __init__


FastAI utilities
******************************
.. autoapimodule:: recommenders.models.fastai.fastai_utils
    :members:


LightFM utilities
******************************
.. autoapimodule:: recommenders.models.lightfm.lightfm_utils
    :members:


LightGBM utilities
******************************
.. autoapimodule:: recommenders.models.lightgbm.lightgbm_utils
    :members:


NCF
******************************
.. autoapimodule:: recommenders.models.ncf.dataset
    :members:
    :special-members: __init__
.. autoapimodule:: recommenders.models.ncf.ncf_singlenode
    :members:
    :special-members: __init__


NewsRec utilities
******************************
Base model
==============================
.. autoapimodule:: recommenders.models.newsrec.models.base_model
    :members:
    :special-members: __init__

Iterators
==============================
.. autoapimodule:: recommenders.models.newsrec.io.mind_iterator
    :members:
    :special-members: __init__
.. autoapimodule:: recommenders.models.newsrec.io.mind_all_iterator
    :members:
    :special-members: __init__


Utilities
==============================
.. autoapimodule:: recommenders.models.newsrec.models.layers
    :members:
    :special-members: __init__
.. autoapimodule:: recommenders.models.newsrec.newsrec_utils
    :members:
    :special-members: __init__


LSTUR
******************************
.. autoapimodule:: recommenders.models.newsrec.models.lstur
    :members:
    :special-members: __init__


NAML
******************************
.. autoapimodule:: recommenders.models.newsrec.models.naml
    :members:
    :special-members: __init__


NPA
******************************
.. autoapimodule:: recommenders.models.newsrec.models.npa
    :members:
    :special-members: __init__


NRMS
******************************
.. autoapimodule:: recommenders.models.newsrec.models.nrms
    :members:
    :special-members: __init__


RBM
******************************
.. autoapimodule:: recommenders.models.rbm.rbm
    :members:
    :special-members: __init__

.. FIXME: Fix Pymanopt dependency. Issue #2038
.. GeoIMC
.. ******************************
.. .. autoapimodule:: recommenders.models.geoimc.geoimc_algorithm
..     :members:
..     :special-members: __init__
.. .. autoapimodule:: recommenders.models.geoimc.geoimc_data
..     :members:
..     :special-members: __init__
.. .. autoapimodule:: recommenders.models.geoimc.geoimc_predict
..     :members:
.. .. autoapimodule:: recommenders.models.geoimc.geoimc_utils
..     :members:


.. FIXME: Fix Pymanopt dependency. Issue #2038
.. RLRMC
.. ******************************
.. .. autoapimodule:: recommenders.models.rlrmc.RLRMCdataset
..     :members:
..     :special-members: __init__
.. .. autoapimodule:: recommenders.models.rlrmc.RLRMCalgorithm
..     :members:
..     :special-members: __init__
.. .. autoapimodule:: recommenders.models.rlrmc.conjugate_gradient_ms
..     :members:
..     :special-members: __init__



SAR
******************************
.. autoapimodule:: recommenders.models.sar.sar_singlenode
    :members:
    :special-members: __init__


SASRec 
******************************
.. autoapimodule:: recommenders.models.sasrec.model
    :members:
    :special-members: __init__
.. autoapimodule:: recommenders.models.sasrec.sampler
    :members:
    :special-members: __init__
.. autoapimodule:: recommenders.models.sasrec.util
    :members:


SSE-PT 
******************************
.. autoapimodule:: recommenders.models.sasrec.ssept
    :members:
    :special-members: __init__


Surprise utilities
******************************
.. autoapimodule:: recommenders.models.surprise.surprise_utils
    :members:


TF-IDF utilities
******************************
.. autoapimodule:: recommenders.models.tfidf.tfidf_utils
    :members:


Standard VAE
******************************
.. autoapimodule:: recommenders.models.vae.standard_vae
    :members:
    :special-members: __init__


Multinomial VAE
******************************
.. autoapimodule:: recommenders.models.vae.multinomial_vae
    :members:
    :special-members: __init__


Vowpal Wabbit utilities
******************************
.. autoapimodule:: recommenders.models.vowpal_wabbit.vw
    :members:


Wide & Deep
******************************
.. autoapimodule:: recommenders.models.wide_deep.wide_deep_utils
    :members:
    :special-members: __init__