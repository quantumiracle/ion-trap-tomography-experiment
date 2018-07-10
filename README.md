# Ion-trap Tomography with experiment data
tomo_expr.py: general API for tomography\
data.pkl: experiment data

choose one from two:\
(1).Traditional methods
* curve_fit_class.py: class of traditional curve fitting methods for estimation of Rabi oscillation curve parameters 

(2).Neural Networks methods
* data_pick.py: generate data from theory for training the neural network predictor
* NNtrainer.py: trianing to get weights & bias for NNpredictor
* NNpredictor.py: the class of neural network predictor


choose one from two:\
(1).[Maximum Likelihood methods](https://arxiv.org/abs/1605.05039)
* measure.py: class of tomography functions

(2).[Iteration methods](https://arxiv.org/abs/quant-ph/0009093)
* ite.py: using  instead of in measure.py for tomography
