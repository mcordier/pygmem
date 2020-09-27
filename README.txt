# Generalized (Non-Linear) Mixed Effect Models as a Machine Learning problem (no inference of the variance).

The library is written with pytorch, and is only working with only one type of group (individualized prediction regarding one type of group). 

It's designed for a usecase with large dataset (Netflix, LinkedIn), compared to a more statistical regression approach.

There is an option for non-linear cases, with a possible input-encoder (for images), and also a decoder, and the mixed-effect model would be done on linearized intermediary space

You also choose the type of regression you choose to do (Gaussian, Poisson, Binary prediction) by choosing an ouput function theta.
