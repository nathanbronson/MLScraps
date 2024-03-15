<p align="center"><img src="https://github.com/nathanbronson/MLScraps/blob/main/logo.jpg?raw=true" alt="logo" width="200"/></p>

_____
# MLScraps
a collection of small machine learning models and utilities

## Modules

### EmbeddingRegression <img align="right" src="https://github.com/nathanbronson/MLScraps/blob/main/EmbeddingRegression/logo.jpg?raw=true" alt="logo" width="65"/>
Regression trained through gradient descent using trained embeddings to incorporate discrete variables.

### MaclaurinRegression <img align="right" src="https://github.com/nathanbronson/MLScraps/blob/main/MaclaurinRegression/logo.jpg?raw=true" alt="logo" width="65"/>
Regression modeling arbitrary function f as a Maclaurin series of a given degree with the form $$\sum_{n=0}^{degree}\frac{f^{(n)}}{n!}x^n$$ for f of a single variable and a generalization of $$f(x,y) + (f_x(x, y)x + f_y(x, y)y) + \frac{1}{2!}(f_{xx}(x, y)x^2 + f_{xy}(x, y)xy + f_{yy}(x, y)y^2) + ...$$ for f of multiple variables. Contains models that estimate partial derivatives of f through gradient descent and OLS. 

### DecisionSurface <img align="right" src="https://github.com/nathanbronson/MLScraps/blob/main/DecisionSurface/logo.jpg?raw=true" alt="logo" width="65"/>
Utility to plot decision surface of `scikit-learn` classifier.

## License
See `LICENSE`.
