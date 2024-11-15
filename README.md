# CoolSearch ❄️

A package for optimizing black-box functions.

- Work in progress

## Features

- Grid-search
- Random-search
- Saving previous samples
- Runtime estimation
- Polynomial regression model

## Usage

The main entry point is the `CoolSearch` object, that can be instantiated from

- Class constructor `CoolSearch(objective, parameters, ...)`.
  - This is useful to optimize a function
- Model validation? `CoolSearch.model_validate(model, parameters, ...)`
  - This is useful if we have a model fulfilling some assumptions.

### What is needed

- `objective` is a function `y_1,...y_m = f(x_1,...x_n)`
- `parameters` is a mapping from _parameter names_ to _values_, _ranges_ or _options_.

## Todo

- smaller range grid search
- Gaussian process model

### ideas

- genetic search

## note

- (note to self) package should be installable as `pip install -e .`
