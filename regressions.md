# Regressions

Miles per gallon = m \* (car horsepower) + b

Run one iteration of GD and update 'm' and 'b'

We'll run gradient descent until we get good values for 'm' and 'b'

Use a 'test' data set to evaluate the accuracy of our calculated 'm' and 'b'

Make a prediction using our calculated 'm' and 'b'

```js
class LinearRegression {
  gradientDescent()
  train()
  test()
  predict()
}
```

## How to implement Gradient Descent?

Working but slow solution uses plain arrays of data and plain for loops - easier to understand

Given mean squared error, need to get the slope of MSE with respect to b and with respsect to m

Can run a certain number of iterations with a set learning rate - can adjust learning rate or standardize the data to get better results

Much faster version with way less code but harder to understand with TensorFlow to dramatically simplify code
