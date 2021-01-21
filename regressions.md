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

- Given mean squared error, need to get the slope of MSE with respect to b and with respsect to m

- Can run a certain number of iterations with a set learning rate - can adjust learning rate or standardize the data to get better results

Much faster version with way less code but harder to understand with TensorFlow to dramatically simplify code

- need to know matrix multiplication

  - are two matrices eligible to be multiplied together? Given tensor with shape: [4,2] ([Row, Column]) and tensor with shape: [2,3] -> inner shape values are the same i.e. 2 === 2 so it's eligible; if different inner shape values, multiplication not allowed i.e. [4,3] [2,3]; order matters

  - what's the output of matrix multiplication? given [4,2] x [2,3] -> [4,3] (the outer shape values make up the new matrix shape)

  - how is matrix multiplication done? -> left to right in the left matrix, top to bottom multiplying and adding for top matrix - summing products across

  - matrix form of slope equation

    - slope of MSE with respect to M and B at the same time => features x ((features x weights) - Labels) / n

    - labels (tensor), features (tensor), n (number of observations), weights (m and b in a tensor)

    - can multiply horsepower (add arbitrary column of 1s trick so we can multiply properly - computing m x xi + b) by weights - output matrix of mx + b values -> same as the guesses for MPG

    - need to subtract actual mpg of cars to get differences matrix i.e. mxi + b - actuali

    - going to multiply engine horsepower with column of 1s by the differences by using the transpose of the matrix on the left i.e. [6,2] -> [2,6] x [6,1] -> [2,1] tensor => transpose of feature matrix x differences matrix => [2,1] with slope values for m and b

      - in first row calculated the slope of MSE with respect to M (x1 x difference1 + x2 x difference2 ...) => M SLOPE

      - in second row calculated the slope of MSE with respect to b (d1 + d2 + ... + dn) => B SLOPE

  - after calculating slopes of M and B, we update M and B with as well

- Refactor steps

  - Turn 'features' and 'label's into tensors

  - Append a column of ones to the feature tensor

  - Make tensor for weights

  - Refactor gradient descent to use new equation

  - vectorized solution with matrices

- Evaluating accuracy by

  - train model with training data

  - use 'test' data to make predictions about observations with known labels

  - gauge accuracy by calculating 'coefficient of determination' -> R^2 value = 1 - SSres/SStot; how good our fit is with the actual data

    - sum of squares residual (summation i to n)(actual - predicted)^2; find difference from predicted trendline and the actual (label - predicted)

    - sum of squares total (summation i to n)(actual - average)^2; baseline accuracy value (label - predicted)

    - negative values mean we're far off from being accurate (residual sum of squares is larger than the total, means it's way worse than using the mean/average value of the dataset rather than fancy analysis)
