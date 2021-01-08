# Tensorflow

js.tensorflow.org

Helps with dealing with arrays of arrays or matrices; similar API to lodash; fast numeric calculations; low level linear algebra API and higher level API for ML; similar API to numpy in Python

`Tensors` aka the objects we're using

`Dimensions` aka 1D (array), 2D (array of arrays - [ [...,...,..], [...]]), 3D

`Shape` - how many records in each dimension?

- Imagine calling .length once on each dimension from the outside in

- [5, 10, 17].length -> 3 records in 1D Tensor -> [3]

- [ [5,10,17], [18,4,2].length ].length -> [2,3]

- [ [ [5,10,17].length ].length ].length -> [1,1,3]

- 2D is most important dimension to work with [#rows, #columns]

## Using Tensorflow

Element-wise operations - shapes must match most of the time unless doing broadcasting!

```js
const data = tf.tensor([1, 2, 3]);
data.shape; // [3]
const otherData = tf.tensor([4, 5, 6]);
data.add(otherData); // [5,7,9]
// data/otherData still contain their original values as operations return a brand new tensor

data.sub(otherData); // [-3,-3,-3]

data.mul(otherData); // [4,10,18]

data.div(otherData); // [0.25,0.4,0.5]
```

```js
// 2D example
const data = tf.tensor([
  [1, 2, 3],
  [4, 5, 6],
]);
const otherData = tf.tensor([
  [4, 5, 6],
  [1, 2, 3],
]);

data.add(otherData); // [[5,7,9],[5,7,9]]
```

Broadcasting works when you take the shape of both tensors and from right to left, the shapes are equal or one is '1'

```
  [3] => works
  [1]

  [2,3,2] => doesn't work
    [2,1]

  [2,3,2] => works
    [3,1]
```

```js
// Broadcasting example where shapes are different
const data = tf.tensor([1, 2, 3]); // Shape [3]
const otherData = tf.tensor([4]); // Shape [1]

data.add(otherData); // [5,6,7]
```

Logging data with tensorflow through `data.print()`

```js
// Accessors
const data = tf.tensor([10, 20, 30]);

data.get(0); // 10
// Can't do data[0]

const twoDimData = tf.tensor([
  [10, 20, 30],
  [40, 50, 60],
]);

data.get(0, 0); // 10
// data.get(row, column) for 2D

// no data.set(...); need to create a new tensor to modify instead
```

```js
// Accessing columns much easier than lodash
const data = tf.tensor([
  [10, 20, 30],
  [40, 50, 60],
  [10, 20, 30],
  [40, 50, 60],
  [10, 20, 30],
  [40, 50, 60],
  [10, 20, 30],
  [40, 50, 60],
]);

// data.shape(); [row, column]

// Get center column with .slice(startIndex, size)
// startIndex => [row, column], size => [numRows, width]
data.slice([0, 1], [-1, 1]); // -1 lets you get number of rows i.e. 8
// [[20], [50], [20], [50], ... ]
```

```js
// Concatenating tensors
const tensorA = tf.tensor([
  [1, 2, 3],
  [4, 5, 6],
]);

const tensorB = tf.tensor([
  [7, 8, 9],
  [10, 11, 12],
]);

tensorA.concat(tensorB).shape; // [4,3] -> adds more rows, axis 0 down/vertical

tensorA.concat(tensorB, 1).shape; // [2,6] -> flattens and adds more columns, axis 1 to right/lengthwise
```

```js
// Adding tensors
const jumpData = tf.tensor([
  [70, 70, 70],
  [70, 70, 70],
  [70, 70, 70],
  [70, 70, 70],
]);

const playerData = tf.tensor([
  [1, 160],
  [2, 160],
  [3, 160],
  [4, 160],
]);

jumpData.sum(); // 840; default looks at every value in tensor and sums them up

jumpData.sum(1); // [210,210,210,210] sum up rows along axis 1

// jumpData.sum(1).concat(playerData); // rank/dimensions must be the same

jumpData.sum(1).shape; // [4] - 1D tensor, reduces dimension of output tensor

jumpData.sum(1, true).shape; // keeps the dimension of the tensor -> [4,1]

jumpData.sum(1, true).concat(playerData, 1); // [[210,1,160],...]

jumpData.sum(1).expandDims().shape; // [1,4]

jumpData.sum(1).expandDims(1).shape; // [4,1], [[210], [210], [210], [210]]

// Better way to fix dimensions
jumpData.sum(1).expandDims(1).concat(playerData, 1);
```

K Nearest Neighbor Algorithm - Regression style

- Find distance between features and prediction point (pythagorean theorem)

- Sort from lowest point to greatest

- Take the top K records

- Average the label value of those top K records for regression vs. counting up the K records and choosing the one that appears the most for classification

Which bucket will a ball go into? -> classification
What is the price of a house given its location? -> regression

Given longitude and latitude tensor (features) and house price tensor (labels), make a prediction on the house price given certain latitude/longitude

```js
const features = tf.tensor([
  [-121, 47],
  [-121.2, 46.5],
  [-122, 46.4],
  [-120.9, 46.7],
]);

const labels = tf.tensor([[200], [250], [215], [240]]);

const predictionPoint = tf.tensor([-121, 47]);
const k = 2;

// Find distance between features and prediction point
// sqrt((lat - lat)^2 + (long - long)^2)
features
  .sub(predictionPoint) // [[0,0], [-0.19999, -0.5],..]
  .pow(2)
  .sum(1)
  .pow(0.5) // [0, 0.538, 1.166, 0.316] -> shape: [4]
  .expandDims(1) // [4,1] shape
  .concat(labels, 1) // Concat distances and labels together so we can sort by rows but we can't naturally sort tensors -> tensor of [distances, house prices]
  .unstack() // Split into a bunch of smaller tensors to sort - ugly output
  .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1)) // Sort distances from least to greatest
  .slice(0, k) // Normal array slice method i.e. [].slice(startIndex, lastIndexExclusive)
  .reduce((acc, pair) => {
    // Get house value from second column
    return acc + pair.get(1);
  }, 0) / k; // Sum up top k house prices and compute the average of the top k house prices;

// $220,000 is the guess of the price of the house given [-121, 47] latitude/longitude
```

Say using latitude, longitude, and sqft lot as the features, we should use standardization to help us
i.e. average sqft lot is 0 and the rest of the points are around the average like within -1 and 1 standard deviation

vs. normalization for sqft lot since we have small number of max value of 1 million, most around 5000, and small number of min values in the hundreds

Standardization = (value - average) / standard deviation

standard deviation = sqrt(variance)

```js
// Standardization example
const numbers = tf.tensor([
  [1, 2],
  [3, 4],
  [5, 6],
]);

// Default works like sum (average of all the elements together), so we need a second argument to handle
// tf.moments(data, 0) -> column computations
// tf.moments(data, 1) -> row computations
const { mean, variance } = tf.moments(numbers, 0);
// [3,4] for mean, [2.67, 2.67] for variance

// Standardization = (value - average) / standard deviation
numbers.sub(mean).div(variance.pow(0.5));
// [[-1.22, -1.22], [0,0], [1.22,1.22]]
```
