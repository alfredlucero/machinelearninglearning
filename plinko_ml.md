# Plinko Case Study

Given some data about where a ball is dropped from, can we predict what bucket it will end up in?

1. Identify data that is relevant to the problem

Drop position in pixels (0 on far left)

The bucket the ball falls into i.e. 1 - 10

Range of ball bounciness (0 to 1) -> more bouncy tend to go farther from where it started?

Range of ball size (1 to 30)
-> larger affects which bucket it falls into

Changing one of these Features:
Drop position
Ball bounciness
Ball size

Will probably change this Label:
Bucket a ball lands in

How to store datasets in JS:

- Array of objects

```js
[
  {
    dropPosition: 300,
    bounciness: 0.4,
    ballSize: 16,
    bucket: 4,
  },
];
```

- Array of arrays (matrices); much easier to use with libraries

```js
// dropPosition, bounciness, ballSize, bucket
[
  [300, 0.4, 16, 4], // one ball drop
];
```

- classification prediction because discrete/finite number of buckets the ball can land in

- example classification algorithm: `K-nearest neighbor` (knn) "birds of a feather flock together"

  - thought experiment: what would happen if we dropped a ball ten times around this spot (300px drop position) -> if it lands at #4 most of the time around the same drop spot, it's likely to go into bucket 4

  - drop a ball a bunch of times all around the board, record which bucket it goes into

  - for each observation, subtract drop point from 300px, take absolute value

  - sort the results from least to greatest (based on distance from 300px)

  - look at the 'k' top records. What was the most common bucket? Take the most common out of top K records i.e. 4th bucket showed up the most out of top 3 where K = 3

  - whichever bucket came up most frequently is the one ours will probably go into

  - K is important to tinker with

- If prediction is bad,

  - try adjusting the k value/parameters

  - add more features to explain the analysis

  - change the prediction point

  - accept that maybe there isn't a good correlation

  - but we need a good way to compare accuracy with different settings

- To find the ideal k

  - record bunch of data points

  - split data into a 'training' set and a 'test' set (can shuffle data first and vary the size of the test set)

  - for each 'test' record, run KNN using the 'training' data

  - does the result of KNN equal the 'test' record bucket?

  - depending on our k and test set size, the accuracy changes, so we may need to add more features -> multi-dimensional KNN

- K-Nearest Neighbor with multiple variables

  - which bucket will a ball go into if dropped at 300px and bounciness of 0.5

  - drop a ball a bunch of times all around the board, record which bucket it goes into

  - for each observation, find distance from observation to prediction point of (300, 0.5) i.e. imagine mapping it to X/Y axes; distance through using pythagorean theorem

  - sort results from least to greatest

  - look at the 'k' top records. What was the most common bucket?

  - Whichever bucket came up most frequently is the one ours will probably go in

  - bounciness not making as much of an impact for k since it's a tiny factor in the pythagorean theorem calculation i.e. 0.55 - 0.52 squared = 0.009

    - we must take the original data and normalize it i.e. normalize the drop position such that it doesn't have such a huge effect on the k

    - we normalize by scaling and mapping data between 0 and 1 with a minmax method

    - normalized dataset = (featureValue - minOfFeatureValues) / (maxOfFeatureValues - minOfFeatureValues)
      ->
      200 .1
      150 0
      650 1
      430 .56

    - weighting our ball drop position and bounciness equally makes our predictions worse than guessing - why?

      - changes to drop position -> predictable changes to output

      - changes to ball bounciness -> changes our output, but not predictably

    - we can also standardize by basing numbers around 0 for a normal distribution around -1 and 1 standard deviations

    - feature selection: deciding which features to include in analysis

      - can run knn only based on drop position and record accuracy - 30% (only good predictor out of the 3 features); do the same for bounciness and record accuracy - 10%; ball size - 10% -> objective feature selection by picking only one at a time and seeing how it fares
