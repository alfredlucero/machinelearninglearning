require("@tensorflow/tfjs-node"); // Runs calculations on CPU here vs. GPU
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

// Debug with node --inspect-brk index.js
// Open Chrome browser and navigate to about:inspect
// Click inspect to see debugger open up so you can see variables of your code
// Set breakpoint inside the knn function
// Log out features i.e. features.shape and features.print(), predictionPoint.print(), scaledPrediction.print() - should be close to 0
// Print out parts of the algorithm to double check data looks right i.e. all data points should be close to 0 after standardizing
// Can later improve algorithm by adjusting k and logging out
// Can also use more features i.e. sqft_living
function knn(features, labels, predictionPoint, k) {
  const { mean, variance } = tf.moments(features, 0);

  const scaledPrediction = predictionPoint.sub(mean).div(variance.pow(0.5));

  return (
    features
      .sub(mean)
      .div(variance.pow(0.5)) // Standardization
      .sub(scaledPrediction) // Distance calculations
      .pow(2)
      .sum(1)
      .pow(0.5) // shape: [4]
      .expandDims(1) // [4,1] shape
      .concat(labels, 1) // Concat distances and labels together so we can sort by rows but we can't naturally sort tensors -> tensor of [features, labels]
      .unstack() // Split into a bunch of smaller tensors to sort - ugly output
      .sort((a, b) => (a.get(0) > b.get(0) ? 1 : -1)) // Sort features from least to greatest
      .slice(0, k) // Normal array slice method i.e. [].slice(startIndex, lastIndexExclusive)
      .reduce((acc, pair) => {
        // Get house value from second column
        return acc + pair.get(1);
      }, 0) / k
  ); // Sum up top k house prices and compute the average of the top k house prices;
}

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long", "sqft_lot", "sqft_living"],
    labelColumns: ["price"],
  }
);

// Should have 10 rows in each
// console.log("Test features", testFeatures);
// console.log("Test labels (house values)", testLabels);

features = tf.tensor(features);
labels = tf.tensor(labels);

testFeatures.forEach((testPoint, i) => {
  const result = knn(features, labels, tf.tensor(testPoint), 10);
  // Error = (expected value - predicted value) / expected value
  const error = (testLabels[i][0] - result) / testLabels[i][0];
  console.log("Guess", result, "Actual Test House Value", testLabels[i][0]);
  console.log("Error", error * 100);
});
