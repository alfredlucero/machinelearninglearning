require("@tensorflow/tfjs-node"); // Runs calculations on CPU here vs. GPU
const tf = require("@tensorflow/tfjs");
const loadCSV = require("./load-csv");

function knn(features, labels, predictionPoint, k) {
  return (
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
      }, 0) / k
  ); // Sum up top k house prices and compute the average of the top k house prices;
}

let { features, labels, testFeatures, testLabels } = loadCSV(
  "kc_house_data.csv",
  {
    shuffle: true,
    splitTest: 10,
    dataColumns: ["lat", "long"],
    labelColumns: ["price"],
  }
);

// Should have 10 rows in each
console.log("Test features (lat, lang)", testFeatures);
console.log("Test labels (house values)", testLabels);

features = tf.tensor(features);
labels = tf.tensor(labels);

const result = knn(features, labels, tf.tensor(testFeatures[0]), 10);
// Error = (expected value - predicted value) / expected value
const error = (testLabels[0][0] - result) / testLabels[0][0];
console.log("Guess", result, "Actual Test House Value", testLabels[0][0]);
console.log("Error", error);
