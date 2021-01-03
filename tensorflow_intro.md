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
