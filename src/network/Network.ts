import * as tf from "@tensorflow/tfjs";
import { shuffle, sigmoid, sigmoidPrime } from "./Network.utils";

type LayerSizes = Array<number>;
type Data = Array<[tf.Tensor2D, tf.Tensor2D]>;
type TestData = Array<[tf.Tensor2D, number]>;
export type Model = {
  biases: Array<Array<Array<number>>>;
  weights: {
    shape: [number, number];
    data: Array<Array<number>>;
  }[];
};

export class Network {
  numberOfLayers: number;
  layerSizes: LayerSizes;
  biases: Array<tf.Tensor2D>;
  weights: Array<tf.Tensor2D>;

  constructor(layerSizes: LayerSizes) {
    this.numberOfLayers = layerSizes.length;
    /* Each number item represents the number of neurons
    in that layer */
    this.layerSizes = layerSizes;
    const layerSizesNoInputLayer = layerSizes.slice(1);
    /* Creating 2d array with one column, which is the bias of each neuron. Each neuron
     * is reprenented by one row in the matrix. The input layer does not have biases. Eg:
     * [[-0.4011963],
     * [-0.7450452],
     * [-0.5654289]]
     */
    this.biases = layerSizesNoInputLayer.map((y) => tf.randomNormal([y, 1]));
    const layerSizesNoOutputLayer = layerSizes.slice(0, -1);
    /* Creating 2d array where each row represents a neuron's weighted connections
     * where each column represents a weighted connection to one of the neurons in
     * the previous layer. Eg:
     * [[-1.3365294, 0.7607032 ],
     * [-0.0870466, -0.8207377],
     * [0.1671798 , 0.8741785 ]] -> 3 neurons, 2 weighted connections for each
     */
    this.weights = layerSizesNoOutputLayer.map((x, i) =>
      tf.randomNormal([layerSizes[i + 1], x])
    );
  }

  feedforward(activations: tf.Tensor2D): tf.Tensor2D {
    for (let i = 0; i < this.biases.length; i++) {
      const b = this.biases[i];
      const w = this.weights[i];
      /* The vector of activations is the result of multiplying the vector of weights associated
       * with a layer with the previous layer's vector of activations (a), adding the vector of biases (b) and passing
       * to the sigmoid (σ):
       *
       *                 a′=σ(wa+b)
       */
      activations = sigmoid(tf.add(tf.matMul(w, activations), b));
    }
    return activations;
  }

  // Stochastic Gradient Descent (SGD), how the network learns
  applySGD(
    trainingData: Data,
    epochs: number,
    miniBatchSize: number,
    learningRate: number,
    testData: TestData | null = null
  ): void {
    for (let j = 0; j < epochs; j++) {
      shuffle(trainingData);

      const miniBatches: Array<Data> = [];
      // Dividing training data into mini batches
      for (let k = 0; k < trainingData.length; k += miniBatchSize) {
        miniBatches.push(trainingData.slice(k, k + miniBatchSize));
      }

      for (const miniBatch of miniBatches) {
        this.updateMiniBatch(miniBatch, learningRate);
      }

      if (testData) {
        // Check how the network is performing after an epoch of training.
        console.log(
          `Epoch ${j}: ${this.evaluate(testData)} / ${testData.length}`
        );
      } else {
        console.log(`Epoch ${j} complete`);
      }
    }
  }

  costDerivative(
    outputActivations: tf.Tensor2D,
    desiredOutputActivations: tf.Tensor2D
  ): tf.Tensor2D {
    return tf.sub(outputActivations, desiredOutputActivations);
  }

  backprop(
    x: tf.Tensor2D,
    y: tf.Tensor2D
  ): [nablaB: Array<tf.Tensor2D>, nablaW: Array<tf.Tensor2D>] {
    const nablaB = this.biases.map((b) => tf.zerosLike(b));
    const nablaW = this.weights.map((w) => tf.zerosLike(w));

    // feedforward
    let activation = x;
    const activations: Array<tf.Tensor2D> = [x];
    const zs: Array<tf.Tensor2D> = [];

    for (let i = 0; i < this.biases.length; i++) {
      const b = this.biases[i];
      const w = this.weights[i];
      const z = tf.add<tf.Tensor2D>(tf.matMul(w, activation), b);
      zs.push(z);
      activation = sigmoid(z);
      activations.push(activation);
    }

    // backward pass
    // delta is dCx / da(L) ⊙ σ'(z(L)) = dCx / db(L)
    let delta = tf.mul<tf.Tensor2D>(
      this.costDerivative(activations[activations.length - 1], y),
      sigmoidPrime(zs[zs.length - 1])
    );
    // nablaB[nablaB.length - 1] is dCx / db(L)
    // nablaW[nablaW.length - 1] is dCx / dw(L)
    nablaB[nablaB.length - 1] = delta;
    nablaW[nablaW.length - 1] = tf.matMul(
      delta,
      activations[activations.length - 2].transpose()
    );

    for (let l = 2; l < this.numberOfLayers; l++) {
      const z = zs[zs.length - l];
      const sp = sigmoidPrime(z);
      delta = tf.mul(
        tf.matMul(this.weights[this.weights.length - l + 1].transpose(), delta),
        sp
      );
      // nablaB[nablaB.length - l] is dCx / db(l)
      nablaB[nablaB.length - l] = delta;
      // nablaW[nablaW.length - l] is dCx / dw(l)
      nablaW[nablaW.length - l] = tf.matMul(
        delta,
        activations[activations.length - l - 1].transpose()
      );
    }

    // Return the gradients
    return [nablaB, nablaW];
  }
  /*
   * Update the network's weights and biases by applying a single step of
   * gradient descent using backpropagation to a single mini batch.
   */
  updateMiniBatch(miniBatch: Data, learningRate: number): void {
    const nablaBiases = this.biases.map((b) => tf.zerosLike(b));
    const nablaWeights = this.weights.map((w) => tf.zerosLike(w));

    // x is the input layer activations, and y is the desired
    // activations output
    for (const [x, y] of miniBatch) {
      // We invoke the backpropagation algorithm for every mini batch,
      // which gives us the gradient of the cost function.
      const [delta_nabla_b, delta_nabla_w] = this.backprop(x, y);
      for (let i = 0; i < nablaBiases.length; i++) {
        nablaBiases[i] = tf.add(nablaBiases[i], delta_nabla_b[i]);
        nablaWeights[i] = tf.add(nablaWeights[i], delta_nabla_w[i]);
      }
    }

    const learningRateOverLength = learningRate / miniBatch.length;
    for (let i = 0; i < this.weights.length; i++) {
      this.weights[i] = tf.sub(
        this.weights[i],
        tf.mul(learningRateOverLength, nablaWeights[i])
      );
      this.biases[i] = tf.sub(
        this.biases[i],
        tf.mul(learningRateOverLength, nablaBiases[i])
      );
    }

    // Dispose of intermediate tensors to free up memory
    // nablaBiases.forEach((tensor) => tensor.dispose());
    // nablaWeights.forEach((tensor) => tensor.dispose());
  }

  evaluate(testData: TestData): number {
    const testResults = testData.map(([x, y]) => {
      const output = this.feedforward(x);
      const predicted = tf.argMax(output, 0); // Index of highest activation
      const actual = y; // Index of the actual label
      return predicted.dataSync()[0] === actual;
    });
    return testResults.reduce((acc, result) => acc + (result ? 1 : 0), 0);
  }

  exportModel(): Model {
    return {
      biases: this.biases.map((b) => b.arraySync()),
      weights: this.weights.map((w) => ({
        shape: w.shape,
        data: w.arraySync(),
      })),
    };
  }
}
