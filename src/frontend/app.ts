import * as tf from "@tensorflow/tfjs";
import * as mnist from "mnist";
import { loadData } from "../network/mnistLoader";
import { Model, Network } from "../network/Network";

const modelEvaluationCanvas = document.getElementById(
  "model-evaluation-canvas"
) as HTMLCanvasElement;

const amountSampleUnitsInput = document.getElementById(
  "amount-sample-units-input"
) as HTMLInputElement;

const amountSampleUnitsForm = document.getElementById(
  "amount-sample-units-form"
) as HTMLFormElement;

const accuracyLabel = document.getElementById(
  "accuracy-label"
) as HTMLParagraphElement;

async function loadModel(): Promise<Model> {
  const response = await fetch("./network-weights.json");
  const model = await response.json();
  return model;
}

// Function to reconstruct the network with loaded weights and biases
async function initNetwork() {
  try {
    const model = await loadModel();
    const net = new Network([784, 30, 10]);

    net.biases = model.biases.map((b) => tf.tensor2d(b, [b.length, 1]));
    net.weights = model.weights.map((w) => tf.tensor2d(w.data, w.shape));

    return net;
  } catch (error) {
    console.log("Could not initialize network! Error: ", error);
    throw error;
  }
}

type CreateEvaluationCanvasGridParams = {
  canvas: HTMLCanvasElement;
  dataSample: number;
};

function createEvaluationCanvasGrid({
  dataSample,
  canvas,
}: CreateEvaluationCanvasGridParams) {
  // Get the canvas context
  const ctx = canvas.getContext("2d")!;
  // Clear the canvas before drawing
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  // Define grid parameters
  const imagesPerRow = 10; // Number of images per row
  const imageSize = 28; // MNIST images are 28x28 pixels
  const scale = 10; // Scale factor to enlarge the images
  const padding = 20; // Padding between images

  // Calculate canvas size based on the number of images
  const rows = Math.ceil(dataSample / imagesPerRow);
  const labelHeight = 50;
  canvas.width = imagesPerRow * (imageSize * scale + padding) + padding;
  canvas.height = rows * (imageSize * scale + padding + labelHeight);

  return { ctx, imagesPerRow, imageSize, scale, padding, labelHeight };
}

type DrawDigitOnCanvasParams = ReturnType<typeof createEvaluationCanvasGrid> & {
  indexInSample: number;
  predictedLabel: number;
  actualLabel: number;
  drawMnistDigit(context: CanvasRenderingContext2D): void;
};

function drawDigitEvaluationOnCanvas({
  ctx,
  imagesPerRow,
  imageSize,
  scale,
  padding,
  indexInSample,
  predictedLabel,
  actualLabel,
  labelHeight,
  drawMnistDigit,
}: DrawDigitOnCanvasParams) {
  // Calculate position on canvas
  const col = indexInSample % imagesPerRow;
  const row = Math.floor(indexInSample / imagesPerRow);

  const x = padding + col * (imageSize * scale + padding);
  const y = padding + row * (imageSize * scale + padding + labelHeight);

  // Create a temporary canvas to draw the image
  const tempCanvas = document.createElement("canvas");
  tempCanvas.width = imageSize;
  tempCanvas.height = imageSize;
  const tempCtx = tempCanvas.getContext("2d")!;

  // Draw the digit on the temporary canvas
  drawMnistDigit(tempCtx);

  // Scale the image and draw it on the main canvas
  ctx.drawImage(
    tempCanvas,
    0,
    0,
    imageSize,
    imageSize,
    x,
    y,
    imageSize * scale,
    imageSize * scale
  );

  // Determine if prediction is correct
  const isCorrect = predictedLabel === actualLabel;

  // Draw rectangle around the image
  ctx.strokeStyle = isCorrect ? "green" : "red";
  ctx.lineWidth = 5;
  ctx.strokeRect(x, y, imageSize * scale, imageSize * scale);

  // If incorrect, display the predicted label
  if (!isCorrect) {
    const fontValue = 38;
    ctx.fillStyle = "red";
    ctx.font = `bold ${fontValue}px Arial`;
    const textOffset = fontValue;
    ctx.fillText(
      `Predicted: ${predictedLabel}`,
      x,
      y + imageSize * scale + textOffset
    );
  }
}

// Function to handle feeding test data forward
// TODO: change amountSampleUnits name to sample
function handleSubmitAmountSampleUnits(amountSampleUnits: number, net: Network) {
  const { modelTestData, mnistTestData } = loadData();

  console.log(`Test data length: ${mnistTestData.length}`)

  // Create the test sample based on amount of amount of sample units selected
  const testSample = modelTestData.slice(0, amountSampleUnits);
  /**
   * Need to use original data from mnist library (i.e. data
   * that has not been formatted for model consumption) since
   * the original data input can be passed to the mnist.draw
   * method.
   */
  const testSampleMnistData = mnistTestData.slice(0, amountSampleUnits);

  const { ctx, imagesPerRow, imageSize, scale, padding, labelHeight } =
    createEvaluationCanvasGrid({
      canvas: modelEvaluationCanvas,
      dataSample: amountSampleUnits,
    });

  let correctGuesses = 0;

  testSample.forEach(([input, label], index) => {
    const mnistDataInput = testSampleMnistData[index].input;
    const output = net.feedforward(input);
    const predictedLabel = tf.argMax(output, 0).dataSync()[0];
    const drawMnistDigit = (context: CanvasRenderingContext2D) =>
      mnist.draw(mnistDataInput, context);

    correctGuesses += predictedLabel === label ? 1 : 0;

    drawDigitEvaluationOnCanvas({
      ctx,
      imagesPerRow,
      imageSize,
      scale,
      padding,
      actualLabel: label,
      predictedLabel,
      drawMnistDigit,
      indexInSample: index,
      labelHeight,
    });

    accuracyLabel.innerText = `Prediction accuracy: ${
      (correctGuesses / amountSampleUnits) * 100
    }%`;
  });
}

// Initialize the network and set up event listeners
(async () => {
  const net = await initNetwork();

  if (!net) {
    return;
  }

  console.log("Network initialized succesfully!");

  amountSampleUnitsForm.addEventListener("submit", (event) => {
    event.preventDefault();

    const amountSampleUnits = Number(amountSampleUnitsInput.value);
    handleSubmitAmountSampleUnits(amountSampleUnits, net);
  });
})();
