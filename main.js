const variance = 0.05;

function targetFunction(x) {
  return 0.5 * (x + 0.8) * (x + 1.8) * (x - 0.2) * (x - 0.3) * (x - 1.9) + 1;
}

function randomUniform(min, max) {
  return Math.random() * (max - min) + min;
}

function generateDataset(n) {
  const data = [];

  for (let i = 0; i < n; i++) {
    const x = randomUniform(-2, 2);
    const y = targetFunction(x);
    data.push({ x, y });
  }

  return data;
}

function shuffleArray(array) {
  const shuffled = [...array];

  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }

  return shuffled;
}

function trainTestSplit(data, trainRatio = 0.5) {
  const shuffled = shuffleArray(data);
  const trainSize = Math.floor(shuffled.length * trainRatio);

  return {
    trainData: shuffled.slice(0, trainSize),
    testData: shuffled.slice(trainSize)
  };
}

function randomGaussian(mean = 0, stdDev = 1) {
  let u1 = 0;
  let u2 = 0;

  while (u1 === 0) u1 = Math.random();
  while (u2 === 0) u2 = Math.random();

  const z0 = Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
  return z0 * stdDev + mean;
}

function addGaussianNoise(data, variance = 0.05) {
  const stdDev = Math.sqrt(variance);

  return data.map(point => ({
    x: point.x,
    y: point.y + randomGaussian(0, stdDev)
  }));
}

function plotDataset(containerId, trainData, testData, title) {
  const trainTrace = {
    x: trainData.map(point => point.x),
    y: trainData.map(point => point.y),
    mode: "markers",
    type: "scatter",
    name: "Trainingsdaten",
    marker: {
      size: 8
    }
  };

  const testTrace = {
    x: testData.map(point => point.x),
    y: testData.map(point => point.y),
    mode: "markers",
    type: "scatter",
    name: "Testdaten",
    marker: {
      size: 8,
      symbol: "circle-open"
    }
  };

  const layout = {
    title,
    xaxis: { title: "x", range: [-2.1, 2.1] },
    yaxis: { title: "y" },
    margin: { t: 50, r: 20, b: 50, l: 60 },
    legend: { orientation: "h" }
  };

  Plotly.newPlot(containerId, [trainTrace, testTrace], layout, {
    responsive: true,
    displayModeBar: false
  });
}

function buildModel() {
  const model = tf.sequential();

  model.add(tf.layers.dense({
    inputShape: [1],
    units: 100,
    activation: "relu"
  }));

  model.add(tf.layers.dense({
    units: 100,
    activation: "relu"
  }));

  model.add(tf.layers.dense({
    units: 1,
    activation: "linear"
  }));

  model.compile({
    optimizer: tf.train.adam(0.01),
    loss: "meanSquaredError"
  });

  return model;
}

function dataToTensors(data) {
  const xs = tf.tensor2d(data.map(point => [point.x]));
  const ys = tf.tensor2d(data.map(point => [point.y]));

  return { xs, ys };
}

async function trainModel(model, trainData, epochs = 200) {
  const { xs, ys } = dataToTensors(trainData);

  const history = await model.fit(xs, ys, {
    epochs,
    batchSize: 32,
    shuffle: true,
    verbose: 0
  });

  xs.dispose();
  ys.dispose();

  return history;
}

function evaluateModel(model, data) {
  const { xs, ys } = dataToTensors(data);
  const lossTensor = model.evaluate(xs, ys);
  const loss = lossTensor.dataSync()[0];

  xs.dispose();
  ys.dispose();
  lossTensor.dispose();

  return loss;
}

function generatePredictionCurve(model, minX = -2, maxX = 2, numPoints = 200) {
  const xValues = [];
  const step = (maxX - minX) / (numPoints - 1);

  for (let i = 0; i < numPoints; i++) {
    xValues.push(minX + i * step);
  }

  const xsTensor = tf.tensor2d(xValues.map(x => [x]));
  const ysTensor = model.predict(xsTensor);
  const yValues = Array.from(ysTensor.dataSync());

  xsTensor.dispose();
  ysTensor.dispose();

  return xValues.map((x, index) => ({
    x,
    y: yValues[index]
  }));
}

function plotPrediction(containerId, data, predictionCurve, title) {
  const dataTrace = {
    x: data.map(point => point.x),
    y: data.map(point => point.y),
    mode: "markers",
    type: "scatter",
    name: "Echte Daten",
    marker: {
      size: 8
    }
  };

  const predictionTrace = {
    x: predictionCurve.map(point => point.x),
    y: predictionCurve.map(point => point.y),
    mode: "lines",
    type: "scatter",
    name: "Modellvorhersage",
    line: {
      width: 3
    }
  };

  const layout = {
    title,
    xaxis: { title: "x", range: [-2.1, 2.1] },
    yaxis: { title: "y" },
    margin: { t: 50, r: 20, b: 50, l: 60 },
    legend: { orientation: "h" }
  };

  Plotly.newPlot(containerId, [dataTrace, predictionTrace], layout, {
    responsive: true,
    displayModeBar: false
  });
}

async function init() {
  const cleanData = generateDataset(100);
  const splitData = trainTestSplit(cleanData, 0.5);

  const cleanTrainData = splitData.trainData;
  const cleanTestData = splitData.testData;

  const noisyTrainData = addGaussianNoise(cleanTrainData, variance);
  const noisyTestData = addGaussianNoise(cleanTestData, variance);

  plotDataset("plot-clean", cleanTrainData, cleanTestData, "Datensatz ohne Rauschen");
  plotDataset("plot-noisy", noisyTrainData, noisyTestData, "Datensatz mit Rauschen");

  const cleanModel = buildModel();
  await trainModel(cleanModel, cleanTrainData, 200);

  const cleanTrainLoss = evaluateModel(cleanModel, cleanTrainData);
  const cleanTestLoss = evaluateModel(cleanModel, cleanTestData);
  const cleanPredictionCurve = generatePredictionCurve(cleanModel);

  plotPrediction(
    "plot-clean-train-pred",
    cleanTrainData,
    cleanPredictionCurve,
    "Vorhersage auf Trainingsdaten"
  );

  plotPrediction(
    "plot-clean-test-pred",
    cleanTestData,
    cleanPredictionCurve,
    "Vorhersage auf Testdaten"
  );

  document.getElementById("clean-train-loss-text").textContent =
    `Train Loss (MSE): ${cleanTrainLoss.toFixed(6)}`;

  document.getElementById("clean-test-loss-text").textContent =
    `Test Loss (MSE): ${cleanTestLoss.toFixed(6)}`;

  const bestFitEpochs = 100;
  const overfitEpochs = 5000;

  const bestModel = buildModel();
  await trainModel(bestModel, noisyTrainData, bestFitEpochs);

  const bestTrainLoss = evaluateModel(bestModel, noisyTrainData);
  const bestTestLoss = evaluateModel(bestModel, noisyTestData);
  const bestPredictionCurve = generatePredictionCurve(bestModel);

  plotPrediction(
    "plot-best-train-pred",
    noisyTrainData,
    bestPredictionCurve,
    `Best-Fit auf Trainingsdaten (${bestFitEpochs} Epochs)`
  );

  plotPrediction(
    "plot-best-test-pred",
    noisyTestData,
    bestPredictionCurve,
    `Best-Fit auf Testdaten (${bestFitEpochs} Epochs)`
  );

  document.getElementById("best-train-loss-text").textContent =
    `Train Loss (MSE): ${bestTrainLoss.toFixed(6)}`;

  document.getElementById("best-test-loss-text").textContent =
    `Test Loss (MSE): ${bestTestLoss.toFixed(6)}`;

  const overfitModel = buildModel();
  await trainModel(overfitModel, noisyTrainData, overfitEpochs);

  const overfitTrainLoss = evaluateModel(overfitModel, noisyTrainData);
  const overfitTestLoss = evaluateModel(overfitModel, noisyTestData);
  const overfitPredictionCurve = generatePredictionCurve(overfitModel);

  plotPrediction(
    "plot-overfit-train-pred",
    noisyTrainData,
    overfitPredictionCurve,
    `Overfit auf Trainingsdaten (${overfitEpochs} Epochs)`
  );

  plotPrediction(
    "plot-overfit-test-pred",
    noisyTestData,
    overfitPredictionCurve,
    `Overfit auf Testdaten (${overfitEpochs} Epochs)`
  );

  document.getElementById("overfit-train-loss-text").textContent =
    `Train Loss (MSE): ${overfitTrainLoss.toFixed(6)}`;

  document.getElementById("overfit-test-loss-text").textContent =
    `Test Loss (MSE): ${overfitTestLoss.toFixed(6)}`;

  console.log("Clean Train Loss:", cleanTrainLoss);
  console.log("Clean Test Loss:", cleanTestLoss);
  console.log("Best Train Loss:", bestTrainLoss);
  console.log("Best Test Loss:", bestTestLoss);
  console.log("Overfit Train Loss:", overfitTrainLoss);
  console.log("Overfit Test Loss:", overfitTestLoss);
}

init();