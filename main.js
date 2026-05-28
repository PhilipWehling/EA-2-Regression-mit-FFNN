const variance = 0.05;

const state = {
  cleanTrainData: null,
  cleanTestData: null,
  noisyTrainData: null,
  noisyTestData: null,
  cleanModel: null,
  bestModel: null,
  overfitModel: null,
  devDataset: null,
  devModel: null,
  isTraining: false
};

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

function addGaussianNoise(data, v = 0.05) {
  const stdDev = Math.sqrt(v);

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
    marker: { size: 8 }
  };

  const testTrace = {
    x: testData.map(point => point.x),
    y: testData.map(point => point.y),
    mode: "markers",
    type: "scatter",
    name: "Testdaten",
    marker: { size: 8, symbol: "circle-open" }
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
    batchSize: 64,
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

  return xValues.map((x, index) => ({ x, y: yValues[index] }));
}

function plotPrediction(containerId, data, predictionCurve, title) {
  const dataTrace = {
    x: data.map(point => point.x),
    y: data.map(point => point.y),
    mode: "markers",
    type: "scatter",
    name: "Echte Daten",
    marker: { size: 8 }
  };

  const predictionTrace = {
    x: predictionCurve.map(point => point.x),
    y: predictionCurve.map(point => point.y),
    mode: "lines",
    type: "scatter",
    name: "Modellvorhersage",
    line: { width: 3 }
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

  state.cleanTrainData = splitData.trainData;
  state.cleanTestData = splitData.testData;

  state.noisyTrainData = addGaussianNoise(state.cleanTrainData, variance);
  state.noisyTestData = addGaussianNoise(state.cleanTestData, variance);

  plotDataset("plot-clean", state.cleanTrainData, state.cleanTestData, "Datensatz ohne Rauschen");
  plotDataset("plot-noisy", state.noisyTrainData, state.noisyTestData, "Datensatz mit Rauschen");

  state.cleanModel = buildModel();
  await trainModel(state.cleanModel, state.cleanTrainData, 200);

  const cleanTrainLoss = evaluateModel(state.cleanModel, state.cleanTrainData);
  const cleanTestLoss = evaluateModel(state.cleanModel, state.cleanTestData);
  const cleanPredictionCurve = generatePredictionCurve(state.cleanModel);

  plotPrediction("plot-clean-train-pred", state.cleanTrainData, cleanPredictionCurve, "Vorhersage auf Trainingsdaten");
  plotPrediction("plot-clean-test-pred", state.cleanTestData, cleanPredictionCurve, "Vorhersage auf Testdaten");

  document.getElementById("clean-train-loss-text").textContent =
    `Train Loss (MSE): ${cleanTrainLoss.toFixed(6)}`;
  document.getElementById("clean-test-loss-text").textContent =
    `Test Loss (MSE): ${cleanTestLoss.toFixed(6)}`;

  const bestFitEpochs = 100;
  const overfitEpochs = 5000;

  state.bestModel = buildModel();
  await trainModel(state.bestModel, state.noisyTrainData, bestFitEpochs);

  const bestTrainLoss = evaluateModel(state.bestModel, state.noisyTrainData);
  const bestTestLoss = evaluateModel(state.bestModel, state.noisyTestData);
  const bestPredictionCurve = generatePredictionCurve(state.bestModel);

  plotPrediction("plot-best-train-pred", state.noisyTrainData, bestPredictionCurve,
    `Best-Fit auf Trainingsdaten (${bestFitEpochs} Epochs)`);
  plotPrediction("plot-best-test-pred", state.noisyTestData, bestPredictionCurve,
    `Best-Fit auf Testdaten (${bestFitEpochs} Epochs)`);

  document.getElementById("best-train-loss-text").textContent =
    `Train Loss (MSE): ${bestTrainLoss.toFixed(6)}`;
  document.getElementById("best-test-loss-text").textContent =
    `Test Loss (MSE): ${bestTestLoss.toFixed(6)}`;

  state.overfitModel = buildModel();
  await trainModel(state.overfitModel, state.noisyTrainData, overfitEpochs);

  const overfitTrainLoss = evaluateModel(state.overfitModel, state.noisyTrainData);
  const overfitTestLoss = evaluateModel(state.overfitModel, state.noisyTestData);
  const overfitPredictionCurve = generatePredictionCurve(state.overfitModel);

  plotPrediction("plot-overfit-train-pred", state.noisyTrainData, overfitPredictionCurve,
    `Overfit auf Trainingsdaten (${overfitEpochs} Epochs)`);
  plotPrediction("plot-overfit-test-pred", state.noisyTestData, overfitPredictionCurve,
    `Overfit auf Testdaten (${overfitEpochs} Epochs)`);

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

  initDevTools();
}

// ── Entwickler-Tools ──────────────────────────────────────────────────────────

function initDevTools() {
  setupTabs();
  setupDatasetTab();
  setupTrainingTab();
  setupModelIOTab();
  setupTestTab();
}

function setupTabs() {
  document.querySelectorAll(".tab-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab-btn").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach(p => { p.style.display = "none"; });
      btn.classList.add("active");
      document.getElementById("tab-" + btn.dataset.tab).style.display = "block";
    });
  });
}

// ── Tab 1: Datensatz ──────────────────────────────────────────────────────────

function setupDatasetTab() {
  document.getElementById("btn-generate-ds").addEventListener("click", () => {
    const n = parseInt(document.getElementById("ds-n").value);
    const v = parseFloat(document.getElementById("ds-variance").value);
    const split = parseFloat(document.getElementById("ds-split").value);
    const noisy = document.getElementById("ds-noisy").checked;

    if (isNaN(n) || n < 2 || isNaN(v) || v < 0 || isNaN(split) || split <= 0 || split >= 1) {
      document.getElementById("dev-ds-info").textContent = "Ungültige Parameter.";
      return;
    }

    const rawData = generateDataset(n);
    const { trainData, testData } = trainTestSplit(rawData, split);
    const finalTrain = noisy ? addGaussianNoise(trainData, v) : trainData;
    const finalTest = noisy ? addGaussianNoise(testData, v) : testData;

    state.devDataset = {
      trainData: finalTrain,
      testData: finalTest,
      config: { n, variance: v, split, noisy }
    };

    plotDataset("dev-ds-plot", finalTrain, finalTest,
      `Erzeugter Datensatz (n=${n}${noisy ? ", verrauscht" : ""})`);
    document.getElementById("dev-ds-info").textContent =
      `${finalTrain.length} Trainingspunkte, ${finalTest.length} Testpunkte`;
    document.getElementById("btn-save-ds").disabled = false;
  });

  document.getElementById("btn-save-ds").addEventListener("click", () => {
    if (!state.devDataset) return;
    const blob = new Blob([JSON.stringify(state.devDataset, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "dataset.json";
    a.click();
    URL.revokeObjectURL(url);
  });

  document.getElementById("input-load-ds").addEventListener("change", e => {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = ev => {
      try {
        const parsed = JSON.parse(ev.target.result);
        if (!parsed.trainData || !parsed.testData) throw new Error("Fehlende Felder.");
        state.devDataset = parsed;
        plotDataset("dev-ds-plot", parsed.trainData, parsed.testData, "Geladener Datensatz");
        document.getElementById("dev-ds-info").textContent =
          `Geladen: ${parsed.trainData.length} Trainingspunkte, ${parsed.testData.length} Testpunkte`;
        document.getElementById("btn-save-ds").disabled = false;
      } catch (err) {
        document.getElementById("dev-ds-info").textContent = "Fehler beim Laden: " + err.message;
      }
    };
    reader.readAsText(file);
  });
}

// ── Tab 2: Training ───────────────────────────────────────────────────────────

function buildCustomModel(units1, units2, lr) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1], units: units1, activation: "relu" }));
  if (units2 > 0) {
    model.add(tf.layers.dense({ units: units2, activation: "relu" }));
  }
  model.add(tf.layers.dense({ units: 1, activation: "linear" }));
  model.compile({ optimizer: tf.train.adam(lr), loss: "meanSquaredError" });
  return model;
}

function setupTrainingTab() {
  document.getElementById("btn-train").addEventListener("click", async () => {
    if (state.isTraining) return;

    const dsSelect = document.getElementById("train-dataset-select").value;
    let trainData, testData;

    if (dsSelect === "clean") {
      trainData = state.cleanTrainData;
      testData = state.cleanTestData;
    } else if (dsSelect === "noisy") {
      trainData = state.noisyTrainData;
      testData = state.noisyTestData;
    } else {
      if (!state.devDataset) {
        document.getElementById("train-progress").textContent =
          "Kein benutzerdefinierter Datensatz vorhanden. Bitte zuerst im Datensatz-Tab erzeugen oder laden.";
        return;
      }
      trainData = state.devDataset.trainData;
      testData = state.devDataset.testData;
    }

    const epochs = parseInt(document.getElementById("train-epochs").value);
    const lr = parseFloat(document.getElementById("train-lr").value);
    const batchSize = parseInt(document.getElementById("train-batchsize").value);
    const units1 = parseInt(document.getElementById("train-units1").value);
    const units2 = parseInt(document.getElementById("train-units2").value);

    if (isNaN(epochs) || epochs < 1 || isNaN(lr) || lr <= 0 || isNaN(batchSize) || batchSize < 1 || isNaN(units1) || units1 < 1) {
      document.getElementById("train-progress").textContent = "Ungültige Trainingsparameter.";
      return;
    }

    state.isTraining = true;
    const btn = document.getElementById("btn-train");
    btn.disabled = true;
    btn.textContent = "Trainiere…";

    const progressEl = document.getElementById("train-progress");
    progressEl.textContent = "Training gestartet…";
    document.getElementById("dev-train-loss").textContent = "";

    if (state.devModel) {
      state.devModel.dispose();
    }
    state.devModel = buildCustomModel(units1, units2, lr);

    const { xs, ys } = dataToTensors(trainData);
    const updateEvery = Math.max(1, Math.floor(epochs / 50));

    await state.devModel.fit(xs, ys, {
      epochs,
      batchSize,
      shuffle: true,
      verbose: 0,
      callbacks: {
        onEpochEnd: async (epoch, logs) => {
          if ((epoch + 1) % updateEvery === 0 || epoch === epochs - 1) {
            progressEl.textContent = `Epoche ${epoch + 1} / ${epochs} – Loss: ${logs.loss.toFixed(6)}`;
          }
          if ((epoch + 1) % 10 === 0) {
            await tf.nextFrame();
          }
        }
      }
    });

    xs.dispose();
    ys.dispose();

    const trainLoss = evaluateModel(state.devModel, trainData);
    const testLoss = evaluateModel(state.devModel, testData);
    const predCurve = generatePredictionCurve(state.devModel);

    plotPrediction("dev-train-plot", trainData, predCurve, "Vorhersage auf Trainingsdaten");
    document.getElementById("dev-train-loss").textContent =
      `Train Loss (MSE): ${trainLoss.toFixed(6)}  |  Test Loss (MSE): ${testLoss.toFixed(6)}`;

    progressEl.textContent = `Training abgeschlossen (${epochs} Epochen).`;
    state.isTraining = false;
    btn.disabled = false;
    btn.textContent = "Modell trainieren";
  });
}

// ── Tab 3: Modell speichern/laden ─────────────────────────────────────────────

async function refreshModelList() {
  const models = await tf.io.listModels();
  const select = document.getElementById("model-ls-select");
  select.innerHTML = "";

  const lsKeys = Object.keys(models).filter(k => k.startsWith("localstorage://"));

  if (lsKeys.length === 0) {
    select.innerHTML = "<option value=''>– Keine gespeicherten Modelle –</option>";
    return;
  }

  lsKeys.forEach(key => {
    const opt = document.createElement("option");
    opt.value = key;
    opt.textContent = key.replace("localstorage://", "");
    select.appendChild(opt);
  });
}

function setupModelIOTab() {
  refreshModelList();

  document.getElementById("btn-save-localstorage").addEventListener("click", async () => {
    if (!state.devModel) {
      document.getElementById("model-io-status").textContent =
        "Kein trainiertes Modell vorhanden. Bitte zuerst im Training-Tab ein Modell trainieren.";
      return;
    }
    const name = document.getElementById("model-save-name").value.trim() || "my-model";
    await state.devModel.save(`localstorage://${name}`);
    document.getElementById("model-io-status").textContent = `Modell "${name}" in localStorage gespeichert.`;
    await refreshModelList();
  });

  document.getElementById("btn-save-download").addEventListener("click", async () => {
    if (!state.devModel) {
      document.getElementById("model-io-status").textContent =
        "Kein trainiertes Modell vorhanden. Bitte zuerst im Training-Tab ein Modell trainieren.";
      return;
    }
    const name = document.getElementById("model-save-name").value.trim() || "my-model";
    await state.devModel.save(`downloads://${name}`);
    document.getElementById("model-io-status").textContent = `Modell "${name}" wird heruntergeladen.`;
  });

  document.getElementById("btn-load-localstorage").addEventListener("click", async () => {
    const key = document.getElementById("model-ls-select").value;
    if (!key) {
      document.getElementById("model-io-status").textContent = "Bitte ein Modell aus der Liste wählen.";
      return;
    }
    try {
      if (state.devModel) state.devModel.dispose();
      state.devModel = await tf.loadLayersModel(key);
      state.devModel.compile({ optimizer: tf.train.adam(0.01), loss: "meanSquaredError" });
      document.getElementById("model-io-status").textContent =
        `Modell "${key.replace("localstorage://", "")}" geladen.`;
    } catch (err) {
      document.getElementById("model-io-status").textContent = "Fehler beim Laden: " + err.message;
    }
  });

  document.getElementById("btn-load-files").addEventListener("click", async () => {
    const jsonFile = document.getElementById("input-model-json").files[0];
    const weightFiles = Array.from(document.getElementById("input-model-weights").files);

    if (!jsonFile) {
      document.getElementById("model-io-status").textContent = "Bitte JSON-Datei auswählen.";
      return;
    }
    try {
      if (state.devModel) state.devModel.dispose();
      state.devModel = await tf.loadLayersModel(tf.io.browserFiles([jsonFile, ...weightFiles]));
      state.devModel.compile({ optimizer: tf.train.adam(0.01), loss: "meanSquaredError" });
      document.getElementById("model-io-status").textContent = "Modell aus Dateien geladen.";
    } catch (err) {
      document.getElementById("model-io-status").textContent = "Fehler beim Laden: " + err.message;
    }
  });
}

// ── Tab 4: Modell testen ──────────────────────────────────────────────────────

function setupTestTab() {
  document.getElementById("btn-test").addEventListener("click", () => {
    const modelSelect = document.getElementById("test-model-select").value;
    const dsSelect = document.getElementById("test-dataset-select").value;
    const statusEl = document.getElementById("dev-test-loss");

    let model;
    if (modelSelect === "clean") model = state.cleanModel;
    else if (modelSelect === "best") model = state.bestModel;
    else if (modelSelect === "overfit") model = state.overfitModel;
    else model = state.devModel;

    if (!model) {
      statusEl.textContent = "Kein Modell verfügbar. Bitte zuerst trainieren oder laden.";
      return;
    }

    let data;
    if (dsSelect === "clean-train") data = state.cleanTrainData;
    else if (dsSelect === "clean-test") data = state.cleanTestData;
    else if (dsSelect === "noisy-train") data = state.noisyTrainData;
    else if (dsSelect === "noisy-test") data = state.noisyTestData;
    else {
      if (!state.devDataset) {
        statusEl.textContent = "Kein benutzerdefinierter Datensatz vorhanden.";
        return;
      }
      data = [...state.devDataset.trainData, ...state.devDataset.testData];
    }

    const loss = evaluateModel(model, data);
    const predCurve = generatePredictionCurve(model);
    plotPrediction("dev-test-plot", data, predCurve, "Modell-Test: Vorhersage vs. Daten");
    statusEl.textContent = `Loss (MSE): ${loss.toFixed(6)}`;
  });
}

init();
