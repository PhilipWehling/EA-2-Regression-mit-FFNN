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

function init() {
  const cleanData = generateDataset(100);
  const splitClean = trainTestSplit(cleanData, 0.5);

  const noisyTrainData = addGaussianNoise(splitClean.trainData, variance);
  const noisyTestData = addGaussianNoise(splitClean.testData, variance);

  console.log("Saubere Trainingsdaten:", splitClean.trainData);
  console.log("Saubere Testdaten:", splitClean.testData);
  console.log("Verrauschte Trainingsdaten:", noisyTrainData);
  console.log("Verrauschte Testdaten:", noisyTestData);

  console.log("Anzahl clean train:", splitClean.trainData.length);
  console.log("Anzahl clean test:", splitClean.testData.length);
  console.log("Anzahl noisy train:", noisyTrainData.length);
  console.log("Anzahl noisy test:", noisyTestData.length);
}

init();