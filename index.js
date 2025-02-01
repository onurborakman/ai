import * as tf from '@tensorflow/tfjs';

// Create a simple model
const model = tf.sequential();
model.add(tf.layers.dense({ inputShape: [2], units: 8, activation: 'relu' }));
model.add(tf.layers.dense({ units: 4, activation: 'relu' }));
model.add(tf.layers.dense({ units: 1, activation: 'sigmoid' })); // Binary classification

model.compile({
    optimizer: tf.train.adam(),
    loss: 'binaryCrossentropy',
    metrics: ['accuracy']
});

// Training data (XOR problem)
const trainingData = tf.tensor2d([
    [0, 0], [0, 1], [1, 0], [1, 1]
]);
const labels = tf.tensor2d([
    [0], [1], [1], [0]
]);

// Train the model
async function trainModel() {
    console.log("Training started...");
    await model.fit(trainingData, labels, {
        epochs: 100,
        callbacks: {
            onEpochEnd: (epoch, logs) => console.log(`Epoch ${epoch}: loss=${logs.loss.toFixed(4)}, acc=${logs.acc?.toFixed(4)}`)
        }
    });
    console.log("Training complete!");

    // Test the model
    testModel();
}

async function testModel() {
    const testInput = tf.tensor2d([
        [0, 0], [0, 1], [1, 0], [1, 1]
    ]);

    const predictions = model.predict(testInput);
    predictions.print(); // Output model's predictions
}

// Run training
trainModel();
