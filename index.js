const tf = require("@tensorflow/tfjs-node");

const model = tf.sequential();
model.add(
    tf.layers.simpleRNN({
        units: 5,
        recurrentInitializer: "GlorotNormal",
        inputShape: [1, 1]
    })
);

const a = tf.variable(tf.scalar(Math.random()));
const b = tf.variable(tf.scalar(Math.random()));
const c = tf.variable(tf.scalar(Math.random()));
const d = tf.variable(tf.scalar(Math.random()));

function predict(x) {
    return tf.tidy(() => {
      return a.mul(x.pow(tf.scalar(3)))
        .add(b.mul(x.square()))
        .add(c.mul(x))
        .add(d)
    });
  }

function loss(predictions, labels) {
    const meanSquareError = predictions.sub(labels).square().mean();
    return meanSquareError;
}

function train(xs, ys, numIterations = 75) {
    const learningRate = 0.5;
    const optimizer = tf.train.sgd(learningRate);

    for (let i = 0; i < numIterations; ++i) {
        optimizer.minimize(() => {
            const predYs = predict(xs);
            return loss(predYs, ys);
        })
    }
}

const X = tf.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0].map(() => {return Math.random()}));
const A = tf.scalar(1), B = tf.scalar(2), C = tf.scalar(3), D = tf.scalar(4);
const Y = A.mul(X.pow(tf.scalar(3))).add(B.mul(X.square())).add(C.mul(X)).add(D)

train(X, Y, 1000);

console.log(`${a.toString().replace("Tensor\n    ", "")}x^3 + ${b.toString().replace("Tensor\n    ", "")}x^2 + ${c.toString().replace("Tensor\n    ", "")}x + ${d.toString().replace("Tensor\n    ", "")}`);