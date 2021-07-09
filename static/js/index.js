let tfjs;
let model;
const webcam = new Webcam(document.getElementById('wc'));
var rockSamples=0, paperSamples=0, scissorsSamples=0;
let isPredicting = false;

async function loadTFJSModel() {
  const tfjs = await tf.loadLayersModel("static/modeljs/model.json");
  return tf.model({inputs: tfjs.inputs, outputs: tfjs.output});
}

// async function train() {
//   dataset.ys = null;
//   dataset.encodeLabels(3);
//   model = tf.sequential({
//     layers: [
//       tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
//       tf.layers.dense({ units: 100, activation: 'relu'}),
//       tf.layers.dense({ units: 3, activation: 'softmax'})
//     ]
//   });
//   const optimizer = tf.train.adam(0.0001);
//   model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
//   let loss = 0;
//   model.fit(dataset.xs, dataset.ys, {
//     epochs: 10,
//     callbacks: {
//       onBatchEnd: async (batch, logs) => {
//         loss = logs.loss.toFixed(5);
//         console.log('LOSS: ' + loss);
//         }
//       }
//    });
// }


function handleButton(elem){
	switch(elem.id){
		case "0":
			rockSamples++;
			document.getElementById("rocksamples").innerText = "Rock samples:" + rockSamples;
			break;
		case "1":
			paperSamples++;
			document.getElementById("papersamples").innerText = "Paper samples:" + paperSamples;
			break;
		case "2":
			scissorsSamples++;
			document.getElementById("scissorssamples").innerText = "Scissors samples:" + scissorsSamples;
			break;
	}
	label = parseInt(elem.id);
	const img = webcam.capture();
	dataset.addExample(tfjs.predict(img), label);

}

async function predict() {
  while (isPredicting) {
    const predictedClass = tf.tidy(() => {
      const img = webcam.capture();
      const activation = tfjs.predict(img);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    var predictionText = "";
    switch(classId){
		case 0:
			predictionText = "I see Rock";
			break;
		case 1:
			predictionText = "I see Paper";
			break;
	}
	document.getElementById("prediction").innerText = predictionText;
			
    
    predictedClass.dispose();
    await tf.nextFrame();
  }
}


function doTraining(){
	train();
}

function startPredicting(){
	isPredicting = true;
	predict();
}

function stopPredicting(){
	isPredicting = false;
	predict();
}

async function init(){
	await webcam.setup();
	tfjs = await loadTFJSModel();
	tf.tidy(() => tfjs.predict(webcam.capture()));
		
}



init();