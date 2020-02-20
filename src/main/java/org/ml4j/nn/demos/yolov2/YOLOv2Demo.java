/*
 * Copyright 2020 the original author or authors.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"); you may not use this file except
 * in compliance with the License. You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software distributed under the License
 * is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
 * or implied. See the License for the specific language governing permissions and limitations under
 * the License.
 */
package org.ml4j.nn.demos.yolov2;

import java.awt.image.BufferedImage;
import java.util.List;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.BlockingQueue;
import java.util.function.Supplier;
import java.util.logging.LogManager;
import java.util.stream.IntStream;
import java.util.stream.Stream;

import org.ml4j.Matrix;
import org.ml4j.MatrixFactory;
import org.ml4j.images.Image;
import org.ml4j.nn.FeedForwardNeuralNetworkContext;
import org.ml4j.nn.ForwardPropagation;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.onetoone.DefaultChainableDirectedComponentAdapter;
import org.ml4j.nn.datasets.BatchedLabeledDataSet;
import org.ml4j.nn.datasets.DataBatch;
import org.ml4j.nn.datasets.FeatureExtractionErrorMode;
import org.ml4j.nn.datasets.LabeledData;
import org.ml4j.nn.datasets.LabeledDataImpl;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;
import org.ml4j.nn.datasets.featureextraction.LabeledImageSupplierFeatureExtractor;
import org.ml4j.nn.datasets.images.LabeledImagesDataSet;
import org.ml4j.nn.demos.yolov2.util.QueueUtils;
import org.ml4j.nn.demos.yolov2.util.YoloImageDisplay;
import org.ml4j.nn.models.yolov2.BoundingBox;
import org.ml4j.nn.models.yolov2.BoundingBoxExtractor;
import org.ml4j.nn.models.yolov2.YOLOv2Factory;
import org.ml4j.nn.models.yolov2.YOLOv2Labels;
import org.ml4j.nn.neurons.NeuronsActivation;
import org.ml4j.nn.neurons.format.ImageNeuronsActivationFormat;
import org.ml4j.nn.supervised.FeedForwardNeuralNetworkContextImpl;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetwork;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.CommandLineRunner;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.context.ApplicationContext;

import com.codepoetics.protonpack.StreamUtils;

/**
 * @author Michael Lavelle
 */
@SpringBootApplication
public class YOLOv2Demo implements CommandLineRunner {

	private static final Logger LOGGER = LoggerFactory.getLogger(YOLOv2Demo.class);

	// Set the thresholds for displaying bounding boxes ( SCORE_THRESHOLD) and to determine overlap constraints ( IOU_THRESHOLD ).
	private final static float SCORE_THRESHOLD = 0.4f;
	private final static float IOU_THRESHOLD = 0.6f;
	
	private final static int BATCH_SIZE = 1;
	
	private final static int BATCH_PREDICTION_TIMEOUT_SECONDS = 15;

	@Autowired
	private MatrixFactory matrixFactory;

	@Autowired
	private YOLOv2Factory yoloV2Factory;

	@Autowired
	private BoundingBoxExtractor boundingBoxExtractor;
		
	@Autowired
	private YOLOv2Labels yoloV2ClassificationNames;
	
	@Autowired
	private ApplicationContext applicationContext;
	
	@Autowired
	private LabeledImagesDataSet<Supplier<BufferedImage>> dataSet;
	
	@Autowired
	private DirectedComponentsContext directedComponentsContext;

	public static void main(String[] args) throws Exception {

		System.setProperty("java.awt.headless", "false");

		// Quieten Logging
		org.jblas.util.Logger.getLogger().setLevel(org.jblas.util.Logger.ERROR);
		LogManager.getLogManager().reset();

		// Application entry-point
		SpringApplication.run(YOLOv2Demo.class, args);
	
	}

	@Override
	public void run(String... args) throws Exception {
		
		try (YoloImageDisplay display = new YoloImageDisplay(applicationContext, 1280, 720)) {
								
			LOGGER.info("Creating the YOLO v2 network and loading weights...");
			
			// Create a runtime (non-training) context for the Yolo V2 Network
			FeedForwardNeuralNetworkContext predictionContext = new FeedForwardNeuralNetworkContextImpl(directedComponentsContext, false);

			// Create the YOLO v2 Network, configuring the prediction context
			SupervisedFeedForwardNeuralNetwork yoloV2Network = yoloV2Factory.createYoloV2(predictionContext);
			
			LOGGER.info("Finished creating the YOLO v2 network and loading weights");
		
			// Map the data set to of a data set of image batches with BATCH_SIZE in each batch
			BatchedLabeledDataSet<Supplier<Image>, Supplier<BufferedImage>> batchedImageSupplierDataSet = dataSet
					.toBatchedLabeledDataSet(BATCH_SIZE);

			// Map the batches to a stream of input NeuronsActivations, along with the corresponding BufferedImages.
			Stream<LabeledData<NeuronsActivation, Stream<BufferedImage>>> labeledNeuronsActivationInputStream = batchedImageSupplierDataSet
					.stream().map(e -> toLabeledNeuronsActivation(e));

			// Obtain the output predictions stream from the input stream
			Stream<LabeledData<ForwardPropagation, Stream<BufferedImage>>> forwardPropagationsOutputStream = yoloV2Network
					.forwardPropagateWithLabels(labeledNeuronsActivationInputStream, predictionContext);
		
			// Map to an output stream of BufferedImages with the corresponding bounding box predictions ( float arrays)		
			Stream<LabeledData<BufferedImage, float[]>> predictions = forwardPropagationsOutputStream.flatMap(e -> 
				toRawPredictionsStreamForEachExample(e.getData(), e.getLabel()));

			// Map to an bounding box float arrays to a list of BoundingBox objects
			Stream<LabeledData<BufferedImage, List<BoundingBox>>> predictionsWithBoundingBoxes = 
						predictions.map(p -> new LabeledDataImpl<>(p.getData(), 
								boundingBoxExtractor
						.getScoreFilteredBoundingBoxesWithNonMaxSuppression(p.getLabel(),
								yoloV2ClassificationNames, SCORE_THRESHOLD, IOU_THRESHOLD)));
						
			// Create a blocking queue to store our predictions as they are produced.
			BlockingQueue<LabeledData<BufferedImage, List<BoundingBox>>> displayQueue = new ArrayBlockingQueue<>(20);
			
			// Set up polling on the queue to display each buffered image as the prediction is added to the queue..
			QueueUtils.startPollingQueue(displayQueue, (e -> 
				display.displayBufferedImage(e.getData(), e.getLabel(), yoloV2ClassificationNames)), BATCH_PREDICTION_TIMEOUT_SECONDS);
			
			// When a prediction is produced, add to the queue.
			predictionsWithBoundingBoxes.forEach(p -> displayQueue.add(p));
			
			DefaultChainableDirectedComponentAdapter.printTimes();

		} 
	}
	
	private Stream<LabeledData<BufferedImage, float[]>> toRawPredictionsStreamForEachExample(ForwardPropagation forwardPropagation, Stream<BufferedImage> bufferedImages) {
		Matrix activationsMatrix = forwardPropagation.getOutput().getActivations(matrixFactory);
		Stream<Matrix> examplesStream = IntStream.range(0, forwardPropagation.getOutput().getExampleCount())
				.mapToObj(i -> { Matrix example = activationsMatrix.getColumn(i); example.asEditableMatrix().reshape(425, 19 * 19); return example; } );
		return StreamUtils.zip(bufferedImages, examplesStream.map(m -> m.transpose()).map(m -> m.getRowByRowArray()), (l, r) -> new LabeledDataImpl<>(l, r));

	}

	private LabeledDataImpl<NeuronsActivation, Stream<BufferedImage>> toLabeledNeuronsActivation(
			DataBatch<LabeledData<Supplier<Image>, Supplier<BufferedImage>>> dataBatch) {

		try {
			return new LabeledDataImpl<NeuronsActivation, Stream<BufferedImage>>(
					dataBatch.toFloatArrayDataBatch(new LabeledImageSupplierFeatureExtractor<>(608 * 608 * 3),
							FeatureExtractionErrorMode.RAISE_EXCEPTION).toNeuronsActivation(matrixFactory,
									ImageNeuronsActivationFormat.ML4J_DEFAULT_IMAGE_FORMAT),
					dataBatch.stream().map(i -> i.getLabel().get()));

		} catch (FeatureExtractionException a) {
			throw new RuntimeException(a);
		}
	}
}
