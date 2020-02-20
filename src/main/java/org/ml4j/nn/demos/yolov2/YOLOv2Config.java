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
import java.io.File;
import java.io.IOException;
import java.nio.file.Path;
import java.util.function.Supplier;

import org.ml4j.MatrixFactory;
import org.ml4j.jblas.JBlasRowMajorMatrixFactoryOptimised;
import org.ml4j.nn.activationfunctions.ActivationFunctionBaseType;
import org.ml4j.nn.activationfunctions.ActivationFunctionProperties;
import org.ml4j.nn.activationfunctions.ActivationFunctionType;
import org.ml4j.nn.activationfunctions.factories.DifferentiableActivationFunctionFactory;
import org.ml4j.nn.axons.factories.AxonsFactory;
import org.ml4j.nn.components.DirectedComponentsContext;
import org.ml4j.nn.components.DirectedComponentsContextImpl;
import org.ml4j.nn.components.factories.DirectedComponentFactory;
import org.ml4j.nn.components.factories.DirectedComponentFactoryAdapter;
import org.ml4j.nn.datasets.images.DirectoryImagesWithBufferedImagesDataSet;
import org.ml4j.nn.datasets.images.LabeledImagesDataSet;
import org.ml4j.nn.factories.DefaultAxonsFactoryImpl;
import org.ml4j.nn.factories.DefaultDifferentiableActivationFunctionFactory;
import org.ml4j.nn.factories.DefaultDirectedComponentFactoryImpl;
import org.ml4j.nn.models.yolov2.BoundingBoxExtractor;
import org.ml4j.nn.models.yolov2.YOLOv2Factory;
import org.ml4j.nn.models.yolov2.YOLOv2Labels;
import org.ml4j.nn.models.yolov2.impl.DefaultYOLOv2BoundingBoxExtractor;
import org.ml4j.nn.models.yolov2.impl.DefaultYOLOv2Factory;
import org.ml4j.nn.sessions.factories.DefaultSessionFactory;
import org.ml4j.nn.sessions.factories.DefaultSessionFactoryImpl;
import org.ml4j.nn.supervised.DefaultSupervisedFeedForwardNeuralNetworkFactory;
import org.ml4j.nn.supervised.SupervisedFeedForwardNeuralNetworkFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author Michael Lavelle
 */
@Configuration
public class YOLOv2Config {

	@Bean
	MatrixFactory matrixFactory() {
		return new JBlasRowMajorMatrixFactoryOptimised();
	}

	@Bean
	AxonsFactory axonsFactory() {
		return new DefaultAxonsFactoryImpl(matrixFactory());
	}
	
	@Bean
	DirectedComponentFactory directedComponentFactory() {
		
		
		DefaultDirectedComponentFactoryImpl factory = new DefaultDirectedComponentFactoryImpl(matrixFactory(), axonsFactory(), 
				activationFunctionFactory(), directedComponentsContext());
		
		DirectedComponentFactoryAdapter adapter = new DirectedComponentFactoryAdapter(factory);
		factory.setDirectedComponentFactory(adapter);
		
		return adapter;

	}
	
	@Bean
	DirectedComponentsContext directedComponentsContext() {
		return new DirectedComponentsContextImpl(matrixFactory(), false);
	}

	@Bean
	DefaultSessionFactory sessionFactory() {
		// We don't require a DirectedLayerFactory for this demo - set null in constructor
		return new DefaultSessionFactoryImpl(matrixFactory(), directedComponentFactory(), null, supervisedFeedForwardNeuralNetworkFactory(), null);
	}

	@Bean
	DifferentiableActivationFunctionFactory activationFunctionFactory() {
		return new DefaultDifferentiableActivationFunctionFactory();
	}

	@Bean
	SupervisedFeedForwardNeuralNetworkFactory supervisedFeedForwardNeuralNetworkFactory() {
		return new DefaultSupervisedFeedForwardNeuralNetworkFactory(directedComponentFactory());
	}
	
	@Bean
	LabeledImagesDataSet<Supplier<BufferedImage>> dataSet() {
		
		// Define images Directory
		Path imagesDirectory = new File(YOLOv2Demo.class.getClassLoader()
				.getResource("test_images").getFile()).toPath();

		// Define data set of scaled images (608 * 608) from a directory labelled with
		// the original buffered images
		return new DirectoryImagesWithBufferedImagesDataSet(
				imagesDirectory, path -> true, 608, 608);
	}
	
	@Bean
	YOLOv2Factory yoloV2Factory() throws IOException {
		return new DefaultYOLOv2Factory(sessionFactory(), matrixFactory(), YOLOv2Demo.class.getClassLoader());
	}
	
	@Bean
	YOLOv2Labels yoloV2ClassificationNames() throws IOException {
		return yoloV2Factory().createYoloV2Labels();
	}
	
	@Bean
	BoundingBoxExtractor boundingBoxExtractor() {
		return new DefaultYOLOv2BoundingBoxExtractor(matrixFactory(), activationFunctionFactory().createActivationFunction(
				ActivationFunctionType.getBaseType(ActivationFunctionBaseType.SOFTMAX), 
				new ActivationFunctionProperties()));
	}
}
