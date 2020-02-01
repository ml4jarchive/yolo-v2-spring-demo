package org.ml4j.nn.demos.yolov2.util;

import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import org.ml4j.imaging.SerializableBufferedImageAdapter;
import org.ml4j.imaging.targets.ImageDisplay;
import org.ml4j.nn.datasets.exceptions.FeatureExtractionException;
import org.ml4j.nn.datasets.featureextraction.BufferedImageFeatureExtractor;
import org.ml4j.nn.models.yolov2.BoundingBox;
import org.ml4j.nn.models.yolov2.YOLOv2Labels;
import org.springframework.boot.SpringApplication;
import org.springframework.context.ApplicationContext;

public class YoloImageDisplay extends ImageDisplay<Long> implements AutoCloseable {

	private ApplicationContext applicationContext;

	public YoloImageDisplay(ApplicationContext applicationContext, int w, int h) {
		super(w, h);
		this.applicationContext = applicationContext;
	}

	@Override
	public void close() {
		super.close();
		SpringApplication.exit(applicationContext, () -> 0);
	}

	public void displayBufferedImage(BufferedImage originalImage, List<BoundingBox> boundingBoxes,
			YOLOv2Labels yoloV2ClassificationNames) {

		BufferedImageFeatureExtractor featureExtractor = new BufferedImageFeatureExtractor(originalImage.getWidth(),
				originalImage.getHeight());

		try {

			float[] imageData = featureExtractor.getFeatures(originalImage);

			drawOneChannel(imageData, this, originalImage.getWidth(), originalImage.getHeight(), Arrays.asList(),
					Arrays.asList());

			Thread.sleep(500);

			drawOneChannel(imageData, this, originalImage.getWidth(), originalImage.getHeight(),
					boundingBoxes.stream()
							.map(box -> box.getScaledCorners(originalImage.getWidth(), originalImage.getHeight()))
							.collect(Collectors.toList()),
							boundingBoxes.stream().map(b -> yoloV2ClassificationNames.getLabel(b.getPredictedClassIndex()))
							.collect(Collectors.toList())
					);

			Thread.sleep(500);

		} catch (FeatureExtractionException e) {
			throw new RuntimeException(e);
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		}
	}

	@SuppressWarnings("deprecation")
	public static void drawOneChannel(float[] data, ImageDisplay<Long> imageDisplay, int w, int h,
			List<float[]> allBoxCorners, List<String> classifications) {
		float[] pixelData = new float[data.length];
		int depth = 1;
		
		int in = 0;
		for (int r1 = 0; r1 < h * depth; r1++) {
			for (int c = 0; c < w; c++) {
				int pixelIndex = (r1) * w + (c);
				float originalPixelValue = (data[pixelIndex]) * 255f;
				pixelData[in++] = originalPixelValue;
			}
		}
		BufferedImage img = new BufferedImage(w, h * depth, BufferedImage.TYPE_BYTE_GRAY);

		WritableRaster raster = img.getRaster();
		byte[] equiv = new byte[pixelData.length];
		for (int i = 0; i < equiv.length; i++) {
			equiv[i] = new Double(pixelData[i]).byteValue();
		}
		raster.setDataElements(0, 0, w, h * depth, equiv);
		Graphics2D graphics = img.createGraphics();
		graphics.setRenderingHint(RenderingHints.KEY_INTERPOLATION, RenderingHints.VALUE_INTERPOLATION_BILINEAR);
		graphics.drawImage(img, 0, 0, w, h * depth, 0, 0, w, h * depth, null);
		int ind = 0;
		for (float[] boxCorners : allBoxCorners) {
			float minY = boxCorners[0];
			float minX = boxCorners[1];
			float maxY = boxCorners[2];
			float maxX = boxCorners[3];
			float width = maxX - minX;
			float height = maxY - minY;
			graphics.drawString(classifications.get(ind), (int) minX, (int) minY);
			graphics.drawRect((int) minX, (int) minY, (int) width, (int) height);
			ind++;
		}

		graphics.dispose();
		imageDisplay.onFrameUpdate(new SerializableBufferedImageAdapter(img), 1000L);
	}
}
