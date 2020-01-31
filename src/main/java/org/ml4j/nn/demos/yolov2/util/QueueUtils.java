package org.ml4j.nn.demos.yolov2.util;

import java.util.concurrent.BlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.function.Consumer;

public class QueueUtils {

	public static <T> void startPollingQueue(
			BlockingQueue<T> queue, Consumer<T> consumer, int timeoutSeconds) {

		Runnable r = new Runnable() {
				@Override
				public void run() {
				
						T nextElement = getNextElement(queue, timeoutSeconds);
						while (nextElement != null) {
							consumer.accept(nextElement);
							nextElement = getNextElement(queue, timeoutSeconds);	
						}
				}
		};

		new Thread(r).start();
	}
	
	private static <T> T getNextElement(BlockingQueue<T> queue,  int timeoutSeconds) {
		try {
			T nextElement = queue.poll(timeoutSeconds, TimeUnit.SECONDS);
			if (nextElement == null) {
				throw new IllegalStateException("No predictions available from queue within 10 seconds - stopping polling");
			}
			return nextElement;
		} catch (InterruptedException e) {
			throw new RuntimeException(e);
		}
	}
}
