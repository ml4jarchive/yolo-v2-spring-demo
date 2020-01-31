package org.ml4j;

import org.junit.jupiter.api.Test;
import org.ml4j.nn.demos.yolov2.YOLOv2Config;
import org.ml4j.nn.demos.yolov2.YOLOv2Demo;
import org.springframework.boot.test.context.SpringBootTest;

@SpringBootTest(classes= { YOLOv2Config.class, YOLOv2Demo.class })
class YOLOV2DemoTests {

	@Test
	void contextLoads() {
	}

}
