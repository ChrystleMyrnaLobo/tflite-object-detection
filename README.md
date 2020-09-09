# Tensorflow lite object detection with SSD MobileNet v2

## Inference
Inference requires the following steps
1. Prepare input:
   - resize to (300, 300) for SSD MobileNet
   - input normalization using mean and standard deviation
   - input tensor accepts NxHxWxC where n=1 for single image

2. Run Inference

3. Post process output:
   - SSD MobileNet returns 10 detection by default. The 4th output tensor indicates how many are valid.
   - For the valid detection, convert the boundary box coordinates to the scale of image size.

4. Evaluate using metrics
   - Use tensorflow metrics (models/research/object_detection/metrics) API to evaluate the detections
