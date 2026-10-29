// Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

package com.ultralytics.yolo

import android.content.Context
import android.graphics.Bitmap
import android.graphics.RectF
import android.util.Log

/**
 * High-performance ObjectDetector on LiteRT 2.x ([LiteRtModel]).
 * Detects only the configured allowed classes.
 */
class ObjectDetector(
    context: Context,
    modelPath: String,
    override var labels: List<String>,
    private val useGpu: Boolean = true,
    private var numItemsThreshold: Int = 30
) : BasePredictor() {

    // Only these classes are returned by the detector.
    private val allowedClasses = setOf("person", "tie")

    // Inference output dimensions.
    private var out1 = 0
    private var out2 = 0

    // Reusable preprocessing buffers.
    private lateinit var scaledBitmap: Bitmap
    private lateinit var intValues: IntArray
    private lateinit var floatInput: FloatArray

    // Thresholds.
    private var confidenceThreshold = 0.25f
    private var iouThreshold = 0.7f

    init {
        val loadedLabels = YOLOFileUtils.loadModelLabels(context, modelPath)

        if (loadedLabels != null) {
            this.labels = loadedLabels
            Log.i(TAG, "Labels loaded from model metadata.")
        } else if (this.labels.isEmpty()) {
            Log.w(
                TAG,
                "No embedded labels found and none provided; detections may lack class names."
            )
        }

        rtModel = InferenceModel.create(
            context,
            modelPath,
            useGpu,
            "ObjectDetector"
        )

        // Input dimensions: [1, H, W, 3].
        val inDims = rtModel.inputDims
        val inHeight = if (inDims.size >= 4) inDims[1] else 640
        val inWidth = if (inDims.size >= 4) inDims[2] else 640

        inputSize = Size(inWidth, inHeight)
        modelInputSize = Pair(inWidth, inHeight)

        // Output dimensions.
        val outDims = rtModel.outputDims.getOrNull(0) ?: IntArray(0)

        if (outDims.size >= 3) {
            out1 = outDims[1]
            out2 = outDims[2]
        } else {
            val count = rtModel.outputElementCounts.getOrElse(0) { 0 }
            val features = labels.size + 4

            if (features in 1..count && count % features == 0) {
                out1 = features
                out2 = count / features
            }
        }

        initPreprocessingResources(inWidth, inHeight)
    }

    private fun initPreprocessingResources(width: Int, height: Int) {
        scaledBitmap = Bitmap.createBitmap(
            width,
            height,
            Bitmap.Config.ARGB_8888
        )
        intValues = IntArray(width * height)
        floatInput = FloatArray(width * height * 3)
    }

    override fun predict(
        bitmap: Bitmap,
        origWidth: Int,
        origHeight: Int,
        rotateForCamera: Boolean,
        isLandscape: Boolean
    ): YOLOResult {
        val overallStartTime = System.nanoTime()
        t0 = overallStartTime

        // Preprocess image.
        ImageUtils.prepareBitmapForModel(
            bitmap = bitmap,
            targetBitmap = scaledBitmap,
            rotateForCamera = rotateForCamera,
            isLandscape = isLandscape,
            isFrontCamera = isFrontCamera,
            rotationDegrees = cameraRotationDegrees
        )

        ImageUtils.copyRgbBitmapToFloatArray(
            scaledBitmap,
            floatInput,
            intValues,
            INPUT_MEAN,
            INPUT_STANDARD_DEVIATION,
            channelsFirst = rtModel.inputUsesNchw
        )

        // Run inference.
        val preEnd = System.nanoTime()
        val flat = rtModel.run(floatInput)[0]
        val inferEnd = System.nanoTime()

        // Postprocess model output.
        val resultBoxes = if (out2 < out1 && out2 >= 6) {
            postprocessEndToEnd(
                flat,
                rows = out1,
                rowLen = out2
            )
        } else {
            val classCount = (out1 - 4).coerceAtLeast(0)

            postprocess(
                flat,
                w = out2,
                h = out1,
                confidenceThreshold = confidenceThreshold,
                iouThreshold = iouThreshold,
                numItemsThreshold = numItemsThreshold,
                numClasses = classCount
            )
        }

        // Convert allowed detections to Box objects.
        val boxes = mutableListOf<Box>()

        for (boxArray in resultBoxes) {
            if (boxArray.size < 6) continue

            val classIdx = boxArray[5].toInt()
            val label = labelName(classIdx).trim()

            if (!isAllowedClass(label)) continue

            val rect = inputRectFromOutputRect(
                RectF(
                    boxArray[0],
                    boxArray[1],
                    boxArray[0] + boxArray[2],
                    boxArray[1] + boxArray[3]
                ),
                origWidth,
                origHeight
            )

            if (rect != null) {
                val normRect = normalizedRectFromInputRect(
                    rect,
                    origWidth,
                    origHeight
                )

                boxes.add(
                    Box(
                        classIdx,
                        label,
                        boxArray[4],
                        rect,
                        normRect
                    )
                )
            }
        }

        val timing = finishTiming(preEnd, inferEnd)

        return YOLOResult(
            origShape = Size(origWidth, origHeight),
            boxes = boxes,
            speed = timing.speedMs,
            fps = timing.fps,
            preMs = timing.preMs,
            inferenceMs = timing.inferenceMs,
            postMs = timing.postMs,
            names = labels
        )
    }

    private fun isAllowedClass(label: String): Boolean {
        return label.trim().lowercase() in allowedClasses
    }

    override fun setConfidenceThreshold(conf: Double) {
        confidenceThreshold = conf.toFloat()
        super.setConfidenceThreshold(conf)
    }

    override fun setIouThreshold(iou: Double) {
        iouThreshold = iou.toFloat()
        super.setIouThreshold(iou)
    }

    override fun getConfidenceThreshold(): Double {
        return confidenceThreshold.toDouble()
    }

    override fun getIouThreshold(): Double {
        return iouThreshold.toDouble()
    }

    override fun setNumItemsThreshold(n: Int) {
        numItemsThreshold = n
        super.setNumItemsThreshold(n)
    }

    // Native postprocessing for standard YOLO output.
    private external fun postprocess(
        predictions: FloatArray,
        w: Int,
        h: Int,
        confidenceThreshold: Float,
        iouThreshold: Float,
        numItemsThreshold: Int,
        numClasses: Int
    ): Array<FloatArray>

    /**
     * Postprocess end-to-end output.
     * Filter allowed classes before applying the result limit.
     */
    private fun postprocessEndToEnd(
        flat: FloatArray,
        rows: Int,
        rowLen: Int
    ): Array<FloatArray> {
        val boxes = mutableListOf<FloatArray>()

        for (i in 0 until rows) {
            val base = i * rowLen

            // Avoid reading beyond the output buffer.
            if (base < 0 || base + 5 >= flat.size) continue

            val confidence = flat[base + 4]
            if (confidence <= confidenceThreshold) continue

            val classIdx = flat[base + 5].toInt()
            val label = labelName(classIdx).trim()

            // Discard classes before they consume the output limit.
            if (!isAllowedClass(label)) continue

            val x1 = flat[base]
            val y1 = flat[base + 1]
            val x2 = flat[base + 2]
            val y2 = flat[base + 3]

            boxes.add(
                floatArrayOf(
                    x1,
                    y1,
                    x2 - x1,
                    y2 - y1,
                    confidence,
                    classIdx.toFloat()
                )
            )

            if (boxes.size >= numItemsThreshold) break
        }

        return boxes.toTypedArray()
    }

    companion object {
        private const val TAG = "ObjectDetector"

        init {
            System.loadLibrary("ultralytics")
        }

        private const val INPUT_MEAN = 0f
        private const val INPUT_STANDARD_DEVIATION = 255f
    }
}
