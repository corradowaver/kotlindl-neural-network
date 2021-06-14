import org.jetbrains.kotlinx.dl.api.core.Sequential
import org.jetbrains.kotlinx.dl.api.core.WritingMode
import org.jetbrains.kotlinx.dl.api.core.activation.Activations
import org.jetbrains.kotlinx.dl.api.core.initializer.Constant
import org.jetbrains.kotlinx.dl.api.core.initializer.GlorotNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.HeNormal
import org.jetbrains.kotlinx.dl.api.core.initializer.Zeros
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.Conv2D
import org.jetbrains.kotlinx.dl.api.core.layer.convolutional.ConvPadding
import org.jetbrains.kotlinx.dl.api.core.layer.core.Dense
import org.jetbrains.kotlinx.dl.api.core.layer.core.Input
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.AvgPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.pooling.MaxPool2D
import org.jetbrains.kotlinx.dl.api.core.layer.reshaping.Flatten
import org.jetbrains.kotlinx.dl.api.core.loss.Losses
import org.jetbrains.kotlinx.dl.api.core.metric.Metrics
import org.jetbrains.kotlinx.dl.api.core.optimizer.Adam
import org.jetbrains.kotlinx.dl.api.inference.InferenceModel
import org.jetbrains.kotlinx.dl.dataset.fashionMnist
import org.jetbrains.kotlinx.dl.dataset.preprocessor.ImageShape
import org.jetbrains.kotlinx.dl.dataset.preprocessor.Preprocessing
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.InterpolationType
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.load
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.resize
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.rotate
import org.jetbrains.kotlinx.dl.dataset.preprocessor.image.save
import org.jetbrains.kotlinx.dl.dataset.preprocessor.preprocess
import org.jetbrains.kotlinx.dl.dataset.preprocessor.rescale
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformImage
import org.jetbrains.kotlinx.dl.dataset.preprocessor.transformTensor
import java.io.File
import javax.imageio.ImageIO
import kotlin.math.roundToInt


val stringLabels = mapOf(
  0 to "T-shirt/top",
  1 to "Trousers",
  2 to "Pullover",
  3 to "Dress",
  4 to "Coat",
  5 to "Sandals",
  6 to "Shirt",
  7 to "Sneakers",
  8 to "Bag",
  9 to "Ankle boots"
)

private const val EPOCHS = 2
private const val TRAINING_BATCH_SIZE = 100
private const val TEST_BATCH_SIZE = 1000
private const val NUM_LABELS = 10
private const val NUM_CHANNELS = 1L
private const val IMAGE_SIZE = 28L
private const val SEED = 12L

private val heNormal = HeNormal(SEED)

private val vgg11 = Sequential.of(
  Input(
    IMAGE_SIZE,
    IMAGE_SIZE,
    NUM_CHANNELS
  ),
  Conv2D(
    filters = 32,
    kernelSize = longArrayOf(3, 3),
    strides = longArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = heNormal,
    biasInitializer = heNormal,
    padding = ConvPadding.SAME
  ),
  MaxPool2D(
    poolSize = intArrayOf(1, 2, 2, 1),
    strides = intArrayOf(1, 2, 2, 1),
    padding = ConvPadding.SAME
  ),
  Conv2D(
    filters = 64,
    kernelSize = longArrayOf(3, 3),
    strides = longArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = heNormal,
    biasInitializer = heNormal,
    padding = ConvPadding.SAME
  ),
  MaxPool2D(
    poolSize = intArrayOf(1, 2, 2, 1),
    strides = intArrayOf(1, 2, 2, 1),
    padding = ConvPadding.SAME
  ),
  Conv2D(
    filters = 128,
    kernelSize = longArrayOf(3, 3),
    strides = longArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = heNormal,
    biasInitializer = heNormal,
    padding = ConvPadding.SAME
  ),
  Conv2D(
    filters = 128,
    kernelSize = longArrayOf(3, 3),
    strides = longArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = heNormal,
    biasInitializer = heNormal,
    padding = ConvPadding.SAME
  ),
  MaxPool2D(
    poolSize = intArrayOf(1, 2, 2, 1),
    strides = intArrayOf(1, 2, 2, 1),
    padding = ConvPadding.SAME
  ),
  Conv2D(
    filters = 256,
    kernelSize = longArrayOf(3, 3),
    strides = longArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = heNormal,
    biasInitializer = heNormal,
    padding = ConvPadding.SAME
  ),
  Conv2D(
    filters = 256,
    kernelSize = longArrayOf(3, 3),
    strides = longArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = heNormal,
    biasInitializer = heNormal,
    padding = ConvPadding.SAME
  ),
  MaxPool2D(
    poolSize = intArrayOf(1, 2, 2, 1),
    strides = intArrayOf(1, 2, 2, 1),
    padding = ConvPadding.SAME
  ),
  Conv2D(
    filters = 128,
    kernelSize = longArrayOf(3, 3),
    strides = longArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = heNormal,
    biasInitializer = heNormal,
    padding = ConvPadding.SAME
  ),
  Conv2D(
    filters = 128,
    kernelSize = longArrayOf(3, 3),
    strides = longArrayOf(1, 1, 1, 1),
    activation = Activations.Relu,
    kernelInitializer = heNormal,
    biasInitializer = heNormal,
    padding = ConvPadding.SAME
  ),
  MaxPool2D(
    poolSize = intArrayOf(1, 2, 2, 1),
    strides = intArrayOf(1, 2, 2, 1),
    padding = ConvPadding.SAME
  ),
  Flatten(),
  Dense(
    outputSize = 2048,
    activation = Activations.Relu,
    kernelInitializer = heNormal,
    biasInitializer = heNormal
  ),
  Dense(
    outputSize = 1000,
    activation = Activations.Relu,
    kernelInitializer = heNormal,
    biasInitializer = heNormal
  ),
  Dense(
    outputSize = NUM_LABELS,
    activation = Activations.Linear,
    kernelInitializer = heNormal,
    biasInitializer = heNormal
  )
)

const val imageName = "tshirt.jpg"

fun train() {
  val (train, test) = fashionMnist()

  vgg11.compile(
    optimizer = Adam(),
    loss = Losses.SOFT_MAX_CROSS_ENTROPY_WITH_LOGITS,
    metric = Metrics.ACCURACY
  )

  vgg11.summary()

  vgg11.fit(dataset = train, epochs = EPOCHS, batchSize = TRAINING_BATCH_SIZE)

  val accuracy = vgg11.evaluate(dataset = test, batchSize = TEST_BATCH_SIZE).metrics[Metrics.ACCURACY]

  println("Accuracy: $accuracy")
  vgg11.save(File("model/my_model"), writingMode = WritingMode.OVERRIDE)

  vgg11.close()
}

fun predict(imageList: FloatArray, imageClass: Float) {
  val model = InferenceModel.load(File("model/my_model"))

  model.reshape(28, 28, 1)
  val prediction = model.predict(imageList)
  println("Predicted label is: $prediction. This corresponds to class ${stringLabels[prediction]}.")
  println("Actual label is: $imageClass.")

}

fun main() {
 // train()

  preprocessImage()
  val grayArray = makeGray()
  val imageClass = 6f
  predict(grayArray, imageClass)
}

fun makeGray(): FloatArray {
  val input =
    File("C:\\Users\\corra\\Documents\\GitHub\\neural-kotlindl\\src\\main\\resources\\preprocessed\\$imageName")
  val image = ImageIO.read(input)
  val grayArray = Array(28) { Array(28) { 0f } }
  for (y in 0 until image.height)
    for (x in 0 until image.width) {
      val rgb: Int = image.getRGB(x, y)
      val gray = (rgb and 0xFF) / 255f
      grayArray[y][x] = gray
    }

  for (y in 0 until image.height)
    for (x in 0 until image.width) {
      image.setRGB(x, y, toRGB(grayArray[y][x]))
    }

  val output =
    File("C:\\Users\\corra\\Documents\\GitHub\\neural-kotlindl\\src\\main\\resources\\grayscaled\\$imageName")
  ImageIO.write(image, "jpg", output)
  grayArray.forEach { row ->
    row.forEach {
      when (it) {
        in 0.9f..1f -> print("■ ")
        in 0.6f..0.9f -> print("# ")
        in 0.4f..0.6f -> print("* ")
        in 0.2f..0.4f -> print("- ")
        in 0f..0.2f -> print(". ")
      }
    }
    println()
  }
  return grayArray.flatten().toFloatArray()
}

fun preprocessImage(): FloatArray {
  val input =
    File("C:\\Users\\corra\\Documents\\GitHub\\neural-kotlindl\\src\\main\\resources\\raw\\$imageName")
  val image = ImageIO.read(input)
  val preprocessedImagesDirectory =
    File("C:\\Users\\corra\\Documents\\GitHub\\neural-kotlindl\\src\\main\\resources\\preprocessed")

  val preprocessing: Preprocessing = preprocess {
    transformImage {
      load {
        pathToData = input
        imageShape = ImageShape(image.width.toLong(), image.height.toLong(), 3)
      }
      rotate {
        degrees = 0f
      }
      resize {
        outputWidth = 28
        outputHeight = 28
        interpolation = InterpolationType.NEAREST
      }
      save {
        dirLocation = preprocessedImagesDirectory
      }
    }
    transformTensor {
      rescale {
        scalingCoefficient = 255f
      }
    }
  }
  println(preprocessing.finalShape.channels)
  return preprocessing().first
}

fun toRGB(value: Float): Int {
  val part = (value * 255).roundToInt()
  return part * 0x10101
}
