import 'dart:io';
import 'dart:math';
import 'dart:typed_data';

import 'package:flutter/services.dart';
import 'package:image/image.dart' as img;
import 'package:flutter_onnxruntime/flutter_onnxruntime.dart';

/// Stores prediction result
class PredictionResult {
  final String label;
  final double confidence;
  final int inferenceMs;
  final List<MapEntry<String, double>> topPredictions;

  PredictionResult({
    required this.label,
    required this.confidence,
    required this.inferenceMs,
    required this.topPredictions,
  });
}

class Classifier {
  late OrtSession _session;
  late List<String> _labels;
  bool _isReady = false;

  bool get isReady => _isReady;

  /// Load model + labels
  Future<void> load() async {
    final ort = OnnxRuntime();

    // Load ONNX model from assets
    _session = await ort.createSessionFromAsset(
      'assets/potato_model_fp16_mixed.onnx',
    );

    // Load labels
    final labelsRaw = await rootBundle.loadString('assets/labels.txt');

    _labels = labelsRaw
        .split('\n')
        .map((e) => e.trim())
        .where((e) => e.isNotEmpty)
        .toList();

    _isReady = true;
  }

  /// Predict image
  Future<PredictionResult> predict(String imagePath) async {
    if (!_isReady) {
      throw Exception("Call load() before predict()");
    }

    // Read image file
    final bytes = await File(imagePath).readAsBytes();

    img.Image? image = img.decodeImage(bytes);

    if (image == null) {
      throw Exception("Failed to decode image");
    }

    // Resize image
    image = img.copyResize(
      image,
      width: 224,
      height: 224,
      interpolation: img.Interpolation.linear,
    );

    // ImageNet normalization values
    const mean = [0.485, 0.456, 0.406];
    const std = [0.229, 0.224, 0.225];

    // Create input tensor [1,3,224,224]
    final inputData = Float32List(1 * 3 * 224 * 224);

    int index = 0;

    // Convert image → CHW format
    for (int c = 0; c < 3; c++) {
      for (int y = 0; y < 224; y++) {
        for (int x = 0; x < 224; x++) {
          final pixel = image.getPixel(x, y);

          double value;

          switch (c) {
            case 0:
              value = pixel.r / 255.0;
              break;
            case 1:
              value = pixel.g / 255.0;
              break;
            default:
              value = pixel.b / 255.0;
          }

          inputData[index] = (value - mean[c]) / std[c];
          index++;
        }
      }
    }

    final shape = [1, 3, 224, 224];

    // Create tensor
    final inputTensor = await OrtValue.fromList(
      inputData.toList(),
      shape,
    );

    // Run inference
    final stopwatch = Stopwatch()..start();

    final outputs = await _session.run({
      'input': inputTensor,
    });

    stopwatch.stop();

    // Get output tensor
    final outputTensor = outputs['output'];

    if (outputTensor == null) {
      throw Exception("No output received from model");
    }

    final outputData = await outputTensor.asList();

    // Convert output to logits
    final List<double> logits = (outputData[0] as List)
        .map((e) => (e as num).toDouble())
        .toList();

    // Dispose tensors
    await inputTensor.dispose();
    await outputTensor.dispose();

    // Softmax calculation
    final maxLogit = logits.reduce(max);

    final exps = logits.map((x) => exp(x - maxLogit)).toList();
    final sumExp = exps.reduce((a, b) => a + b);

    final probs = exps.map((x) => x / sumExp).toList();

    // Best prediction
    int bestIndex = 0;

    for (int i = 1; i < probs.length; i++) {
      if (probs[i] > probs[bestIndex]) {
        bestIndex = i;
      }
    }

    // Top 3 predictions
    final allPredictions = List.generate(
      probs.length,
      (i) => MapEntry(_labels[i], probs[i]),
    );

    allPredictions.sort((a, b) => b.value.compareTo(a.value));

    final top3 = allPredictions.take(3).toList();

    return PredictionResult(
      label: _labels[bestIndex],
      confidence: probs[bestIndex],
      inferenceMs: stopwatch.elapsedMilliseconds,
      topPredictions: top3,
    );
  }

  /// Release resources
  Future<void> dispose() async {
    if (_isReady) {
      await _session.close();
      _isReady = false;
    }
  }
}