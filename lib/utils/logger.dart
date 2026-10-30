import 'package:flutter/foundation.dart';

void logInfo(String message) {
  if (kDebugMode) {
    debugPrint('[YOLO DEBUG] $message');
  }
}
