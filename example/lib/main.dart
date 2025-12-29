import 'package:flutter/material.dart';
import 'package:flutter_pose_detection/flutter_pose_detection.dart';

import 'pages/camera_detection_page.dart';
import 'pages/image_detection_page.dart';
import 'pages/video_analysis_page.dart';

void main() {
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'NPU Pose Detection Demo',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      home: const HomePage(),
    );
  }
}

class HomePage extends StatefulWidget {
  const HomePage({super.key});

  @override
  State<HomePage> createState() => _HomePageState();
}

class _HomePageState extends State<HomePage> {
  String _accelerationMode = 'Not initialized';
  bool _isInitializing = false;
  NpuPoseDetector? _detector;

  @override
  void initState() {
    super.initState();
    _initializeDetector();
  }

  Future<void> _initializeDetector() async {
    setState(() {
      _isInitializing = true;
    });

    try {
      _detector = NpuPoseDetector();
      final mode = await _detector!.initialize();

      if (mounted) {
        setState(() {
          _accelerationMode = mode.name.toUpperCase();
          _isInitializing = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _accelerationMode = 'Error: $e';
          _isInitializing = false;
        });
      }
    }
  }

  @override
  void dispose() {
    _detector?.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('NPU Pose Detection'),
        backgroundColor: Theme.of(context).colorScheme.inversePrimary,
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Card(
              child: Padding(
                padding: const EdgeInsets.all(16.0),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Status',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 8),
                    Row(
                      children: [
                        const Text('Acceleration Mode: '),
                        if (_isInitializing)
                          const SizedBox(
                            width: 16,
                            height: 16,
                            child: CircularProgressIndicator(strokeWidth: 2),
                          )
                        else
                          Text(
                            _accelerationMode,
                            style: TextStyle(
                              fontWeight: FontWeight.bold,
                              color: _accelerationMode.contains('Error')
                                  ? Colors.red
                                  : Colors.green,
                            ),
                          ),
                      ],
                    ),
                  ],
                ),
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              'Detection Modes',
              style: TextStyle(
                fontSize: 18,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            _buildFeatureCard(
              context,
              icon: Icons.image,
              title: 'Image Detection',
              description: 'Detect poses in static images',
              onTap: _detector != null
                  ? () => Navigator.push(
                        context,
                        MaterialPageRoute(
                          builder: (context) =>
                              ImageDetectionPage(detector: _detector!),
                        ),
                      )
                  : null,
            ),
            const SizedBox(height: 12),
            _buildFeatureCard(
              context,
              icon: Icons.videocam,
              title: 'Camera Detection',
              description: 'Real-time pose detection from camera',
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => const CameraDetectionPage(),
                ),
              ),
            ),
            const SizedBox(height: 12),
            _buildFeatureCard(
              context,
              icon: Icons.movie,
              title: 'Video Analysis',
              description: 'Analyze poses in video files',
              onTap: () => Navigator.push(
                context,
                MaterialPageRoute(
                  builder: (context) => const VideoAnalysisPage(),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFeatureCard(
    BuildContext context, {
    required IconData icon,
    required String title,
    required String description,
    required VoidCallback? onTap,
    bool enabled = true,
  }) {
    return Card(
      child: ListTile(
        leading: Icon(
          icon,
          size: 40,
          color: enabled
              ? Theme.of(context).colorScheme.primary
              : Colors.grey,
        ),
        title: Text(
          title,
          style: TextStyle(
            fontWeight: FontWeight.bold,
            color: enabled ? null : Colors.grey,
          ),
        ),
        subtitle: Text(
          description,
          style: TextStyle(
            color: enabled ? null : Colors.grey,
          ),
        ),
        trailing: enabled
            ? const Icon(Icons.chevron_right)
            : const Text('Coming Soon', style: TextStyle(color: Colors.grey)),
        onTap: onTap,
        enabled: enabled,
      ),
    );
  }
}
