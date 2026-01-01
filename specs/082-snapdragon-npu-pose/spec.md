# Feature Specification: Snapdragon NPU Pose Detection

## Overview

### Problem Statement
현재 flutter_pose_detection 플러그인은 Android에서 MediaPipe Tasks를 사용하여 GPU 가속만 지원합니다. Snapdragon 칩셋의 NPU(Neural Processing Unit)를 활용하지 못해 최적의 성능을 발휘하지 못하고 있습니다.

### Proposed Solution
Qualcomm AI Hub에서 제공하는 MediaPipe Pose 모델(QNN 형식)을 사용하여 Snapdragon 기기에서 NPU 가속을 지원합니다. 이를 통해 추론 시간을 현재 <25ms에서 ~1ms로 25배 개선할 수 있습니다.

### Target Users
- Snapdragon 칩셋 탑재 Android 기기 사용자
- 실시간 포즈 감지가 필요한 피트니스/헬스케어 앱 개발자
- AR/VR 애플리케이션 개발자

### Success Criteria
- Snapdragon 8 Elite 기기에서 추론 시간 2ms 미만 달성
- Snapdragon 8 Gen 1-3 기기에서 추론 시간 5ms 미만 달성
- 기존 API 호환성 100% 유지 (breaking change 없음)
- NPU 미지원 기기에서 자동으로 GPU/CPU fallback

---

## User Scenarios & Testing

### Primary User Flow
1. 개발자가 flutter_pose_detection 플러그인을 앱에 통합
2. `NpuPoseDetector().initialize()` 호출
3. Snapdragon 기기에서 자동으로 NPU 가속 활성화
4. `accelerationMode`가 `AccelerationMode.npu` 반환
5. 포즈 감지 수행 시 ~1ms 추론 시간으로 실시간 처리

### Acceptance Scenarios

#### Scenario 1: Snapdragon NPU 가속 활성화
- **Given**: 사용자가 Snapdragon 8 Elite 기기를 사용
- **When**: detector.initialize() 호출
- **Then**: accelerationMode가 AccelerationMode.npu 반환
- **And**: 추론 시간이 2ms 미만

#### Scenario 2: NPU 미지원 기기 fallback
- **Given**: 사용자가 Exynos 또는 MediaTek 칩셋 기기 사용
- **When**: detector.initialize() 호출
- **Then**: 자동으로 GPU delegate로 fallback
- **And**: accelerationMode가 AccelerationMode.gpu 반환

#### Scenario 3: 기존 API 호환성
- **Given**: 기존 flutter_pose_detection 사용 앱
- **When**: 플러그인을 새 버전으로 업데이트
- **Then**: 코드 변경 없이 정상 동작
- **And**: Snapdragon 기기에서 자동 성능 향상

### Edge Cases
- QNN 런타임 초기화 실패 시 GPU fallback
- 모델 파일 로드 실패 시 적절한 에러 메시지
- 메모리 부족 상황에서의 graceful degradation

---

## Functional Requirements

### FR-1: Qualcomm AI Hub 모델 통합
- PoseDetector 모델 (815K params, 3.14MB) 통합
- PoseLandmarkDetector 모델 (3.36M params, 12.9MB) 통합
- QNN DLC 형식 모델 파일 사용

### FR-2: QNN Delegate 구현
- Snapdragon 칩셋 감지 및 NPU 지원 확인
- QNN 런타임 초기화 및 모델 로드
- NPU → GPU → CPU 자동 fallback 체인

### FR-3: 2-Stage Pipeline 구현
- Stage 1: PoseDetector (256x256 입력) → Bounding box 출력
- Stage 2: PoseLandmarkDetector (256x256 입력) → 33 landmarks 출력
- iOS 구현과 동일한 파이프라인 구조

### FR-4: 기존 API 호환성 유지
- NpuPoseDetector 클래스 인터페이스 변경 없음
- AccelerationMode enum에 변경 없음 (npu 이미 존재)
- PoseResult, Pose, PoseLandmark 출력 형식 동일

### FR-5: 성능 모니터링
- 추론 시간 측정 및 processingTimeMs로 반환
- 실제 사용된 acceleration mode 정확히 반환

---

## Non-Functional Requirements

### Performance
- Snapdragon 8 Elite: 추론 시간 < 2ms
- Snapdragon 8 Gen 1-3: 추론 시간 < 5ms
- Snapdragon 7 시리즈: 추론 시간 < 10ms

### Compatibility
- Android API 31+ (기존과 동일)
- Snapdragon 7 Gen 4, 8 Gen 1-3, 888, 8 Elite 지원

### Reliability
- NPU 초기화 실패 시 100% GPU fallback 성공
- 연속 1000프레임 처리 시 크래시 없음

---

## Key Entities

### Models
| Model | Parameters | Size | Input | Output |
|-------|------------|------|-------|--------|
| PoseDetector | 815K | 3.14MB | 256x256 RGB | Bounding box |
| PoseLandmarkDetector | 3.36M | 12.9MB | 256x256 RGB | 33 landmarks |

### Supported Chipsets
- Snapdragon 8 Elite
- Snapdragon 8 Gen 3
- Snapdragon 8 Gen 2
- Snapdragon 8 Gen 1
- Snapdragon 888
- Snapdragon 7 Gen 4

---

## Dependencies & Assumptions

### Dependencies
- Qualcomm QNN SDK 2.41.0+
- TensorFlow Lite 2.14.0
- Qualcomm AI Hub 모델 (Apache 2.0 라이센스)

### Assumptions
- Qualcomm AI Hub 모델은 MediaPipe와 동일한 33 landmark 형식 출력
- QNN DLC 모델은 assets에 포함하여 배포 가능
- 모델 파일 크기 증가(~16MB)는 허용 가능

---

## Out of Scope
- iOS NPU 지원 (이미 CoreML로 Neural Engine 사용 중)
- Exynos NPU 지원 (Samsung NPU SDK 별도 조사 필요)
- MediaTek APU 지원 (별도 SDK 필요)
- Google Tensor TPU 지원 (별도 SDK 필요)
