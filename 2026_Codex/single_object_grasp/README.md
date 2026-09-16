# Single-object grasp data generation

이 폴더는 만들어진 `conf`와 `pre_grasp`를 Isaac Lab 물리에서 검증해
`output_grasp`를 생성하고, 장면별 결과를 object zero pose로 병합한다.
지원대(platform)는 사용하지 않는다.

`conf`와 `pre_grasp` 생성기는 다음 위치에 있다.

```text
/home/uon/ochansol/isaac_code/python/sanjabu/2026/grasp_data_gen_for_isaaclab/
├── scene_gen.py
└── pregrasp/
    ├── scene_pregrasp.py
    ├── finger_sampler.py
    ├── hand_heightmap.py
    └── generate_hand_heightmaps.py
```

`scene_gen.py`는 기존 방식처럼 Isaac Sim top-view pointcloud를 내부에서 받아
1,500점만 샘플링해 pre-grasp를 계산한다. PCD 파일이나 카메라 이미지는 저장하지
않고, 물체 자세별 `conf`와 `pre_grasp` JSON만 저장한다.

## 데이터 폴더 구조

```text
<output-root>/
├── <object>/
│   └── <gripper>/
│       ├── conf/
│       │   └── 0000.json
│       ├── pre_grasp/
│       │   └── 0000.json
│       ├── output_grasp/
│       │   └── 0000.json
│       └── generation_summary.json
└── _batch_summaries/
    └── <gripper>.json
```

즉 데이터 루트 순서는 `물체/그리퍼 종류/데이터 종류`이다.

## 1. conf + pre-grasp 일괄 생성

모델 루트 아래의 `<object>/edited/<object>.usd`를 전부 찾아
물체별 자세 0~100의 `conf`와 `pre_grasp`를 만든다.

먼저 `scene_gen.py` 위쪽의 실행 설정을 수정한다.

```python
OBJECT_ROOT = Path("/nas/ochansol/3d_model/2026_real_world_test_objects")
OUTPUT_ROOT = Path("/nas/Dataset/Dataset_2026/isaacsim_grasp_data_gen")
GRIPPER = "Robotiq_2f140"
OBJECTS = None  # 전체 물체. 일부만 할 때는 ["black_pepper_shaker"]
SCENE_START = 0
SCENE_END = 100
PCD_SAMPLE_COUNT = 1_500  # 내부 계산에만 사용, 파일 저장 안 함
PCD_CANDIDATE_DISTANCE = 0.01
SIM_DT = 0.001
RENDER_DT = 0.005
OVERWRITE = False
```

인자 없이 실행한다.

```bash
cd /home/uon/ochansol/isaac_code/python/sanjabu/2026/grasp_data_gen_for_isaaclab
/home/uon/ochansol/isaaclab_232/.venv/bin/python scene_gen.py
```

다른 그리퍼 데이터는 `GRIPPER = "Inspire-F1_right"`처럼 바꿔 별도 실행하면
같은 물체 아래에 그리퍼별로 구분된다.

완성된 `conf/####.json`과 `pre_grasp/####.json` 쌍은 자동으로 건너뛰므로
중간에 멈춰도 같은 명령으로 이어서 실행할 수 있다. 다시 만들 때만
`OVERWRITE = True`로 바꾼다. 실패한 자세는 기본 5회까지 다른 물체 회전으로
재시도하고 다음 물체로 계속 진행한다.

핸드 그리퍼는 각 자세에서 접근 충돌과 접촉 가능성을 검사할 때 gripper
heightmap을 사용한다. 아직 heightmap이 없다면 최초 한 번 Isaac Sim에서 만든다.

```bash
/home/uon/ochansol/isaaclab_232/.venv/bin/python \
  pregrasp/generate_hand_heightmaps.py
```

경로와 해상도는 `generate_hand_heightmaps.py` 상단의 `GRIPPER_INFO`,
`OUTPUT_DIR`, `RESOLUTION`에서 설정한다.

## 2. grasp 물리 검증

```bash
cd /home/uon/ochansol/isaaclab_232/2026_Codex/single_object_grasp
/home/uon/ochansol/isaaclab_232/.venv/bin/python collect_grasps.py
```

데이터 경로와 scene 번호 등은 `collect_grasps.py` 상단의 `ROOT`, `SCENE`,
`ENVS`, `HEADLESS`에서 설정한다. 기본 병렬 수는 기존 수집기와 같은 20개다. 물리 timestep과 action 주기는 각각
`SIM_DT`, `DECIMATION`으로 설정하며 실제 policy 주기는
`SIM_DT * DECIMATION`이다.

관통 억제/검사는 같은 설정부의 `ENABLE_CCD`,
`PHYSX_SOLVE_ARTICULATION_CONTACT_LAST`, `CONTACT_OFFSET_MM`, `REST_OFFSET_MM`,
`CONTACT_PENETRATION_THRESHOLD_MM`, `PRE_STRESS_OBJECT_MOTION_THRESHOLD`,
`ROOT_MAX_LINEAR_SPEED`,
`ROOT_MAX_ANGULAR_SPEED_DEG`, `GRIPPER_INITIAL_PARK_Z`,
`APPLY_FINGER_Z_HOP`으로 조절한다. 기본값은 TGS solver를 사용하고 CCD를
요청하며, 손 루트의 한 step 이동량을 제한해 PhysX contact solver가 반응할 시간을
준다. Isaac Sim 5.1의 GPU PhysX는 CCD 요청을 무시하므로 GPU 실행에서는 contact
offset, solver 반복, 속도 제한과 아래 관통 탈락 판정이 실제로 적용되는 보호
장치다. 현재 fixed-root position-control 그리퍼에서는
`PHYSX_SOLVE_ARTICULATION_CONTACT_LAST = False`가 관통이 적어서 기본값으로
사용한다.
`DEVICE = "cuda:0"`은 GPU PhysX, `DEVICE = "cpu"`는 CPU PhysX를 사용한다.
같은 값이 AppLauncher와 `SimulationCfg.device` 양쪽에 적용되므로,
변수 하나만 바꾸면 된다.
CPU에서 지원되지 않는 batched Fabric 연산은 자동으로 비활성화한다.
GPU PhysX에서는 `GPU_PHYSX_BUFFER_SCALE = 4`로 contact, patch,
found/lost pair, aggregate pair, collision stack, heap, temporary, soft-body,
particle buffer를 Isaac Lab 기본값의 4배로 늘린다.
`GPU_MAX_NUM_PARTITIONS = 16`도 함께 적용하며 CPU 실행에서는 이 값들을
적용하지 않는다.
접촉 길이 설정은 `CONTACT_OFFSET_MM`, `REST_OFFSET_MM`,
`CONTACT_PENETRATION_THRESHOLD_MM`, `CONTACT_SEPARATION_PRINT_DELTA_MM`처럼
mm 단위로 입력하고 내부에서만 m로 변환한다. 예를 들어
`CONTACT_PENETRATION_THRESHOLD_MM = 1.0`은 1 mm이다.
`CONTACT_OFFSET_MM`는 0과 `REST_OFFSET_MM`보다 커야 한다. 기본값은
contact 1.0 mm / rest 0.1 mm, root 속도 0.25 m/s / 90 deg/s,
articulation solver 반복 24 position / 4 velocity이다.
손-물체의 실제 contact separation이 변환된 음수 임계값보다 작아진
시도는 `penetration` 실패로 제외된다.
`ENABLE_OBJECT_CONTACT_SENSOR = True`이면 object rigid body를 source로 하고
gripper의 모든 rigid body를 filter로 둔 물체 측 단일 상세 센서로
separation을 판정한다. 여러 gripper-side 상세 센서는 생성하지 않으며,
파지 force 확인용 aggregate fingertip 센서만 별도로 유지한다.
`ENABLE_OBJECT_CONTACT_SENSOR = False`면 기존 gripper-side 상세 센서로 fallback한다.
`PENETRATION_BACKEND`는 현재
`"contact_sensor"`만 지원하며, 추후 Warp 판정으로 교체할 때 policy와 결과
형식을 변경하지 않기 위한 backend 경계다.

finger gripper는 fingertip 이름만 보지 않고 물체 측 센서의 filter에
USD의 모든 gripper rigid body를 넣는다. CLOSE 중 실제 contact force가 한 번도
확인되지 않은 시도는 joint가 stall하더라도 외력 시험으로 넘어가지 않고
`empty_close`로 제외한다.
`GRIPPER_INITIAL_PARK_Z`는 PhysX scene 최초 생성 시 아직 pregrasp pose가
적용되지 않은 gripper와 원점 물체가 겹치지 않게 한다. 이 초기 겹침을 그대로
두면 reset 뒤 첫 physics step에서 이전 contact manifold가 물체를 튕기지만,
그 접촉은 reset된 sensor frame에는 남지 않는다.
또한 APPROACH/CLOSE 중 물체가 초기 pose에서
`PRE_STRESS_OBJECT_MOTION_THRESHOLD` 이상 움직이면
`pre_stress_object_motion`으로 제외한다. 기본 0.1 m는 정상적인 닫힘 정렬
움직임을 관통으로 오인하지 않는 비상 이탈 기준이다.
finger gripper의 `APPLY_FINGER_Z_HOP` 기본값은 `False`다. `True`이면 닫히는
joint의 현재 위치를 `joint_to_z` calibration에 넣어 base Z도 함께 올리며,
`False`이면 pregrasp 높이에 base를 고정한 채 손가락만 닫는다.
finger의 최종 close joint 목표는 CLOSE 진입 시 PhysX drive에 한 번만 전달되며,
이후 같은 목표를 매 frame 반복해서 쓰지 않는다.
다접점 핸드의 상세 접촉 버퍼 크기는 `CONTACT_MAX_DATA_COUNT_PER_PRIM`에서
설정한다. 현재 값은 body당 4,096개다.
PhysX 전체 버퍼는 설치된 Isaac Lab의 표준 용량을 사용한다.

센서별 PhysX separation을 확인하려면 `PRINT_CONTACT_SEPARATION = True`로
설정한다. 출력에는 sensor 이름, 실제 rigid-body 경로, logical env, source pregrasp 번호, separation과
관통 깊이가 함께 표시된다. `CONTACT_SEPARATION_PRINT_DELTA_MM`는 반복 출력 간의
최소 변화량이다.

pre-grasp 위치에서 접근하고 닫은 뒤, 들어 올리는 단계 없이 현재 위치에서 바로
임의 방향 증가 하중 시험을 수행한다. 최종 점수는 다음 성분의 가중합이며
항상 0~1 범위다.

- 외력 생존 `force_score`
- 파지 전후 자세 `pregrasp_pose_score`
- 외력 전후 자세 `stress_pose_score`
- 파지 완료 시점의 추정 접촉 패치 면적 `contact_area_score`

각 자세 점수는 center 점수 50%와 회전 점수 50%의 평균이다. center 점수는
`clamp(1 - center 이동거리 / 10 mm, 0, 1)`, 회전 점수는 quaternion의
최단 회전각을 사용한 `clamp(1 - 회전각 차이 / 180도, 0, 1)`이다.
따라서 center와 회전 변화가 모두 없으면 자세 점수는 1이며, 변화가 클수록
0에 가까워진다. 접촉 면적 점수는 링크별 PhysX contact manifold 점들을
contact normal 방향별 평면 패치로 분리한 뒤 등가 직사각형으로 근사해 합산하고,
500 mm²에서 1점으로 포화한다. 이 값은 PhysX rigid contact의 이산 manifold
점들로 계산한 접촉 footprint이며, 변형 가능한 물체의 실제 물리 접촉면적은 아니다.
반대쪽 손가락들의 점은 서로 다른 패치로 계산하므로 물체 폭이 면적에 포함되지
않는다. 네 항목은 `policy.py` 상단의 각 `*_SCORE_WEIGHT`를 곱해 합산한 뒤
가중치 합으로 나누므로, 가중치 합을 직접 1로 맞추지 않아도 최종 점수는
0~1 범위다. `SCORE_COMPONENT_POWER=1.0`은 선형 점수이고, 1보다 크게 하면
높은 component score를 더 강조한다. 최종 점수에 로그를 취하는 것은 정렬을
바꾸지 않으므로 사용하지 않는다. 각 grasp에는
자세 및 하위 center/회전 점수, 두 시점 사이의 center 거리와 회전각,
추정 접촉 면적(m²/mm²), stress 생존율, 최대 시험 하중, 상대 위치/회전 오차와
contact force가 저장된다.
외력 시험 중 상대 이동이 `MAX_RELATIVE_TRANSLATION_M`(기본 10 mm),
상대 회전이 `MAX_RELATIVE_ROTATION_DEG`(기본 20°)를 넘거나,
contact force가 `CONTACT_LOST_DURATION_S`(기본 0.1 s) 이상 연속으로
사라지면 각각 `stress_drop` / `contact_lost` 실패로 처리한다.
실패는 `*.attempts.json`에만 남고 `output_grasp` 성공 데이터에는
저장하지 않는다. 기존 output을 병합할 때도
`merge_grasps.py` 기본값 `REQUIRE_COMPLETED = True`가 실패 record를 제외한다.
저장된 grasp의 `quality.minimum_contact_separation_m`에는 전체 시험 중 가장 깊었던
손-물체 contact separation도 함께 저장된다(음수는 겹침 깊이).
`quality.minimum_close_contact_separation_m`에는 닫힘 구간만의 최솟값이,
`quality.maximum_contact_penetration_mm`와
`quality.maximum_close_contact_penetration_mm`에는 같은 값을 읽기 쉬운 mm
관통 깊이로 저장한다.
`quality.finger_z_hop_m`과 `quality.finger_z_hop_enabled`에는 적용된 Z 이동량과
스위치 상태가 저장된다. 이 separation은 action step의 마지막 값뿐 아니라
`DECIMATION` 사이의 각 physics substep 최솟값도 포함한다.

## 3. grasp 개별 physics replay

`replay_grasps.py` 상단의 `ROOT`, `SCENE`, `DEVICE`, grasp 범위를
설정한 뒤 실행한다.

```bash
/home/uon/ochansol/isaaclab_232/.venv/bin/python replay_grasps.py
```

`output_grasp/<scene>.json`의 저장 순서로 하나씩 재생한다.
각 grasp의 `source_pregrasp_index`로 원본 pregrasp를 찾고, 접근→닫기→외력
시험을 다시 수행하며 저장된 `normal`을 같은 외력 방향으로 사용한다.
종료되면 마지막 물리 상태에서 자동 reset하지 않고 멈춘다.
Isaac Sim debug draw로 저장된 `grasp_box`를 빨간/점수 색 선으로,
`normal`을 자주색 선으로, `grasp_mat` frame을 RGB 축으로,
`target_points`를 노란색 점으로 함께 표시한다. 다음 grasp로 넘어가면
기존 debug draw를 지우고 새 데이터로 갱신한다.

- `N` / `Right` / `Space`: 다음 grasp
- `P` / `Left`: 이전 grasp
- `R`: 현재 grasp 다시 재생
- `1` / 숫자패드 `1`: 외력 점수 내림차순 정렬 후 0번부터 재생
- `2` / 숫자패드 `2`: 파지 전후 자세 점수 내림차순 정렬 후 0번부터 재생
- `3` / 숫자패드 `3`: 외력 전후 자세 점수 내림차순 정렬 후 0번부터 재생
- `4` / 숫자패드 `4`: 접촉 면적 점수 내림차순 정렬 후 0번부터 재생
- `Q` / `Esc`: 종료

기존 파일은 100 mm²에서 면적 점수가 포화되어 동률이 많으므로, 4번 정렬은
`quality.grasp_contact_area_mm2` 원값이 있으면 그 값을 우선 사용한다.

## 4. grasp 병합

```bash
python3 merge_grasps.py
```

병합 경로와 필터값은 `merge_grasps.py` 상단의 `ROOT`, `SCENE_START`,
`SCENE_END`, `SCORE_THRESHOLD` 등에서 설정한다.

`merge_grasps.py`는 각 물체 자세의 grasp를 object zero pose로 변환한다. 기본으로
`score >= 0.25`를 남긴 다음 empty-box filter와 NMS를 적용한다. 한 conf에는
`objects`가 정확히 하나 있어야 한다.
