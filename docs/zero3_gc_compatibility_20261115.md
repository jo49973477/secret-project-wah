WAH Multilevel / DeepSpeed 0.19.9 호환성 검증 (2026-11-15, spl)

실행 환경: `.venv/bin/python`, PyTorch `2.9.0+cu128`, DeepSpeed `0.19.9`,
RTX 3090 GPU 0·1, BF16, ZeRO stage 3. 데이터는
`/ssdg/spl_yeongyoo/univtac_new` 전체 UniVTAC이다.
`meta/info.json` 기준 799 episodes, 151,825 frames이며 episode 제한은 없다.

1. BF16 초기화 오류 원인과 수정

   정확한 파라미터는 `backbone.model.model.visual.patch_embed.proj.weight`다.
   두 rank의 수정 전 로그:

   ```text
   shape=(1024, 3, 2, 16, 16)
   stride=(1536, 1, 768, 48, 3)
   dtype=torch.bfloat16 is_contiguous=False requires_grad=False
   storage_offset=0 shared_storage=[자기 자신] is_view=False
   RuntimeError: self.stride(-1) must be 1 ... but got 3
   ```

   `Qwen3Backbone._apply_vision_patch_embed_channels_last()`의
   `module.to(memory_format=torch.channels_last_3d)`가 만든 레이아웃이다.
   이 함수는 일반 생성 경로와 pretrained 로딩 후 finalize 경로에서 호출된다.
   DeepSpeed 0.19.9의 초기 broadcast는 BF16을 uint8 view로 바꾸므로 실패했다.
   가중치 레이아웃 변환을 제거하고 기존 입력 activation 변환 훅만 유지했다.
   parameter object, storage, 값, requires_grad 및 tied embedding을 교체하지 않는다.
   전역 `.data` 교체, 매 forward weight clone, DeepSpeed 수정/monkey patch는 없다.

   증거: `outputs/debug_zero3_gc_fix_audit_20261115_2144/logs/train.log`.
   1,613개 parameter name을 검사했고 비연속 파라미터는 위 Conv3d 하나였다.
   별도로 embedding/lm_head의 의도된 tied storage 두 이름을 확인했다.

2. 원래 backward 오류 원인과 수정

   레이아웃만 수정했을 때 0.19.9에서도 동일한 `0 vs 256` 오류가 재현됐다.
   anomaly trace는 `LocalJointActionStage.output_projection`을 지목했다.
   기존 `checkpoint(stage_module, use_reentrant=False)`는 `_z3_leaf=True`인
   stage의 `__call__`까지 재실행했다. 재계산 종료 시 leaf post-forward 훅이
   gathered 파라미터를 다시 partition하지만 원래 backward는 그 가중치를
   여전히 필요로 하는 구조였다. `set_checkpoint_early_stop(False)`만으로는
   이 파라미터 수명 문제가 해결되지 않았다.

   checkpoint를 `LocalJointActionStage.forward` 안으로 옮겨 `_forward` 계산만
   재계산한다. ZeRO gather/release 및 backward 훅은 checkpoint 바깥을 감싼다.
   `use_reentrant=False`, leaf 설정, trainable 모듈, 모델/state_dict 구조는 유지한다.
   입력/point feature gradient를 detach하지 않는다.
   따라서 이 환경에서는 non-reentrant 방식을 사용할 수 있으며 전역 reentrant
   전환은 필요하지 않다. 모든 DeepSpeed/모델 조합에 대한 일반적 보장은 아니다.

   증거: `outputs/debug_zero3_gc_fix_layout_20261115_2145/logs/train.log`.

3. 변경 파일

   | 파일 | 변경 |
   |---|---|
   | `gr00t/model/modules/qwen3_backbone.py` | Conv3d weight channels-last 변환 제거, 입력 훅 유지 |
   | `gr00t/model/extension/sidecar_dit.py` | stage 호출을 checkpoint로 감싸지 않고 내부 checkpoint 활성화 전달 |
   | `gr00t/model/extension/multilevel_local_sidecar.py` | leaf 내부 non-reentrant 재계산 경계 |
   | `gr00t/experiment/parameter_layout.py` | 모든 파라미터 stride/alias/offset/storage 감사, weight 값 미출력 |
   | `gr00t/experiment/experiment.py` | prepare 이전 감사 실행 |
   | `gr00t/experiment/zero3_debug.py` | opt-in autograd 참여, 전체 gradient shard 유한성, FP32 update 표본, rank별 loss 기록 |
   | `gr00t/experiment/trainer.py` | debug 설치 및 첫-step 메모리 래퍼와의 복원 순서 보장 |
   | `tests/gr00t/model/test_parameter_layout.py` | 값·identity·tie·requires_grad·byte view·재계산·gradient parity 회귀 |
   | `tests/gr00t/model/test_pretrained_loading.py` | 실제 작은 HF checkpoint 로딩 값 보존 및 layout 검사 |
   | `tests/gr00t/model/test_zero3_module_boundaries.py`, `_run_zero3_local_checkpoint.py` | 두 GPU BF16 전체 gradient parity 및 accumulation=2 검증 |
   | `tests/gr00t/experiment/test_training_memory.py` | 첫-step 래퍼 이후에도 gradient 감사 지속 검증 |
   | `pyproject.toml`, `uv.lock` | DeepSpeed 0.19.9 고정 |

4. gradient 검증의 범위

   일반 tensor `register_hook`은 이 ZeRO 실행의 일부 선형층에서 실제 gradient가
   있어도 호출되지 않았다. 작은 두-GPU 모델에서 gradient 수치가 기준과 같고
   FP32 업데이트가 발생하는 것으로 이를 확인했다. 참여 증거는
   `register_post_accumulate_grad_hook`으로 수집하며, ZeRO가 unused 파라미터에도
   생성하는 zero buffer만으로 참여했다고 판단하지 않는다.

   로컬 `wah_multilevel_local_full`은 point backbone을 frozen으로 사용하고,
   `tactile_fusion_mode=none`, `contact_input_mode=none`으로 tactile 계산을 생략한다.
   기존부터 trainable로 표시된 tactile projection 4개는 비활성 사유와 함께
   JSONL에 별도 기록한다. 이 작업에서 requires_grad를 변경하거나 해당 경로를
   새로 비활성화하지 않았다. 나머지 516개 trainable tensor가 활성 검증 대상이다.
   원래 frozen인 point/tactile backbone의 실제 학습 backward를 검증했다고 주장하지 않는다.

   FP32 업데이트는 파라미터당 최대 1,024개의 분산된 local shard 표본을 비교한다.
   전체 파라미터의 모든 원소를 비교하는 검사는 아니다. gradient 유한성은
   전체 local gradient shard를 검사하며 두 rank 모두 검사한다.

5. 의존성

   `uv pip check --python .venv/bin/python`: `All installed packages are compatible`.
   `uv lock --locked --offline`: 통과. 변경 전후 lock의 패키지 버전을 비교했을 때
   DeepSpeed `0.17.6 → 0.19.9`만 변경됐다. Torch/CUDA 패키지는 유지했다.
   editable 프로젝트 metadata만 `--no-deps --no-build-isolation -e .`로 갱신했다.
   DeepSpeed engine/utils/linear/stage3 파일은 설치 RECORD 해시와 일치한다.
   `uv sync`나 환경 초기화, commit/push는 실행하지 않았다.

6. 실제 학습 결과

   | 실행 | 결과 |
   |---|---|
   | `debug_zero3_gc_fix_step1_20261115_2156` | 1-step, accumulation=1, exit 0, loss=1.3297, 양쪽 rank 516개 활성 gradient 확인 |
   | `debug_zero3_gc_fix_step10_20261115_2158` | 첫 10-step 완료, exit 0. 기존 메모리 래퍼 때문에 gradient 감사가 첫 step에만 남아 최종 검증을 별도로 재실행 |
   | `debug_zero3_gc_fix_verified10_20261115_2202` | 최종 10-step 및 native checkpoint 저장 완료, exit 0, 모든 step/rank 감사 통과 |
   | `debug_zero3_gc_fix_accum2_20261115_2208` | accumulation=2, 각 rank 4 microbatch·2 optimizer-step, checkpoint-2 저장 및 exit 0 |

   최종 로그 `outputs/debug_zero3_gc_fix_verified10_20261115_2202/logs/train.log`:

   ```text
   zero3-gradient-audit stage=3 bf16=True checkpoint_modules=[..., 'action_head.model', ...]
   rank=0 step=2 seen=516/520 nonzero=460 missing=[] invalid=[] sampled_updates=460 finite=True
   rank=0 step=3 seen=516/520 nonzero=516 missing=[] invalid=[] sampled_updates=516 finite=True
   rank=0 step=10 seen=516/520 nonzero=516 missing=[] invalid=[] sampled_updates=516 finite=True
   rank=1 step=10 seen=516/520 nonzero=516 missing=[] invalid=[] sampled_updates=512 finite=True
   Step 10/10 ... loss=1.0373 ... grad=0.754851
   Training completed!
   ```

   각 rank의 `zero3_gradient_audit_rank*.jsonl`은 정확히 10개 optimizer 기록,
   `zero3_loss_audit_rank*.jsonl`은 10개 microbatch 기록을 포함한다.
   모든 step에서 활성 gradient 516개가 참여했고 유한했다. 두 rank를 합쳐
   516개 전부의 FP32 업데이트 표본을 확인했다. rank 1은 scalar gate 4개의
   shard가 padding이라 업데이트 표본 수가 512개다. 초기 zero gate 때문에
   첫 두 backward의 일부 branch gradient는 0이며, 3-step부터 516개 모두
   비영 gradient였다. 첫 optimizer LR은 warmup으로 0이므로 업데이트가 없다.

   rank 0 loss 범위 `1.0347278–1.4447522`, rank 1 `1.0398659–1.3256575`.
   유한성, 기록 수, native checkpoint global_step=10, 모든 stage의 업데이트,
   오류/trace cache invalidation 부재를 `verification.json`으로 별도 검증했다.
   새 partition state 오류나 shape 오류는 없었다. 기존 deprecation/FutureWarning은
   남아 있으며 이를 새 ZeRO partition 경고로 분류하지 않았다.

   실제 UniVTAC accumulation=2의 step loss는 `1.2916`, `1.2726`이었다.
   각 rank의 microbatch 반환 시 optimizer-step 카운터는 `[0, 1, 1, 2]`로,
   첫 microbatch에서는 업데이트하지 않고 두 번째에서 업데이트함을 확인했다.
   이 저장소의 global_batch_size=2는 accumulation 이전 batch이므로 이 추가
   실행의 유효 batch는 4다. 활성 gradient 누락/비유한 값은 없었고 두 번째
   optimizer step에서 460개 tensor의 FP32 업데이트 표본을 확인했다.
   해당 실행의 root INFO 종료 문구는 로그에 없었지만 실제 process exit 0,
   양쪽 rank의 gradient/loss 기록 및 완성된 checkpoint-2로 완료를 검증했다.
   이 실행도 별도 `verification.json`을 보존했다.

   관련 CPU 회귀: `83 passed, 2 deselected` (GPU 테스트는 별도 실행).
   최종 두-GPU 회귀: `1 passed` 및 두 rank 모두
   `ZERO3_LOCAL_CHECKPOINT_PARITY ... optimizer_steps=2 accumulation=2 PASS`.
   checkpoint OFF 기준과 BF16 출력/전체 parameter gradient를 비교했다.
   수정 Python 12개 파일의 Ruff lint/format 및 `git diff --check` 통과.
   CPU/GPU 회귀 로그와 재검증 스크립트는 최종 실행 디렉터리의
   `cpu_regression.log`, `gpu_regression.log`, `verify_run.py`에 보존했다.
   기존 Sparsh background 미지정 경고도 남아 있지만 현재 구성은 tactile
   경로를 사용하지 않는다. 주요 ZeRO-3 + GC + BF16 오류의 미해결 항목은 없다.

7. 전체 학습 새 실행 명령

   아래 명령은 새 학습이며 기존 smoke checkpoint resume 명령은 아니다.
   W&B 외부 동기화는 offline으로 유지한다. 디버그 감사는 비용이 있으므로
   전체 학습에서는 `GR00T_ZERO3_DEBUG`를 설정하지 않는다.

   ```bash
   CUDA_VISIBLE_DEVICES=0,1 \
   WANDB_MODE=offline \
   GR00T_ZERO3_DEBUG=0 \
   FLORENCE_DATASET=/ssdg/spl_yeongyoo/univtac_new \
   uv run --no-sync python scripts/train_hydra.py \
     --config-name experiment/wah_multilevel_local_full \
     hardware=dual_3090_zero3 \
     hardware.gradient_checkpointing=true \
     hardware.global_batch_size=2 \
     training.gradient_accumulation_steps=1 \
     training.max_steps=10000 \
     trainctl.enabled=false \
     output_dir="outputs/wah_multilevel_zero3_gc_$(date +%Y%m%d_%H%M%S)"
   ```
