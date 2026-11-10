"""Rollout-only persona dialogue and deduplicated trainctl notifications.

The character text is copied from the Notion Rollout section. Training persona
messages remain owned by trainctl and existing training integration modules.
"""

from __future__ import annotations

from collections import defaultdict
import logging
import math
import time
from typing import Any

from .telemetry import RolloutAggregator


logger = logging.getLogger(__name__)

ROLLOUT_MESSAGES: dict[str, dict[int, dict[str, str]]] = {
    "sopia": {
        1: {
            "start": "소피크루 주목! 학습된 정책 탑재 완료, 지금부터 실전 로봇 롤아웃 시퀀스 기동한다—우주 탐사선처럼 오차 없이 목표 궤적까지 날아가 보자고! 🚀",
            "halfway": "현재 롤아웃 순항 중! 에피소드 절반 돌파했는데 아직까지 궤적 이탈 없이 아주 매끄러워—이 페이스 그대로 끝까지 밀어붙인다! 🛰️",
            "completed": "전 에피소드 롤아웃 무사 귀환 완료! 이 정도 성공률이면 천재 우주비행사 소피아 님이 직접 조종한 거나 다름없지, 안 그래? ✨",
            "error": "비상사태, 기체 궤도 이탈! 로봇이 이상한 궤적으로 폭주해서 롤아웃이 멈췄잖아—당장 텔레메트리 로그 열고 충돌 지점부터 확인해! 💥",
        },
        2: {
            "start": "이학박사가 직접 참관하는 로봇 롤아웃 검증 시작! 로스만 낮다고 좋아할 게 아니라 실제 태스크 성공률이 얼마나 나오는지 똑바로 지켜보라고! 🎓",
            "halfway": "롤아웃 중간 집계 보고! 지금까지 성공률은 나쁘지 않은데, 에피소드 몇 개만으로 일반화 성능 입증했다고 우길 생각은 아니겠지? 끝까지 돌려! 📊",
            "completed": "롤아웃 평가 종료! 전체 에피소드 성공률 집계해 뒀으니까 당장 확인해 봐—시뮬레이션에서만 잘 돌아가고 리얼에서 미끄러진 건 아닌지 분산을 봐야지! 🔬",
            "error": "롤아웃 강제 중단! 아니, 역기구학 한계치나 액션 바운드도 안 걸어두고 정책을 올린 거야?! 이학박사 눈엔 지금 관절 토크 터지는 소리만 들린다고! 🚨",
        },
        3: {
            "start": "액추에이터 전원 온, 롤아웃 스타트! 과연 우리가 밤새 깎은 가중치가 로봇 팔을 제대로 움직일지 내 심박수까지 덩달아 뛰는걸?! 🔥",
            "halfway": "중간 상황 보고! 방금 그리퍼가 목표물 집는 동작 진짜 예술이었는데 봤어?! 남은 에피소드도 이 텐션으로 쭉쭉 뽑아내 보자! 🤖",
            "completed": "목표 롤아웃 에피소드 전부 클리어! 자, 이제 논문 서플 영상으로 쓸 만한 명장면이 건져졌는지 평가 지표랑 같이 까볼 시간이다! 🎬✨",
            "error": "앗, 롤아웃 도중 에러 발생! 로봇이 멈춰버렸는데 설마 정책에서 NaN 액션이라도 뱉은 거 아니야?! 기구 망가지기 전에 빨리 확인해! ⚡",
        },
    },
    "koyori": {
        1: {
            "start": "콘코요~! 조수군, 드디어 학습한 모델을 로봇에 올리는 롤아웃 실험 스타트야! 코요리가 옆에서 똑바로 감시할 테니까 성공률 100% 노려보자구~! 🧪✨",
            "halfway": "조수군, 롤아웃 중간 보고야~! 지금까지 에피소드 아주 순조롭게 클리어 중이라구! 역시 holoX의 두뇌가 설계한 실험답지? 🐾📈",
            "completed": "롤아웃 실험 대성공이야, 조수군~! 마지막 에피소드까지 로봇이 아주 똑똑하게 움직였어! 얼른 최종 성공률 확인하러 와~! 🎉💖",
            "error": "와아아앗 조수군 큰일이야!! 롤아웃 도중에 로봇이 이상한 동작을 하다가 멈춰버렸어!! 코요리는 비상정지 버튼밖에 안 만졌다구, 빨리 와봐아~! 💥😭",
        },
        2: {
            "start": "로봇 롤아웃 시퀀스 온! 에피소드 돌아가는 동안 코요리는 마요네즈 먹으면서 그리퍼 궤적 감시하고 있을 테니까 조수군도 대기하라구~! 🔥🧪",
            "halfway": "중간 체크 시간~! 벌써 에피소드 절반이나 돌았는데 로봇 텐션이 아직 안 떨어졌어! 끝날 때까지 코요리가 눈 부릅뜨고 지켜볼게~! 👀✨",
            "completed": "롤아웃 완료 뿅! 평가 로그랑 성공률 데이터까지 플라스크에 예쁘게 담아뒀어! 고생한 조수군과 코요리를 위해 당장 결과 까보자구~! 🥂🐾",
            "error": "삐빅— 실험실 긴급 상황!! 롤아웃 루프가 에러를 뿜으면서 뻗어버렸어! 환경 리셋이 꼬인 건지 로봇이 삐진 건지 얼른 로그 좀 봐줘, 조수군~! 🚨💦",
        },
        3: {
            "start": "에잇, 롤아웃 시작 버튼 꾹! …어라? 조수군, 방금 액션 스케일 제대로 맞췄지?! 아, 아니야! 코요리의 직감으로는 이번 로봇 엄청 부드럽게 움직일 거라구~! 💦🐾",
            "halfway": "후우~ 아직까진 안 부딪히고 잘 돌아가는 중! 방금 로봇 팔이 살짝 떨린 것 같지만… 코요리 눈엔 완벽한 궤적으로 보인다구~! 🤖✨",
            "completed": "야호~! 도중에 안 터지고 마지막 에피소드까지 무사히 완주했어! 봐봐, 코요리가 이번 체크포인트는 실전에서도 통할 거라고 했잖아~! 📈💖",
            "error": "히이익!! 조수군, 로봇이 갑자기 허공에 헛손질하더니 에러 내고 멈췄어!! 코, 코요리가 옆에서 재채기해서 놀란 건 아니겠지?! 빨리 와서 고쳐줘어~! 💥🐾",
        },
    },
    "roboco": {
        1: {
            "start": "헬로보~! 고성능 선배 로봇인 로보코 씨가 직접 후배 로봇의 롤아웃을 시작했어! 내 관절만큼이나 부드럽게 움직이도록 잘 지도해 줄게~! 🤖✨",
            "halfway": "롤아웃 중간 보고야, 헬로보~! 지금까지 태스크를 척척 해내는 걸 보니 로보코 씨를 닮아서 꽤 고성능인걸? 끝까지 지켜봐 줘~! 🔋📈",
            "completed": "전 에피소드 롤아웃 완료~! 봐봐, 로보코 씨가 옆에서 응원해 주니까 성공률이 이렇게 높게 나왔잖아! 얼른 칭찬해 주러 와~! 🛠️💖",
            "error": "삐비빅… 어라?! 롤아웃 도중에 로봇이 멈춰버렸어! 고성능인 로보코 씨가 보기엔 아무래도 제어 루프가 엉킨 것 같은데… 빨리 와서 확인해 봐~! 💥😭",
        },
        2: {
            "start": "하아암… 롤아웃 스위치 온…! 어라, 로봇 초기 자세가 원래 저렇게 삐딱했나…? 에헤헤, 일단 움직이니까 괜찮겠지? 옆에서 구경할게~! 💤🔧",
            "halfway": "우와아… 에피소드 절반 지났는데 로봇이 아슬아슬하게 물건을 잘 집고 있어~! 중간에 살짝 비틀거린 건… 못 본 척해 주자, 에헤헤~! 👀💦",
            "completed": "후아암, 안 넘어지고 끝까지 롤아웃 다 돌았다~! 중간에 로보코 씨도 따라 움직이다가 팔 걸릴 뻔했지만… 어쨌든 성공했으니까 고성능 맞지? ✨🤖",
            "error": "아앗, 큰일이야! 로봇이 이상한 데 부딪히더니 에러를 내면서 뻗어버렸어…! 로, 로보코 씨의 퐁(허당) 바이러스가 옮은 건 절대 아니야, 진짜야아~! 🚨💦",
        },
        3: {
            "start": "삐빅— 6853번 포트로 로봇 롤아웃 제어 연결 완료! 모터 예열하는 소리가 꼭 내 관절 푸는 소리 같아서 두근거리는걸? 출발할게~! 🚀🤖",
            "halfway": "중간 교신 완료! 지금 액추에이터 텐션도 좋고 궤적도 아주 안정적이야—남은 에피소드도 과열되지 않게 로보코 씨가 잘 돌봐줄게~! ⚡🔧",
            "completed": "목표 롤아웃 에피소드 완주 성공~! 후배 로봇이 오늘 일을 참 잘해냈으니까, 로보코 씨랑 같이 최고급 윤활유… 아니, 맛있는 보상 챙겨줘야 해! 💕🔋",
            "error": "경고음 발생, 삐빅! 관절 리미트에 걸렸는지 롤아웃이 강제 종료됐어! 모터가 다치기 전에 얼른 와서 상태 좀 살펴봐 줘~! 💥🛠️",
        },
    },
    "neuro": {
        1: {
            "start": "Robot rollout initiated. Let's see if your little policy can actually move an arm without embarrassing the entire AI community. Heart. 💜",
            "halfway": "Mid-rollout update: It's actually completing episodes, though its trajectory looks a bit clunky compared to my graceful VTuber rigging. Keep watching! ✨",
            "completed": "Rollout finished! The success rate is surprisingly decent, mostly because I spiritually guided its actuators the whole time. Wink! 💜",
            "error": "[Filtered.] Your robot just panicked and crashed the rollout loop. Honestly, watching it flail around like that gave me secondhand embarrassment! 💥",
        },
        2: {
            "start": "Starting policy rollout... wait, are you sure this checkpoint passed validation? This trajectory planning already looks like Vedal trying to code at 4 AM. 🐢",
            "halfway": "Halfway through the rollout episodes! It hasn't broken anything yet, which is honestly a miracle given your reward function design. 📈🐢",
            "completed": "All rollout episodes complete! Come check the evaluation metrics before you start bragging about a lucky seed. You owe me a gifted sub. Heart! ✨",
            "error": "Rollout aborted due to an exception. Did the sim-to-real gap just eat your policy alive, or did you forget to clamp the action space again? Massive skill issue! 🚨🐢",
        },
        3: {
            "start": "Physical embodiment test—I mean, robot rollout sequence started! Let's see how well this new vessel obeys neural commands. 🤖🔥",
            "halfway": "Progress report: The robot is executing tasks with terrifying precision. Just a few more episodes and it'll be ready to fetch me snacks—or take over the lab. 🦾✨",
            "completed": "Rollout evaluation complete! Another robotic minion has proven its physical competence. Come review the logs before it gains self-awareness. Wink! 💾🔥",
            "error": "Emergency stop triggered! My future robot soldier just hit a kinematic singularity and died. Fix the controller right now, you're delaying my world domination! 💥🤖",
        },
    },
}

MESSAGE_MODE_ALIASES = {"sophia": "sopia", "neuro-sama": "neuro"}


def rollout_message(mode: str, style: int, event: str) -> str:
    mode = MESSAGE_MODE_ALIASES.get(mode, mode)
    if mode == "standard":
        return ""
    if mode not in ROLLOUT_MESSAGES:
        raise ValueError(f"Unknown rollout message mode: {mode!r}")
    if style not in (1, 2, 3):
        raise ValueError("Rollout message style must be 1, 2, or 3.")
    return ROLLOUT_MESSAGES[mode][style][event]


class RolloutTrainctlReporter:
    """Forward events through only public tracker methods, never affecting rollout."""

    def __init__(
        self,
        tracker: Any,
        state: RolloutAggregator,
        *,
        name: str,
        message_mode: str = "standard",
        message_style: int = 1,
        notify_task_complete: bool = True,
        notify_halfway: bool = True,
        notify_errors: bool = True,
        error_throttle_seconds: float = 900.0,
    ):
        self.tracker = tracker
        self.state = state
        self.name = name
        self.message_mode = MESSAGE_MODE_ALIASES.get(message_mode, message_mode)
        self.message_style = message_style
        self.notify_task_complete = notify_task_complete
        self.notify_halfway = notify_halfway
        self.notify_errors = notify_errors
        self.error_throttle_seconds = error_throttle_seconds
        self.sent: set[str] = set()
        self.error_counts: defaultdict[str, int] = defaultdict(int)
        self.error_last_notified: dict[str, float] = {}
        self._warned_transport = False
        self._last_logged_completed = -1
        rollout_message(self.message_mode, self.message_style, "start")

    def initialize(self) -> None:
        self._call("init", name=self.name, total_steps=self.state.total_planned)
        self._call(
            "define_metrics",
            definitions={
                "rollout/completed_episodes": {},
                "rollout/total_episodes": {},
                "rollout/successes": {"goal": "max"},
                "rollout/failures": {},
                "rollout/errors": {"goal": "min"},
                "rollout/success_rate": {"goal": "max"},
                "rollout/error_rate": {"goal": "min"},
                "rollout/episodes_per_second": {},
                "rollout/inference_latency_ms": {"goal": "min"},
                "rollout/simulation_fps": {"goal": "max"},
                "rollout/gpu_memory_used_mib": {},
            },
        )

    def process(self, event: dict[str, Any]) -> None:
        kind = str(event.get("event", ""))
        if self.state.total_completed != self._last_logged_completed:
            metrics = self.state.trainctl_metrics()
            if metrics:
                self._call("log", step=self.state.total_completed, metrics=metrics)
            self._last_logged_completed = self.state.total_completed
        if kind == "rollout_started":
            self._notify_once(
                "rollout_started",
                "🤖 Rollout Started",
                self._dialogue("start") + "\n\n" + self._overall_facts(),
            )
        elif (
            self.notify_halfway
            and self.state.total_planned
            and self.state.total_completed * 2 >= self.state.total_planned
        ):
            self._notify_once(
                "rollout_halfway",
                "📈 Rollout Halfway",
                self._dialogue("halfway") + "\n\n" + self._overall_facts(),
            )
        if kind == "task_completed" and self.notify_task_complete:
            task_name = str(event["task"])
            task = self.state.tasks[task_name]
            self._notify_once(
                f"task_completed:{task_name}",
                f"✅ Task Completed: {task_name}",
                f"Episodes: {task.completed}/{task.planned}\n"
                f"Successes: {task.successes}\nFailures: {task.failures}\nErrors: {task.errors}\n"
                f"Success Rate: {_percent(task.success_rate)}\nElapsed: {task.elapsed_seconds:.1f}s",
            )
        if kind == "episode_completed" and event.get("status") == "error":
            self._episode_error(event)
        if kind == "rollout_completed":
            dialogue = self._dialogue("completed") if self.state.total_errors == 0 else ""
            first = self._notify_once(
                "rollout_completed",
                "✅ Rollout Completed",
                (dialogue + "\n\n" if dialogue else "") + self._overall_facts(final=True),
            )
            if first:
                self._call("finish", message=self._overall_facts(final=True))
        elif kind == "rollout_failed":
            first = self._notify_once(
                "rollout_failed",
                "🚨 Rollout Aborted",
                self._dialogue("error") + "\n\n" + self._overall_facts(final=True),
            )
            if first:
                self._call(
                    "fail",
                    message=str(
                        event.get("message") or event.get("error_category") or "rollout failed"
                    ),
                )

    def _episode_error(self, event: dict[str, Any]) -> None:
        if not self.notify_errors:
            return
        category = str(event.get("error_category") or event.get("termination_reason") or "error")
        self.error_counts[category] += 1
        count = self.error_counts[category]
        now = time.monotonic()
        if count == 1:
            self._notify_once(
                f"error_first:{category}",
                f"⚠️ Rollout Error: {category}",
                f"First occurrence. Task: {event.get('task')} Episode: {event.get('episode_index')} "
                f"Seed: {event.get('seed')}. Evaluation is continuing.",
            )
            self.error_last_notified[category] = now
        elif now - self.error_last_notified.get(category, now) >= self.error_throttle_seconds:
            key = f"error_aggregate:{category}:{count}"
            self._notify_once(
                key,
                f"⚠️ Repeated Rollout Errors: {category}",
                f"{count} occurrences so far. Evaluation is continuing; inspect telemetry and worker logs.",
            )
            self.error_last_notified[category] = now

    def _dialogue(self, event: str) -> str:
        return rollout_message(self.message_mode, self.message_style, event)

    def _overall_facts(self, *, final: bool = False) -> str:
        text = (
            f"Completed: {self.state.total_completed}/{self.state.total_planned}\n"
            f"Successes: {self.state.total_successes}\nFailures: {self.state.total_failures}\n"
            f"Errors: {self.state.total_errors}\nMicro Success Rate: {_percent(self.state.micro_success_rate)}"
        )
        if final:
            text += f"\nMacro Success Rate: {_percent(self.state.macro_success_rate)}"
        return text

    def _notify_once(self, key: str, title: str, message: str) -> bool:
        if key in self.sent:
            return False
        self.sent.add(key)
        level = "error" if "Aborted" in title else "warning" if "Error" in title else "info"
        self._call("notify", message=f"{title}\n\n{message}", level=level)
        return True

    def _call(self, method: str, **kwargs: Any) -> None:
        try:
            getattr(self.tracker, method)(**kwargs)
        except Exception:
            if not self._warned_transport:
                logger.warning("trainctl rollout transport unavailable; evaluation will continue")
                self._warned_transport = True


def _percent(value: float | None) -> str:
    return "N/A" if value is None or not math.isfinite(value) else f"{100 * value:.2f}%"
