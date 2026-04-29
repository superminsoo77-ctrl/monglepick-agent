"""
관리자 AI — Draft Args validator 단위 테스트 (잔존 결함 P1 패치 회귀 방지선).

검증 대상: `_DraftWithModeMixin._validate_update_requires_target_id`
  - mode="update" + target_id 누락(None/0/"") 조합은 ValidationError 로 차단돼야 한다.
  - mode="update" + target_id 명시(int) 는 통과해야 한다.
  - mode="create" 는 target_id 유무와 무관하게 통과해야 한다.
  - TicketReplyDraftArgs 는 mode/target_id 가 없으므로 mixin 영향을 받지 않아야 한다.

배경: 2026-04-29 길 A v3 보강에서 mode/target_id 필드는 추가됐지만, target_id=None
      인 채 mode="update" 가 들어오면 _resolve_target_path 가 조용히 create 모달로
      폴백하는 결함이 잔존했다. "공지 수정해줘" 발화에 새 글이 만들어지는 사고를 막기
      위해 model_validator 로 Pydantic 단계에서 차단하도록 보강.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from monglepick.tools.admin_tools.drafts import (
    BannerDraftArgs,
    ChatSuggestionDraftArgs,
    FaqDraftArgs,
    HelpArticleDraftArgs,
    NoticeDraftArgs,
    PointPackDraftArgs,
    QuizDraftArgs,
    RewardPolicyDraftArgs,
    TermDraftArgs,
    TicketReplyDraftArgs,
    WorldcupCandidateDraftArgs,
)


# ============================================================
# 테스트 픽스처: mode/target_id 외 필수 필드 최소 세트
# ============================================================
# 각 Args 클래스가 요구하는 다른 필수 필드를 mode 와 target_id 검증과 분리하기 위해
# 픽스처로 미리 채워둔다. (값 자체는 어떤 형태든 무방)

_MINIMAL_REQUIRED_FIELDS: dict[type, dict] = {
    NoticeDraftArgs: {"title": "test", "content": "test"},
    FaqDraftArgs: {"category": "PAYMENT", "question": "Q", "answer": "A"},
    HelpArticleDraftArgs: {"title": "T", "category": "PAYMENT", "content": "C"},
    BannerDraftArgs: {"title": "T"},
    QuizDraftArgs: {
        "movieId": "m1",
        "question": "Q",
        "choices": ["a", "b"],
        "answerIndex": 0,
    },
    ChatSuggestionDraftArgs: {"surface": "admin_assistant", "text": "hi"},
    TermDraftArgs: {"type": "SERVICE", "version": "v1", "content": "c"},
    WorldcupCandidateDraftArgs: {"movieId": "m1"},
    RewardPolicyDraftArgs: {"code": "X", "pointAmount": 10, "condition": "c"},
    PointPackDraftArgs: {"packCode": "P", "points": 10, "priceKrw": 990},
}

# mixin 을 inherit 한 Args 10개 (TicketReplyDraftArgs 제외)
_MIXIN_ARGS_CLASSES = list(_MINIMAL_REQUIRED_FIELDS.keys())


# ============================================================
# 차단 테스트 — mode="update" + target_id 누락
# ============================================================

@pytest.mark.parametrize("cls", _MIXIN_ARGS_CLASSES)
def test_update_without_target_id_is_rejected(cls):
    """mode='update' 인데 target_id 가 없으면 ValidationError."""
    minimal = _MINIMAL_REQUIRED_FIELDS[cls]
    with pytest.raises(ValidationError) as exc_info:
        cls(mode="update", **minimal)
    # 에러 메시지에 안내 문구가 포함돼야 한다 (LLM 또는 운영자가 원인 파악 가능).
    assert "target_id" in str(exc_info.value)


@pytest.mark.parametrize("cls", _MIXIN_ARGS_CLASSES)
@pytest.mark.parametrize("falsy_target", [None, 0, ""])
def test_update_with_falsy_target_id_is_rejected(cls, falsy_target):
    """target_id 가 None / 0 / "" (falsy) 면 update 차단."""
    minimal = _MINIMAL_REQUIRED_FIELDS[cls]
    with pytest.raises(ValidationError):
        # 일부 클래스는 target_id 가 Optional[int] 라 ""/0 자체가 타입 오류일 수
        # 있는데, 그 또한 차단의 일종이므로 ValidationError 만 검증한다.
        cls(mode="update", target_id=falsy_target, **minimal)


# ============================================================
# 통과 테스트 — 정상 케이스 회귀 방지
# ============================================================

@pytest.mark.parametrize("cls", _MIXIN_ARGS_CLASSES)
def test_update_with_valid_target_id_passes(cls):
    """mode='update' + target_id=42 는 정상 통과."""
    minimal = _MINIMAL_REQUIRED_FIELDS[cls]
    instance = cls(mode="update", target_id=42, **minimal)
    assert instance.mode == "update"
    assert instance.target_id == 42


@pytest.mark.parametrize("cls", _MIXIN_ARGS_CLASSES)
def test_create_mode_does_not_require_target_id(cls):
    """mode='create' 는 target_id 없이도 통과 (기본 흐름)."""
    minimal = _MINIMAL_REQUIRED_FIELDS[cls]
    instance = cls(mode="create", **minimal)
    assert instance.mode == "create"
    assert instance.target_id is None


@pytest.mark.parametrize("cls", _MIXIN_ARGS_CLASSES)
def test_default_mode_is_create(cls):
    """mode 미지정 시 default='create' — target_id 검증 우회."""
    minimal = _MINIMAL_REQUIRED_FIELDS[cls]
    instance = cls(**minimal)
    assert instance.mode == "create"


# ============================================================
# Mixin 미적용 클래스 — TicketReplyDraftArgs
# ============================================================

def test_ticket_reply_draft_does_not_inherit_mixin():
    """TicketReplyDraftArgs 는 mode/target_id 가 없으므로 mixin 검증 영향을 받지 않음."""
    instance = TicketReplyDraftArgs(ticket_id=1, content="안녕하세요")
    assert instance.ticket_id == 1
    # mode 속성 자체가 없어야 한다 (혹시라도 추가됐으면 mixin 적용 여부 재검토 필요).
    assert not hasattr(instance, "mode")
    assert not hasattr(instance, "target_id")
