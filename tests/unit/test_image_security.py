"""
이미지 업로드 보안 강화 테스트.

보안 헬퍼 함수 4개를 단위 테스트한다:
- _strip_base64_prefix: Data URL 접두사 제거 + 패딩 보정
- _validate_image_bytes: JPEG/PNG 매직바이트 검증
- _check_upload_rate_limit: IP당 분당 업로드 횟수 제한 (Redis 기반)
- DecompressionBomb 방어: IMAGE_MAX_PIXELS 설정 확인

테스트 클래스:
- TestStripBase64Prefix (5개)
- TestValidateImageBytes (6개)
- TestCheckUploadRateLimit (4개) — Redis mock 사용
- TestDecompressionBombDefense (1개)
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from monglepick.api.chat import (
    _check_upload_rate_limit,
    _strip_base64_prefix,
    _validate_image_bytes,
)


# ============================================================
# Data URL 접두사 제거 + 패딩 보정 테스트
# ============================================================


class TestStripBase64Prefix:
    """_strip_base64_prefix 헬퍼 테스트."""

    def test_jpeg_data_url_prefix_removed(self):
        """JPEG Data URL 접두사가 제거된다."""
        data = "data:image/jpeg;base64,/9j/4AAQ"
        result = _strip_base64_prefix(data)
        assert result == "/9j/4AAQ"

    def test_png_data_url_prefix_removed(self):
        """PNG Data URL 접두사가 제거된다."""
        data = "data:image/png;base64,iVBORw0K"
        result = _strip_base64_prefix(data)
        assert result == "iVBORw0K"

    def test_padding_correction(self):
        """base64 패딩이 4의 배수가 되도록 보정된다."""
        # 길이 5 → 나머지 1 → "===" 추가하여 길이 8로 보정
        data = "AAAAA"
        result = _strip_base64_prefix(data)
        assert len(result) % 4 == 0
        assert result.endswith("===")

    def test_already_valid_base64_passthrough(self):
        """유효한 base64 문자열은 변경 없이 통과한다."""
        data = "/9j/4AAQ"  # 길이 8, 4의 배수
        result = _strip_base64_prefix(data)
        assert result == "/9j/4AAQ"

    def test_empty_string(self):
        """빈 문자열은 그대로 반환된다."""
        result = _strip_base64_prefix("")
        assert result == ""


# ============================================================
# 매직바이트 검증 테스트
# ============================================================


class TestValidateImageBytes:
    """_validate_image_bytes 헬퍼 테스트."""

    def test_jpeg_magic_bytes_pass(self):
        """JPEG 매직바이트(FF D8 FF)가 통과한다."""
        jpeg_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        # 예외 없이 통과해야 함
        _validate_image_bytes(jpeg_bytes)

    def test_png_magic_bytes_pass(self):
        """PNG 매직바이트(89 PNG)가 통과한다."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        _validate_image_bytes(png_bytes)

    def test_gif_magic_bytes_rejected(self):
        """GIF 매직바이트(GIF89a)가 거부된다 (415)."""
        gif_bytes = b"GIF89a" + b"\x00" * 100
        with pytest.raises(ValueError, match="415"):
            _validate_image_bytes(gif_bytes)

    def test_executable_rejected(self):
        """실행 파일 매직바이트(MZ)가 거부된다 (415)."""
        exe_bytes = b"MZ" + b"\x00" * 100
        with pytest.raises(ValueError, match="415"):
            _validate_image_bytes(exe_bytes)

    def test_empty_bytes_rejected(self):
        """빈 바이트가 거부된다 (400)."""
        with pytest.raises(ValueError, match="400"):
            _validate_image_bytes(b"")

    def test_svg_text_rejected(self):
        """SVG 텍스트 데이터가 거부된다 (415)."""
        svg_bytes = b"<svg xmlns='http://www.w3.org/2000/svg'></svg>"
        with pytest.raises(ValueError, match="415"):
            _validate_image_bytes(svg_bytes)


# ============================================================
# IP당 업로드 Rate Limiting 테스트 (Redis 기반)
# ============================================================


def _make_mock_redis(zcard_return: int = 0):
    """Redis mock을 생성한다. zcard_return은 현재 윈도우 내 요청 수."""
    mock_pipe = AsyncMock()
    # pipeline().execute() 반환값: [zremrangebyscore결과, zcard결과, zadd결과, expire결과]
    mock_pipe.execute = AsyncMock(return_value=[0, zcard_return, 1, True])
    mock_pipe.zremrangebyscore = AsyncMock()
    mock_pipe.zcard = AsyncMock()
    mock_pipe.zadd = AsyncMock()
    mock_pipe.expire = AsyncMock()

    mock_redis = AsyncMock()
    mock_redis.pipeline = lambda: mock_pipe
    mock_redis.zrem = AsyncMock()
    return mock_redis


class TestCheckUploadRateLimit:
    """_check_upload_rate_limit Redis 기반 Rate Limiting 테스트."""

    @pytest.mark.asyncio
    async def test_under_limit_passes(self):
        """제한 이하의 요청은 통과한다."""
        mock_redis = _make_mock_redis(zcard_return=5)  # 5회 < 기본 10회
        with patch("monglepick.api.chat.get_redis", return_value=mock_redis):
            # 예외 없이 통과해야 함
            await _check_upload_rate_limit("192.168.1.1")

    @pytest.mark.asyncio
    async def test_over_limit_raises_429(self):
        """제한 초과 시 429 에러가 발생한다."""
        mock_redis = _make_mock_redis(zcard_return=10)  # 10회 >= 기본 10회
        with patch("monglepick.api.chat.get_redis", return_value=mock_redis):
            with pytest.raises(ValueError, match="429"):
                await _check_upload_rate_limit("10.0.0.1")
            # 한도 초과 시 방금 추가한 타임스탬프가 롤백(zrem)되어야 함
            mock_redis.zrem.assert_called_once()

    @pytest.mark.asyncio
    async def test_redis_failure_graceful_degradation(self):
        """Redis 연결 실패 시 요청을 차단하지 않고 통과한다 (graceful degradation)."""
        mock_redis = AsyncMock()
        mock_pipe = AsyncMock()
        mock_pipe.execute = AsyncMock(side_effect=ConnectionError("Redis down"))
        mock_pipe.zremrangebyscore = AsyncMock()
        mock_pipe.zcard = AsyncMock()
        mock_pipe.zadd = AsyncMock()
        mock_pipe.expire = AsyncMock()
        mock_redis.pipeline = lambda: mock_pipe

        with patch("monglepick.api.chat.get_redis", return_value=mock_redis):
            # ConnectionError 발생해도 ValueError 없이 통과해야 함
            await _check_upload_rate_limit("10.0.0.2")

    @pytest.mark.asyncio
    async def test_different_ips_use_different_keys(self):
        """서로 다른 IP는 독립적인 Redis 키를 사용한다."""
        keys_used = []

        def make_capturing_pipe():
            """호출마다 새 mock pipe를 생성하여 키를 캡처한다.
            Redis pipeline 메서드는 동기(버퍼링)이므로 MagicMock을 사용한다."""
            pipe = MagicMock()
            pipe.execute = AsyncMock(return_value=[0, 0, 1, True])

            def capture_key(key, *args, **kwargs):
                keys_used.append(key)

            pipe.zremrangebyscore = MagicMock(side_effect=capture_key)
            return pipe

        mock_redis = AsyncMock()
        mock_redis.pipeline = make_capturing_pipe

        with patch("monglepick.api.chat.get_redis", return_value=mock_redis):
            await _check_upload_rate_limit("ip-a")
            await _check_upload_rate_limit("ip-b")

        # 서로 다른 키가 사용되었는지 확인
        assert len(keys_used) == 2
        assert keys_used[0] != keys_used[1]
        assert "ip-a" in keys_used[0]
        assert "ip-b" in keys_used[1]


# ============================================================
# DecompressionBomb 방어 테스트
# ============================================================


class TestDecompressionBombDefense:
    """Pillow DecompressionBomb 방어 설정 테스트."""

    def test_max_image_pixels_set_in_resize(self):
        """_resize_image_bytes가 Image.MAX_IMAGE_PIXELS를 설정한다."""
        from PIL import Image as PILImage

        from monglepick.api.chat import _resize_image_bytes
        from monglepick.config import settings

        # 유효한 1x1 JPEG 이미지 생성
        import io
        img = PILImage.new("RGB", (1, 1), color=(255, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        tiny_jpeg = buf.getvalue()

        # _resize_image_bytes 호출 → MAX_IMAGE_PIXELS가 설정되어야 함
        _resize_image_bytes(tiny_jpeg)
        assert PILImage.MAX_IMAGE_PIXELS == settings.IMAGE_MAX_PIXELS
