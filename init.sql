-- ============================================================
-- 몽글픽 MySQL 전체 스키마 (39개 테이블)
-- ============================================================
--
-- v4_t2 최종 개발문서 기준 + 설계서 §4-8 + Qdrant/Neo4j/ES 미러링.
-- Spring Boot 백엔드, AI Agent, monglepick-recommend 3개 서비스에서 공유.
--
-- ── 영화/사용자 기본 (9개) ────────────────────────────
--   1. movies                — 영화 경량 참조 (Qdrant 미러링)
--   2. users                 — 사용자 기본 정보 (인증 포함)
--   3. admin                 — 관리자 계정
--   4. user_preferences      — 사용자 취향 (JSON)
--   5. watch_history         — 시청 이력 + 평점
--   6. user_wishlist         — 찜 목록
--   7. likes                 — 영화 좋아요
--   8. fav_genre             — 선호 장르 (온보딩)
--   9. fav_movie             — 최애 영화 (온보딩)
--
-- ── AI 추천 (3개) ────────────────────────────────────
--  10. recommendation_log    — 추천 이력 로그
--  11. recommendation_feedback — 추천 피드백
--  12. event_logs            — 유저 이벤트 로그 (추천 이벤트 추적)
--
-- ── AI Agent 전용 (2개) ──────────────────────────────
--  13. movie_mentions        — 커뮤니티 영화 언급 집계
--  14. user_achievements     — 사용자 업적
--  15. toxicity_log          — 비속어 검출 로그
--  16. chat_session_archive  — 대화 세션 아카이브
--
-- ── 커뮤니티 (7개) ───────────────────────────────────
--  17. category              — 게시판 상위 카테고리
--  18. category_child        — 게시판 하위 카테고리
--  19. posts                 — 커뮤니티 게시글
--  20. post_comment          — 게시글 댓글
--  21. post_like             — 게시글 좋아요
--  22. post_declaration      — 게시글/댓글 신고
--  23. reviews               — 영화 리뷰
--
-- ── 도장깨기 (4개) ───────────────────────────────────
--  24. roadmap_courses       — 도장깨기 코스
--  25. course_review         — 도장깨기 코스 인증/리뷰
--  26. quiz_attempts         — 퀴즈 도전 기록
--
-- ── 개인화/플레이리스트 (3개) ────────────────────────
--  27. playlist              — 플레이리스트
--  28. playlist_item         — 플레이리스트 아이템
--  29. calander              — 사용자 스케줄/캘린더
--
-- ── 검색 (monglepick-recommend, 3개) ─────────────────
--  30. search_history        — 사용자별 최근 검색 이력
--  31. trending_keywords     — 인기 검색어 집계
--  32. worldcup_results      — 이상형 월드컵 결과
--
-- ── 포인트/리워드 (v4_t2 통합 재화, 4개) ─────────────
--  33. user_points           — 유저 포인트 잔액
--  34. points_history        — 포인트 변동 이력
--  35. point_items           — 포인트 교환 아이템 (AI 추천 이용권 포함)
--  36. user_attendance       — 출석 체크
--
-- ── 결제/구독 (3개) ─────────────────────────────────────
--  37. subscription_plans   — 구독 상품 마스터
--  38. user_subscriptions   — 사용자 구독 현황
--  39. payment_orders       — 결제 주문 (Toss Payments)
--
-- ※ 기존 크레딧 5개 테이블 삭제 → 포인트+구독+결제로 재구성
-- ============================================================

CREATE DATABASE IF NOT EXISTS monglepick
    CHARACTER SET utf8mb4
    COLLATE utf8mb4_unicode_ci;

USE monglepick;


-- ============================================================
-- 1. movies — 영화 경량 참조 테이블
-- ============================================================
-- Qdrant/Neo4j/ES에 상세 데이터가 있고, MySQL에는 Spring Boot가
-- 참조할 경량 컬럼만 저장한다. Qdrant scroll → 배치 INSERT로 동기화.
--
-- movie_id: TMDB ID(숫자), KOBIS 코드(영문 포함), KMDb ID(숫자) 등
-- 다양한 소스의 ID가 공존하므로 VARCHAR(50)으로 통일.
-- ============================================================
CREATE TABLE IF NOT EXISTS movies (
    movie_id        VARCHAR(50)  NOT NULL PRIMARY KEY COMMENT '영화 ID (TMDB/KOBIS/KMDb)',
    title           VARCHAR(500) NOT NULL             COMMENT '한국어 제목',
    title_en        VARCHAR(500) DEFAULT NULL          COMMENT '영어 제목',
    poster_path     VARCHAR(500) DEFAULT NULL          COMMENT 'TMDB 포스터 경로',
    backdrop_path   VARCHAR(500) DEFAULT NULL          COMMENT 'TMDB 배경 이미지 경로',
    release_year    INT          DEFAULT NULL          COMMENT '개봉 연도',
    runtime         INT          DEFAULT NULL          COMMENT '상영 시간 (분)',
    rating          FLOAT        DEFAULT NULL          COMMENT '평균 평점 (0~10)',
    vote_count      INT          DEFAULT NULL          COMMENT '투표 수',
    popularity_score FLOAT       DEFAULT NULL          COMMENT 'TMDB 인기도 점수',
    genres          JSON         DEFAULT NULL          COMMENT '장르 목록 ["액션","드라마"]',
    director        VARCHAR(200) DEFAULT NULL          COMMENT '감독 이름',
    cast            JSON         DEFAULT NULL          COMMENT '주연 배우 목록 ["배우1","배우2"]',
    certification   VARCHAR(50)  DEFAULT NULL          COMMENT '관람등급 (전체관람가, 12세 등)',
    trailer_url     VARCHAR(500) DEFAULT NULL          COMMENT 'YouTube 트레일러 URL',
    overview        TEXT         DEFAULT NULL          COMMENT '줄거리',
    tagline         VARCHAR(500) DEFAULT NULL          COMMENT '태그라인',
    imdb_id         VARCHAR(20)  DEFAULT NULL          COMMENT 'IMDb ID (tt로 시작)',
    original_language VARCHAR(10) DEFAULT NULL         COMMENT '원본 언어 코드 (en, ko 등)',
    collection_name VARCHAR(200) DEFAULT NULL          COMMENT '프랜차이즈/컬렉션 이름',
    -- KOBIS 보강 컬럼
    kobis_movie_cd  VARCHAR(20)  DEFAULT NULL          COMMENT 'KOBIS 영화 코드',
    sales_acc       BIGINT       DEFAULT NULL          COMMENT '누적 매출액 (KRW)',
    audience_count  BIGINT       DEFAULT NULL          COMMENT '관객수',
    screen_count    INT          DEFAULT NULL          COMMENT '최대 상영 스크린 수',
    kobis_watch_grade VARCHAR(50) DEFAULT NULL         COMMENT 'KOBIS 관람등급',
    kobis_open_dt   VARCHAR(10)  DEFAULT NULL          COMMENT 'KOBIS 개봉일 (YYYYMMDD)',
    -- KMDb 보강 컬럼
    kmdb_id         VARCHAR(50)  DEFAULT NULL          COMMENT 'KMDb 영화 ID',
    awards          TEXT         DEFAULT NULL          COMMENT '수상 내역',
    filming_location TEXT        DEFAULT NULL          COMMENT '촬영 장소',
    -- 데이터 출처 추적
    source          VARCHAR(20)  DEFAULT NULL          COMMENT '데이터 출처 (tmdb/kaggle/kobis/kmdb)',
    -- 타임스탬프
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    -- 인덱스
    INDEX idx_movies_title (title(100)),
    INDEX idx_movies_release_year (release_year),
    INDEX idx_movies_rating (rating),
    INDEX idx_movies_popularity (popularity_score),
    INDEX idx_movies_director (director),
    INDEX idx_movies_source (source),
    INDEX idx_movies_imdb_id (imdb_id),
    INDEX idx_movies_kobis_cd (kobis_movie_cd),
    INDEX idx_movies_kmdb_id (kmdb_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 2. users — 사용자 기본 정보 (인증 포함)
-- ============================================================
-- Spring Boot 회원가입 시 생성. AI Agent는 읽기 전용.
-- Kaggle 시드 유저는 user_id = 'kaggle_{userId}' 형태로 구분.
--
-- v4_t2 기준 컬럼 통합:
--   - 기존 8컬럼 + 신규 7컬럼 (password_hash, provider, provider_id,
--     user_role, user_birth, option_term, required_term)
--   - 민규DB mongle_users 13컬럼 기준으로 확장
--   - password_hash: BCrypt 암호화 (소셜 로그인 시 NULL 가능)
--   - provider: 로그인 제공자 (LOCAL, NAVER, KAKAO, GOOGLE)
-- ============================================================
CREATE TABLE IF NOT EXISTS users (
    user_id         VARCHAR(50)  NOT NULL PRIMARY KEY COMMENT '사용자 ID (UUID)',
    nickname        VARCHAR(100) DEFAULT NULL          COMMENT '닉네임',
    email           VARCHAR(200) DEFAULT NULL          COMMENT '이메일',
    password_hash   VARCHAR(255) DEFAULT NULL          COMMENT '비밀번호 (BCrypt, 소셜 로그인 시 NULL)',
    provider        ENUM('LOCAL','NAVER','KAKAO','GOOGLE')
                                 NOT NULL DEFAULT 'LOCAL'
                                                       COMMENT '로그인 제공자',
    provider_id     VARCHAR(200) DEFAULT NULL          COMMENT '소셜 제공자 고유 ID',
    user_role       VARCHAR(20)  NOT NULL DEFAULT 'USER'
                                                       COMMENT '역할 (USER, ADMIN)',
    profile_image   VARCHAR(500) DEFAULT NULL          COMMENT '프로필 이미지 URL',
    user_birth      VARCHAR(20)  DEFAULT NULL          COMMENT '생년월일 (YYYYMMDD)',
    age_group       VARCHAR(10)  DEFAULT NULL          COMMENT '연령대 (10대, 20대 등)',
    gender          VARCHAR(10)  DEFAULT NULL          COMMENT '성별 (M/F/O)',
    option_term     BOOLEAN      DEFAULT FALSE         COMMENT '선택 약관 동의 여부',
    required_term   BOOLEAN      DEFAULT FALSE         COMMENT '필수 약관 동의 여부',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_users_email (email),
    UNIQUE KEY uk_users_provider_id (provider, provider_id),
    INDEX idx_users_created_at (created_at),
    INDEX idx_users_role (user_role)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 3. admin — 관리자 계정
-- ============================================================
-- v4_t2 t2_10 기준. 관리자 전용 계정 테이블.
-- users 테이블과 별도로 관리자 권한/접근을 분리한다.
-- ============================================================
CREATE TABLE IF NOT EXISTS admin (
    admin_id        BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '연결된 사용자 ID',
    admin_role      VARCHAR(50)  NOT NULL DEFAULT 'ADMIN'
                                                       COMMENT '관리자 역할 (ADMIN, SUPER_ADMIN)',
    is_active       BOOLEAN      DEFAULT TRUE          COMMENT '활성화 여부',
    last_login_at   TIMESTAMP    DEFAULT NULL           COMMENT '마지막 로그인 시각',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_admin_user (user_id),
    CONSTRAINT fk_admin_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 4. user_preferences — 사용자 취향 프로필
-- ============================================================
-- §6-4 preference_refiner에서 추출한 선호 조건을 누적 저장.
-- JSON 필드로 유연한 스키마 지원.
-- ============================================================
CREATE TABLE IF NOT EXISTS user_preferences (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    preferred_genres JSON        DEFAULT NULL           COMMENT '선호 장르 ["액션","SF"]',
    preferred_moods  JSON        DEFAULT NULL           COMMENT '선호 무드 ["스릴","감동"]',
    preferred_directors JSON     DEFAULT NULL           COMMENT '선호 감독 ["봉준호"]',
    preferred_actors JSON        DEFAULT NULL           COMMENT '선호 배우 ["송강호"]',
    preferred_eras   JSON        DEFAULT NULL           COMMENT '선호 시대 ["2020s"]',
    excluded_genres  JSON        DEFAULT NULL           COMMENT '제외 장르 ["호러"]',
    preferred_platforms JSON     DEFAULT NULL           COMMENT '선호 OTT ["넷플릭스"]',
    preferred_certification VARCHAR(50) DEFAULT NULL    COMMENT '선호 관람등급',
    -- 누적 대화에서 학습된 추가 선호 (자유 형식)
    extra_preferences JSON      DEFAULT NULL           COMMENT '추가 선호 조건 (키-값 자유 형식)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_user_preferences (user_id),
    CONSTRAINT fk_user_preferences_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 4. watch_history — 시청 이력 + 평점
-- ============================================================
-- 사용자가 본 영화와 평점을 기록한다.
-- Kaggle ratings.csv (26M행) 시드 데이터를 여기에 적재.
-- ============================================================
CREATE TABLE IF NOT EXISTS watch_history (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '영화 ID',
    rating          FLOAT        DEFAULT NULL           COMMENT '사용자 평점 (0.5~5.0)',
    watched_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP COMMENT '시청 일시',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_watch_user (user_id),
    INDEX idx_watch_movie (movie_id),
    INDEX idx_watch_user_movie (user_id, movie_id),
    INDEX idx_watch_watched_at (watched_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 5. user_wishlist — 찜 목록
-- ============================================================
-- 사용자가 '나중에 볼 영화'로 찜한 영화 목록.
-- ============================================================
CREATE TABLE IF NOT EXISTS user_wishlist (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '영화 ID',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_wishlist_user_movie (user_id, movie_id),
    INDEX idx_wishlist_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 7. likes — 영화 좋아요
-- ============================================================
-- v4_t2 t2_09 기준. 사용자가 영화에 좋아요를 누르면 기록.
-- user_wishlist(찜)과 별도로 좋아요를 관리한다.
-- deleted_at이 NULL이면 활성, NOT NULL이면 좋아요 취소.
-- ============================================================
CREATE TABLE IF NOT EXISTS likes (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '영화 ID',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    deleted_at      TIMESTAMP    DEFAULT NULL           COMMENT '좋아요 취소 시각 (NULL=활성)',
    UNIQUE KEY uk_likes_user_movie (user_id, movie_id),
    INDEX idx_likes_user (user_id),
    INDEX idx_likes_movie (movie_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 8. fav_genre — 선호 장르 (온보딩)
-- ============================================================
-- v4_t2 t2_09 기준. 온보딩 시 사용자가 선택한 선호 장르.
-- 원안은 fav_genre_1~6 개별 컬럼이었으나, 정규화하여
-- 사용자당 여러 행으로 저장한다. (최대 6개 장르 선택 가능)
-- user_preferences.preferred_genres (JSON)와 병행 사용.
-- ============================================================
CREATE TABLE IF NOT EXISTS fav_genre (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    genre_name      VARCHAR(50)  NOT NULL              COMMENT '장르명 (액션, SF, 드라마 등)',
    priority        INT          DEFAULT 0              COMMENT '선호 우선순위 (1이 가장 높음)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_fav_genre_user_genre (user_id, genre_name),
    INDEX idx_fav_genre_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 9. fav_movie — 최애 영화 (온보딩)
-- ============================================================
-- v4_t2 t2_09 기준. 온보딩 시 사용자가 선택한 최애 영화.
-- 원안은 fav_movie_1~6 개별 컬럼이었으나, 정규화하여
-- 사용자당 여러 행으로 저장한다. (최대 6개 영화 선택 가능)
-- ============================================================
CREATE TABLE IF NOT EXISTS fav_movie (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '영화 ID',
    priority        INT          DEFAULT 0              COMMENT '선호 우선순위 (1이 가장 높음)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_fav_movie_user_movie (user_id, movie_id),
    INDEX idx_fav_movie_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 10. recommendation_log — 추천 이력 로그
-- ============================================================
-- AI Agent가 추천을 생성할 때마다 기록한다.
-- 기존 recommendations 테이블을 대체.
-- ============================================================
CREATE TABLE IF NOT EXISTS recommendation_log (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    session_id      VARCHAR(36)  NOT NULL              COMMENT '대화 세션 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '추천된 영화 ID',
    reason          TEXT         NOT NULL               COMMENT '추천 이유 (AI 생성)',
    score           FLOAT        NOT NULL               COMMENT '최종 추천 점수',
    cf_score        FLOAT        DEFAULT NULL           COMMENT 'CF 점수',
    cbf_score       FLOAT        DEFAULT NULL           COMMENT 'CBF 점수',
    hybrid_score    FLOAT        DEFAULT NULL           COMMENT '하이브리드 합산 점수',
    genre_match     FLOAT        DEFAULT NULL           COMMENT '장르 일치도',
    mood_match      FLOAT        DEFAULT NULL           COMMENT '무드 일치도',
    rank_position   INT          DEFAULT NULL           COMMENT '추천 순위 (1~5)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_reclog_user (user_id),
    INDEX idx_reclog_session (session_id),
    INDEX idx_reclog_movie (movie_id),
    INDEX idx_reclog_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 7. recommendation_feedback — 추천 피드백
-- ============================================================
-- 사용자가 추천 결과에 대해 남긴 피드백 (좋아요/싫어요/이미 봤어요).
-- ============================================================
CREATE TABLE IF NOT EXISTS recommendation_feedback (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    recommendation_id BIGINT     NOT NULL               COMMENT '추천 로그 ID',
    feedback_type   ENUM('like','dislike','watched','not_interested')
                                 NOT NULL               COMMENT '피드백 유형',
    comment         TEXT         DEFAULT NULL           COMMENT '사용자 코멘트',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_feedback_user_rec (user_id, recommendation_id),
    INDEX idx_feedback_rec (recommendation_id),
    CONSTRAINT fk_feedback_rec FOREIGN KEY (recommendation_id)
        REFERENCES recommendation_log(id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 12. event_logs — 유저 이벤트 로그
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 정한나). 사용자의 행동 이벤트를 기록.
-- 추천 점수 산정, 분석, A/B 테스트 등에 활용.
-- event_type: view(조회), click(클릭), recommend(추천),
--             search(검색), wishlist(찜), rating(평점)
-- ============================================================
CREATE TABLE IF NOT EXISTS event_logs (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    movie_id        VARCHAR(50)  DEFAULT NULL           COMMENT '영화 ID (영화 관련 이벤트)',
    event_type      VARCHAR(50)  NOT NULL              COMMENT '이벤트 유형 (view, click, recommend, search, wishlist, rating)',
    recommend_score FLOAT        DEFAULT NULL           COMMENT '추천 관련 점수',
    metadata        JSON         DEFAULT NULL           COMMENT '이벤트 추가 메타데이터',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_event_user (user_id),
    INDEX idx_event_movie (movie_id),
    INDEX idx_event_type (event_type),
    INDEX idx_event_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 13. movie_mentions — 커뮤니티 영화 언급 집계
-- ============================================================
-- §8 콘텐츠 분석 에이전트가 커뮤니티에서 영화 언급을 수집·집계.
-- 기간별 버즈량 추적에 사용.
-- ============================================================
CREATE TABLE IF NOT EXISTS movie_mentions (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '영화 ID',
    source          VARCHAR(50)  NOT NULL              COMMENT '소스 (reddit, twitter, naver 등)',
    mention_count   INT          DEFAULT 0              COMMENT '언급 횟수',
    sentiment_avg   FLOAT        DEFAULT NULL           COMMENT '평균 감성 점수 (-1~1)',
    period_start    DATE         NOT NULL               COMMENT '집계 기간 시작',
    period_end      DATE         NOT NULL               COMMENT '집계 기간 종료',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_mention_movie_source_period (movie_id, source, period_start),
    INDEX idx_mention_movie (movie_id),
    INDEX idx_mention_period (period_start, period_end)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 9. user_achievements — 사용자 업적 (도장깨기)
-- ============================================================
-- §9 로드맵 에이전트의 도장깨기 코스 달성 기록.
-- ============================================================
CREATE TABLE IF NOT EXISTS user_achievements (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    achievement_type VARCHAR(50) NOT NULL               COMMENT '업적 유형 (course_complete, quiz_pass 등)',
    achievement_key VARCHAR(100) NOT NULL               COMMENT '업적 키 (코스 ID, 퀴즈 ID 등)',
    achieved_at     TIMESTAMP    DEFAULT CURRENT_TIMESTAMP COMMENT '달성 일시',
    metadata        JSON         DEFAULT NULL           COMMENT '업적 메타데이터 (점수, 순위 등)',
    UNIQUE KEY uk_achievement (user_id, achievement_type, achievement_key),
    INDEX idx_achievement_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 10. toxicity_log — 비속어 검출 로그
-- ============================================================
-- §8 콘텐츠 분석 에이전트가 검출한 비속어/유해 표현 로그.
-- ============================================================
CREATE TABLE IF NOT EXISTS toxicity_log (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  DEFAULT NULL           COMMENT '사용자 ID (익명 가능)',
    session_id      VARCHAR(36)  DEFAULT NULL           COMMENT '대화 세션 ID',
    input_text      TEXT         NOT NULL               COMMENT '원본 입력 텍스트',
    toxicity_score  FLOAT        NOT NULL               COMMENT '유해도 점수 (0~1)',
    toxicity_type   VARCHAR(50)  DEFAULT NULL           COMMENT '유해 유형 (profanity, hate 등)',
    action_taken    VARCHAR(50)  DEFAULT 'flagged'      COMMENT '조치 (flagged, blocked, warned)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_toxicity_user (user_id),
    INDEX idx_toxicity_session (session_id),
    INDEX idx_toxicity_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 11. chat_session_archive — 대화 세션 아카이브
-- ============================================================
-- Redis에서 TTL 만료 전 대화 세션을 MySQL에 아카이브.
-- 장기 분석 및 학습 데이터로 활용.
-- ============================================================
CREATE TABLE IF NOT EXISTS chat_session_archive (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    session_id      VARCHAR(36)  NOT NULL              COMMENT '대화 세션 ID',
    messages        JSON         NOT NULL               COMMENT '전체 대화 메시지 배열',
    turn_count      INT          DEFAULT 0              COMMENT '대화 턴 수',
    intent_summary  JSON         DEFAULT NULL           COMMENT '의도 분류 요약',
    started_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP COMMENT '대화 시작 시각',
    ended_at        TIMESTAMP    DEFAULT NULL           COMMENT '대화 종료 시각',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_session (session_id),
    INDEX idx_archive_user (user_id),
    INDEX idx_archive_started_at (started_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 16. category — 게시판 상위 카테고리
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 이민수). 커뮤니티 게시판의 상위 카테고리.
-- 예: 영화, 드라마, 공연 등.
-- ============================================================
CREATE TABLE IF NOT EXISTS category (
    category_id     BIGINT       AUTO_INCREMENT PRIMARY KEY,
    up_category     VARCHAR(100) NOT NULL              COMMENT '상위 카테고리명 (영화, 드라마, 공연 등)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_category_name (up_category)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 17. category_child — 게시판 하위 카테고리
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 이민수). 상위 카테고리 하의 세부 분류.
-- 예: 영화 > 리뷰, 영화 > 잡담, 영화 > 뉴스 등.
-- ============================================================
CREATE TABLE IF NOT EXISTS category_child (
    down_category_id BIGINT      AUTO_INCREMENT PRIMARY KEY,
    category_id     BIGINT       NOT NULL              COMMENT '상위 카테고리 ID',
    category_child  VARCHAR(100) NOT NULL              COMMENT '하위 카테고리명 (리뷰, 잡담, 뉴스 등)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_category_child (category_id, category_child),
    CONSTRAINT fk_category_child_parent FOREIGN KEY (category_id) REFERENCES category(category_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 18. posts — 커뮤니티 게시글
-- ============================================================
-- Spring Boot 커뮤니티 기능. AI Agent는 읽기 전용 분석.
-- v4_t2 t2_09 기준 (담당: 이민수). 기존 board 테이블을 posts로 매핑.
-- category_id FK로 카테고리 연결.
-- ============================================================
CREATE TABLE IF NOT EXISTS posts (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '작성자 ID',
    category_id     BIGINT       DEFAULT NULL           COMMENT '카테고리 ID (category 테이블 FK)',
    title           VARCHAR(300) NOT NULL               COMMENT '게시글 제목',
    content         TEXT         NOT NULL               COMMENT '게시글 본문',
    movie_id        VARCHAR(50)  DEFAULT NULL           COMMENT '관련 영화 ID (없을 수 있음)',
    like_count      INT          DEFAULT 0              COMMENT '좋아요 수',
    comment_count   INT          DEFAULT 0              COMMENT '댓글 수',
    view_count      INT          DEFAULT 0              COMMENT '조회 수',
    status          VARCHAR(20)  DEFAULT 'active'      COMMENT '상태 (active, deleted, hidden)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_posts_user (user_id),
    INDEX idx_posts_movie (movie_id),
    INDEX idx_posts_category (category_id),
    INDEX idx_posts_created_at (created_at),
    INDEX idx_posts_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 19. post_comment — 게시글 댓글
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 이민수). 게시글에 달린 댓글.
-- is_deleted: 소프트 삭제 지원 (삭제된 댓글도 "삭제된 댓글입니다" 표시용).
-- ============================================================
CREATE TABLE IF NOT EXISTS post_comment (
    comment_id      BIGINT       AUTO_INCREMENT PRIMARY KEY,
    post_id         BIGINT       NOT NULL              COMMENT '게시글 ID',
    category_id     BIGINT       DEFAULT NULL           COMMENT '카테고리 ID',
    user_id         VARCHAR(50)  NOT NULL              COMMENT '작성자 ID',
    content         TEXT         NOT NULL               COMMENT '댓글 내용',
    is_deleted      BOOLEAN      DEFAULT FALSE          COMMENT '삭제 여부 (소프트 삭제)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_comment_post (post_id),
    INDEX idx_comment_user (user_id),
    INDEX idx_comment_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 20. post_like — 게시글 좋아요
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 이민수). 게시글 좋아요.
-- 사용자당 게시글당 1개만 가능 (UNIQUE 제약).
-- ============================================================
CREATE TABLE IF NOT EXISTS post_like (
    like_id         BIGINT       AUTO_INCREMENT PRIMARY KEY,
    post_id         BIGINT       NOT NULL              COMMENT '게시글 ID',
    category_id     BIGINT       DEFAULT NULL           COMMENT '카테고리 ID',
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_post_like_user_post (user_id, post_id),
    INDEX idx_post_like_post (post_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 21. post_declaration — 게시글/댓글 신고
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 이민수). 부적절한 게시글/댓글 신고.
-- target_type: post(게시글), comment(댓글) 구분.
-- status: pending(대기), approved(승인), rejected(기각).
-- ============================================================
CREATE TABLE IF NOT EXISTS post_declaration (
    declaration_id  BIGINT       AUTO_INCREMENT PRIMARY KEY,
    post_id         BIGINT       NOT NULL              COMMENT '신고 대상 게시글 ID',
    category_id     BIGINT       DEFAULT NULL           COMMENT '카테고리 ID',
    user_id         VARCHAR(50)  NOT NULL              COMMENT '신고자 ID',
    reported_user_id VARCHAR(50) NOT NULL              COMMENT '피신고자 ID',
    target_type     VARCHAR(20)  NOT NULL DEFAULT 'post'
                                                       COMMENT '신고 대상 유형 (post, comment)',
    declaration_content TEXT     NOT NULL               COMMENT '신고 사유',
    toxicity_score  FLOAT        DEFAULT NULL           COMMENT '유해도 점수 (AI 분석)',
    status          VARCHAR(20)  DEFAULT 'pending'     COMMENT '처리 상태 (pending, approved, rejected)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_declaration_post (post_id),
    INDEX idx_declaration_user (user_id),
    INDEX idx_declaration_reported (reported_user_id),
    INDEX idx_declaration_status (status)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 22. reviews — 영화 리뷰
-- ============================================================
-- 사용자가 작성한 영화 리뷰. 평점 + 텍스트.
-- ============================================================
CREATE TABLE IF NOT EXISTS reviews (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '작성자 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '영화 ID',
    rating          FLOAT        NOT NULL               COMMENT '평점 (0.5~5.0)',
    content         TEXT         DEFAULT NULL           COMMENT '리뷰 본문',
    spoiler         BOOLEAN      DEFAULT FALSE          COMMENT '스포일러 포함 여부',
    like_count      INT          DEFAULT 0              COMMENT '좋아요 수',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_review_user_movie (user_id, movie_id),
    INDEX idx_reviews_movie (movie_id),
    INDEX idx_reviews_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 14. roadmap_courses — 도장깨기 코스
-- ============================================================
-- §9 로드맵 에이전트가 생성하는 영화 도장깨기 코스.
-- 각 코스는 테마별 영화 목록 + 순서를 포함.
-- ============================================================
CREATE TABLE IF NOT EXISTS roadmap_courses (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    course_id       VARCHAR(50)  NOT NULL              COMMENT '코스 고유 ID',
    title           VARCHAR(300) NOT NULL               COMMENT '코스 제목 ("봉준호 감독 마스터 코스")',
    description     TEXT         DEFAULT NULL           COMMENT '코스 설명',
    theme           VARCHAR(100) DEFAULT NULL           COMMENT '코스 테마 (감독, 장르, 시대 등)',
    movie_ids       JSON         NOT NULL               COMMENT '코스에 포함된 영화 ID 배열 (순서 보장)',
    movie_count     INT          NOT NULL               COMMENT '코스 내 영화 수',
    difficulty      ENUM('beginner','intermediate','advanced')
                                 DEFAULT 'beginner'    COMMENT '난이도',
    quiz_enabled    BOOLEAN      DEFAULT FALSE          COMMENT '퀴즈 포함 여부',
    created_by      VARCHAR(50)  DEFAULT 'ai_agent'    COMMENT '생성자 (ai_agent 또는 user_id)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_course_id (course_id),
    INDEX idx_course_theme (theme),
    INDEX idx_course_difficulty (difficulty)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 25. course_review — 도장깨기 코스 인증/리뷰
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 김민규). 도장깨기 코스 내 개별 영화
-- 시청 인증 및 한줄 리뷰. 사용자가 영화를 보고 인증글을 남긴다.
-- ============================================================
CREATE TABLE IF NOT EXISTS course_review (
    course_review_id BIGINT      AUTO_INCREMENT PRIMARY KEY,
    course_id       VARCHAR(50)  NOT NULL              COMMENT '코스 ID (roadmap_courses.course_id)',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '인증 영화 ID',
    user_id         VARCHAR(50)  NOT NULL              COMMENT '작성자 ID',
    review_text     TEXT         DEFAULT NULL           COMMENT '인증 리뷰 텍스트',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_course_review (course_id, movie_id, user_id),
    INDEX idx_course_review_user (user_id),
    INDEX idx_course_review_course (course_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 26. quiz_attempts — 퀴즈 도전 기록
-- ============================================================
-- 도장깨기 코스의 퀴즈 도전 기록.
-- ============================================================
CREATE TABLE IF NOT EXISTS quiz_attempts (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    course_id       VARCHAR(50)  NOT NULL              COMMENT '코스 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '퀴즈 대상 영화 ID',
    question        TEXT         NOT NULL               COMMENT '퀴즈 질문',
    user_answer     TEXT         NOT NULL               COMMENT '사용자 답변',
    correct_answer  TEXT         NOT NULL               COMMENT '정답',
    is_correct      BOOLEAN      NOT NULL               COMMENT '정답 여부',
    score           INT          DEFAULT 0              COMMENT '획득 점수',
    attempted_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_quiz_user (user_id),
    INDEX idx_quiz_course (course_id),
    INDEX idx_quiz_user_course (user_id, course_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 28. playlist — 플레이리스트
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 김민규). 사용자가 만든 영화 플레이리스트.
-- 여러 영화를 그룹핑하여 "볼 영화 목록", "최고의 액션 영화" 등 관리.
-- ============================================================
CREATE TABLE IF NOT EXISTS playlist (
    playlist_id     BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '소유자 ID',
    playlist_name   VARCHAR(200) NOT NULL              COMMENT '플레이리스트 제목',
    description     TEXT         DEFAULT NULL           COMMENT '플레이리스트 설명',
    is_public       BOOLEAN      DEFAULT FALSE          COMMENT '공개 여부',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_playlist_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 29. playlist_item — 플레이리스트 아이템
-- ============================================================
-- v4_t2 t2_09 기준. 플레이리스트에 포함된 개별 영화.
-- sort_order로 영화 순서 관리.
-- ============================================================
CREATE TABLE IF NOT EXISTS playlist_item (
    item_id         BIGINT       AUTO_INCREMENT PRIMARY KEY,
    playlist_id     BIGINT       NOT NULL              COMMENT '플레이리스트 ID',
    movie_id        VARCHAR(50)  NOT NULL              COMMENT '영화 ID',
    sort_order      INT          DEFAULT 0              COMMENT '정렬 순서',
    added_at        TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_playlist_item (playlist_id, movie_id),
    INDEX idx_playlist_item_playlist (playlist_id),
    CONSTRAINT fk_playlist_item_playlist FOREIGN KEY (playlist_id)
        REFERENCES playlist(playlist_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 30. calander — 사용자 스케줄/캘린더
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 김민규). 사용자의 영화 관련 일정 관리.
-- 예: 영화 개봉일 알림, 시청 일정 등.
-- 테이블명은 v4_t2 원안의 'calander' (오타) 유지.
-- ============================================================
CREATE TABLE IF NOT EXISTS calander (
    calander_id     BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    schedule_title  VARCHAR(200) NOT NULL              COMMENT '일정 제목',
    schedule_description TEXT    DEFAULT NULL           COMMENT '일정 설명',
    start_time      DATETIME     NOT NULL              COMMENT '시작 시각',
    end_time        DATETIME     DEFAULT NULL           COMMENT '종료 시각',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_calander_user (user_id),
    INDEX idx_calander_start (start_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 31. search_history — 사용자별 최근 검색 이력
-- ============================================================
-- monglepick-recommend 소유. 사용자의 최근 검색 키워드를 저장한다.
-- 동일 사용자+키워드 조합은 UNIQUE 제약으로 중복 방지,
-- ON DUPLICATE KEY UPDATE로 searched_at만 갱신.
-- ============================================================
CREATE TABLE IF NOT EXISTS search_history (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    keyword         VARCHAR(200) NOT NULL              COMMENT '검색 키워드',
    searched_at     TIMESTAMP    DEFAULT CURRENT_TIMESTAMP COMMENT '검색 시각',
    -- 인덱스
    INDEX idx_search_history_user_time (user_id, searched_at),
    UNIQUE KEY uk_search_history_user_keyword (user_id, keyword)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 17. trending_keywords — 인기 검색어 집계
-- ============================================================
-- monglepick-recommend 소유. 전체 사용자의 검색 키워드를 집계하여
-- 인기 검색어 순위를 제공한다. search_count 내림차순으로 조회.
-- ============================================================
CREATE TABLE IF NOT EXISTS trending_keywords (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    keyword         VARCHAR(200) NOT NULL UNIQUE       COMMENT '검색 키워드',
    search_count    INT          NOT NULL DEFAULT 0    COMMENT '누적 검색 횟수',
    last_searched_at TIMESTAMP   DEFAULT CURRENT_TIMESTAMP COMMENT '마지막 검색 시각',
    -- 인덱스
    INDEX idx_trending_count (search_count)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 18. worldcup_results — 이상형 월드컵 결과 저장
-- ============================================================
-- monglepick-recommend 소유. 온보딩 시 이상형 월드컵 결과를 저장하여
-- 사용자 장르 선호도를 분석하고, 초기 추천에 활용한다.
-- selection_log와 genre_preferences는 JSON 형태로 저장.
-- ============================================================
CREATE TABLE IF NOT EXISTS worldcup_results (
    id              BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    round_size      INT          NOT NULL DEFAULT 16   COMMENT '라운드 크기 (16 또는 32)',
    winner_movie_id VARCHAR(50)  NOT NULL              COMMENT '우승 영화 ID',
    runner_up_movie_id VARCHAR(50) DEFAULT NULL         COMMENT '준우승 영화 ID',
    semi_final_movie_ids TEXT    DEFAULT NULL           COMMENT '4강 영화 ID 목록 (JSON 배열)',
    selection_log   TEXT         DEFAULT NULL           COMMENT '전체 라운드별 선택 로그 (JSON)',
    genre_preferences TEXT       DEFAULT NULL           COMMENT '분석된 장르 선호도 (JSON)',
    onboarding_completed BOOLEAN NOT NULL DEFAULT FALSE COMMENT '온보딩 완료 여부',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    -- 인덱스
    INDEX idx_worldcup_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 34. user_points — 유저 포인트 잔액
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 김민규). 게이미피케이션 포인트 시스템.
-- 출석 체크, 리뷰 작성, 도장깨기 완료 등으로 포인트를 획득하고,
-- 포인트 아이템(point_items)과 교환할 수 있다.
-- 크레딧(AI 추천 과금)과 별도의 게이미피케이션 재화.
-- ============================================================
CREATE TABLE IF NOT EXISTS user_points (
    point_id        BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    point_have      INT          NOT NULL DEFAULT 0    COMMENT '현재 보유 포인트',
    total_earned    INT          NOT NULL DEFAULT 0    COMMENT '누적 획득 포인트',
    daily_earned    INT          NOT NULL DEFAULT 0    COMMENT '오늘 획득 포인트',
    daily_reset     DATE         DEFAULT NULL           COMMENT '일일 리셋 기준일',
    user_grade      VARCHAR(20)  DEFAULT 'BRONZE'     COMMENT '등급 (BRONZE, SILVER, GOLD, PLATINUM)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    UNIQUE KEY uk_user_points (user_id),
    CONSTRAINT fk_user_points_user FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 35. points_history — 포인트 변동 이력
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 김민규). 모든 포인트 증감 내역 기록.
-- point_type: earn(획득), spend(사용), expire(만료), bonus(보너스)
-- ============================================================
CREATE TABLE IF NOT EXISTS points_history (
    point_history_id BIGINT      AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    point_change    INT          NOT NULL              COMMENT '변동량 (+획득, -사용)',
    point_after     INT          NOT NULL              COMMENT '변동 후 잔액',
    point_type      VARCHAR(50)  NOT NULL              COMMENT '변동 유형 (earn, spend, expire, bonus)',
    description     VARCHAR(300) DEFAULT NULL           COMMENT '변동 사유',
    reference_id    VARCHAR(100) DEFAULT NULL           COMMENT '참조 ID (이벤트, 아이템 등)',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_points_history_user (user_id),
    INDEX idx_points_history_type (point_type),
    INDEX idx_points_history_created_at (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 36. point_items — 포인트 교환 아이템
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 김민규). 포인트로 교환 가능한 아이템 마스터.
-- 예: AI 추천 1회 이용권 (100P), 프로필 꾸미기 (50P) 등.
-- ============================================================
CREATE TABLE IF NOT EXISTS point_items (
    point_item_id   BIGINT       AUTO_INCREMENT PRIMARY KEY,
    item_name       VARCHAR(200) NOT NULL              COMMENT '아이템명',
    item_description TEXT        DEFAULT NULL           COMMENT '아이템 설명',
    item_price      INT          NOT NULL              COMMENT '필요 포인트',
    item_category   VARCHAR(50)  DEFAULT 'general'    COMMENT '아이템 카테고리',
    is_active       BOOLEAN      DEFAULT TRUE          COMMENT '판매 활성화 여부',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 37. user_attendance — 출석 체크
-- ============================================================
-- v4_t2 t2_09 기준 (담당: 김민규). 사용자 출석 체크 기록.
-- 연속 출석일(streak)에 따라 보너스 포인트 지급.
-- check_date별 UNIQUE로 하루 1회만 체크 가능.
-- ============================================================
CREATE TABLE IF NOT EXISTS user_attendance (
    attendance_id   BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '사용자 ID',
    check_date      DATE         NOT NULL              COMMENT '출석 날짜',
    streak_count    INT          DEFAULT 1              COMMENT '연속 출석일 수',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    UNIQUE KEY uk_attendance_user_date (user_id, check_date),
    INDEX idx_attendance_user (user_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 37. subscription_plans — 구독 상품 마스터
-- ============================================================
-- Toss Payments 연동. 운영팀이 관리하는 구독 상품 정의.
-- 삭제 대신 is_active=FALSE로 비활성화 (기존 FK 참조 보존).
-- ============================================================
CREATE TABLE IF NOT EXISTS subscription_plans (
    plan_id         BIGINT       AUTO_INCREMENT PRIMARY KEY,
    plan_code       VARCHAR(50)  NOT NULL UNIQUE       COMMENT '상품 코드 (monthly_basic 등)',
    name            VARCHAR(100) NOT NULL              COMMENT '상품명',
    period_type     ENUM('MONTHLY','YEARLY') NOT NULL  COMMENT '구독 주기',
    price           INT          NOT NULL              COMMENT '가격 (KRW)',
    points_per_period INT        NOT NULL              COMMENT '주기당 지급 포인트',
    description     VARCHAR(500) DEFAULT NULL           COMMENT '상품 설명',
    is_active       BOOLEAN      DEFAULT TRUE          COMMENT '판매 활성화 여부',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 38. user_subscriptions — 사용자 구독 현황
-- ============================================================
-- 사용자의 구독 상태를 관리한다. active/cancelled/expired 3가지 상태.
-- 한 사용자가 active 구독을 동시에 2개 이상 가질 수 없다 (서비스 레이어 검증).
-- auto_renew=TRUE이면 만료일에 자동 결제 시도.
-- ============================================================
CREATE TABLE IF NOT EXISTS user_subscriptions (
    subscription_id BIGINT       AUTO_INCREMENT PRIMARY KEY,
    user_id         VARCHAR(50)  NOT NULL              COMMENT '구독자 ID',
    plan_id         BIGINT       NOT NULL              COMMENT '구독 상품 ID',
    status          ENUM('ACTIVE','CANCELLED','EXPIRED') NOT NULL DEFAULT 'ACTIVE' COMMENT '구독 상태',
    started_at      TIMESTAMP    NOT NULL              COMMENT '구독 시작일',
    expires_at      TIMESTAMP    NOT NULL              COMMENT '만료 예정일',
    cancelled_at    TIMESTAMP    DEFAULT NULL           COMMENT '취소 시각',
    auto_renew      BOOLEAN      DEFAULT TRUE          COMMENT '자동 갱신 여부',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_sub_user (user_id),
    INDEX idx_sub_status (status),
    INDEX idx_sub_expires (expires_at),
    CONSTRAINT fk_sub_plan FOREIGN KEY (plan_id) REFERENCES subscription_plans(plan_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 39. payment_orders — 결제 주문
-- ============================================================
-- Toss Payments 결제를 추적하는 테이블.
-- order_id는 UUID로 PG에 전달되며, PK로 사용한다.
-- pending→completed/failed 전이, 환불 시 refunded.
-- ============================================================
CREATE TABLE IF NOT EXISTS payment_orders (
    order_id        VARCHAR(50)  NOT NULL PRIMARY KEY   COMMENT '주문 UUID (PG에 전달)',
    user_id         VARCHAR(50)  NOT NULL              COMMENT '주문자 ID',
    order_type      ENUM('POINT_PACK','SUBSCRIPTION') NOT NULL COMMENT '주문 유형',
    amount          INT          NOT NULL              COMMENT '결제 금액 (KRW)',
    points_amount   INT          DEFAULT NULL           COMMENT '지급될 포인트 (포인트팩인 경우)',
    plan_id         BIGINT       DEFAULT NULL           COMMENT '구독 상품 ID (구독인 경우)',
    status          ENUM('PENDING','COMPLETED','FAILED','REFUNDED') NOT NULL DEFAULT 'PENDING' COMMENT '주문 상태',
    pg_transaction_id VARCHAR(100) DEFAULT NULL         COMMENT 'PG사 거래 ID',
    pg_provider     VARCHAR(50)  DEFAULT NULL           COMMENT 'PG사 이름',
    failed_reason   VARCHAR(500) DEFAULT NULL           COMMENT '실패 사유',
    completed_at    TIMESTAMP    DEFAULT NULL           COMMENT '결제 완료 시각',
    created_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
    updated_at      TIMESTAMP    DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
    INDEX idx_order_user (user_id),
    INDEX idx_order_status (status),
    INDEX idx_order_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;


-- ============================================================
-- 포인트 아이템 초기 시드 데이터
-- ============================================================
-- AI 추천 이용권, 프로필 꾸미기 등 포인트 교환 아이템.
-- ============================================================
INSERT IGNORE INTO point_items (item_name, item_description, item_price, item_category)
VALUES
    ('AI 추천 1회',     'AI 영화 추천 1회 이용',        100, 'ai_feature'),
    ('AI 추천 5회 팩',  'AI 영화 추천 5회 이용 (10% 할인)', 450, 'ai_feature'),
    ('프로필 테마',     '프로필 커스텀 테마 적용',       200, 'profile'),
    ('칭호 변경',       '커뮤니티 닉네임 칭호 변경',     150, 'profile'),
    ('도장깨기 힌트',   '퀴즈 힌트 1회 사용',            50, 'roadmap');


-- ============================================================
-- 구독 상품 초기 시드 데이터
-- ============================================================
-- 월간/연간 기본/프리미엄 4종. 구독 시 주기마다 포인트 자동 지급.
-- ============================================================
INSERT IGNORE INTO subscription_plans (plan_code, name, period_type, price, points_per_period, description)
VALUES
    ('monthly_basic',   '월간 기본',     'monthly',  3900,  3000,   '매월 3,000 포인트 지급 (AI 추천 30회)'),
    ('monthly_premium', '월간 프리미엄',  'monthly',  7900,  8000,   '매월 8,000 포인트 지급 (AI 추천 80회)'),
    ('yearly_basic',    '연간 기본',     'yearly',   39000, 40000,  '연간 40,000 포인트 지급 (AI 추천 400회)'),
    ('yearly_premium',  '연간 프리미엄',  'yearly',   79000, 100000, '연간 100,000 포인트 지급 (AI 추천 1,000회)');
