# -*- coding: utf-8 -*-
"""
YouTube 댓글 분석기 (Streamlit Cloud용)
- 영상 URL만 넣으면 댓글 수집 → 통계 → 멋진 워드클라우드 생성
- API 키는 Streamlit Secrets의 YOUYUBR_API_KRT 에서 읽어옵니다.
"""

import re
import glob
from collections import Counter
from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from wordcloud import WordCloud, STOPWORDS

# ----------------------------------------------------------------------
# 기본 설정
# ----------------------------------------------------------------------
st.set_page_config(
    page_title="YouTube 댓글 분석기",
    page_icon="🎬",
    layout="wide",
)

st.markdown(
    """
    <style>
    .big-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0;}
    .sub-title {color: #888; margin-top: 0.2rem;}
    div[data-testid="stMetricValue"] {font-size: 1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<p class="big-title">🎬 YouTube 댓글 분석기</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">영상 URL을 넣으면 댓글을 수집해서 통계와 워드클라우드를 만들어 드립니다.</p>',
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# API 키 로드 (Secrets → 없으면 수동 입력)
# ----------------------------------------------------------------------
def get_api_key():
    for key_name in ("YOUYUBR_API_KRT", "YOUTUBE_API_KEY"):
        try:
            if key_name in st.secrets:
                return st.secrets[key_name]
        except Exception:
            pass
    return None


API_KEY = get_api_key()

# ----------------------------------------------------------------------
# 한글 폰트 찾기 (packages.txt 로 fonts-nanum 설치됨)
# ----------------------------------------------------------------------
def find_korean_font():
    candidates = (
        glob.glob("/usr/share/fonts/truetype/nanum/NanumGothic*.ttf")
        + glob.glob("/usr/share/fonts/**/Nanum*.ttf", recursive=True)
        + glob.glob("/usr/share/fonts/**/NotoSansCJK*.ttc", recursive=True)
        + glob.glob("/usr/share/fonts/**/NotoSansKR*.ttf", recursive=True)
    )
    return candidates[0] if candidates else None


FONT_PATH = find_korean_font()

# ----------------------------------------------------------------------
# 유틸 함수
# ----------------------------------------------------------------------
def extract_video_id(url: str):
    """다양한 형태의 유튜브 URL에서 video id 추출"""
    patterns = [
        r"(?:v=|/videos/|embed/|youtu\.be/|/v/|/e/|watch\?v=|shorts/)([A-Za-z0-9_-]{11})",
        r"^([A-Za-z0-9_-]{11})$",
    ]
    for p in patterns:
        m = re.search(p, url.strip())
        if m:
            return m.group(1)
    return None


@st.cache_data(show_spinner=False, ttl=600)
def fetch_video_info(api_key: str, video_id: str):
    youtube = build("youtube", "v3", developerKey=api_key)
    resp = (
        youtube.videos()
        .list(part="snippet,statistics", id=video_id)
        .execute()
    )
    items = resp.get("items", [])
    if not items:
        return None
    it = items[0]
    return {
        "title": it["snippet"]["title"],
        "channel": it["snippet"]["channelTitle"],
        "published": it["snippet"]["publishedAt"],
        "views": int(it["statistics"].get("viewCount", 0)),
        "likes": int(it["statistics"].get("likeCount", 0)),
        "comments": int(it["statistics"].get("commentCount", 0)),
        "thumbnail": it["snippet"]["thumbnails"].get("high", {}).get("url"),
    }


@st.cache_data(show_spinner=False, ttl=600)
def fetch_comments(api_key: str, video_id: str, max_comments: int, order: str):
    youtube = build("youtube", "v3", developerKey=api_key)
    rows, token = [], None
    while len(rows) < max_comments:
        resp = (
            youtube.commentThreads()
            .list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                order=order,          # "relevance" 또는 "time"
                textFormat="plainText",
                pageToken=token,
            )
            .execute()
        )
        for item in resp.get("items", []):
            s = item["snippet"]["topLevelComment"]["snippet"]
            rows.append(
                {
                    "author": s.get("authorDisplayName", ""),
                    "text": s.get("textDisplay", ""),
                    "likes": int(s.get("likeCount", 0)),
                    "replies": int(item["snippet"].get("totalReplyCount", 0)),
                    "published": s.get("publishedAt", ""),
                }
            )
            if len(rows) >= max_comments:
                break
        token = resp.get("nextPageToken")
        if not token:
            break
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published"] = pd.to_datetime(df["published"])
    return df


# 한국어 + 영어 불용어
KR_STOPWORDS = {
    "그리고", "그런데", "하지만", "그래서", "진짜", "정말", "너무", "완전",
    "그냥", "근데", "이거", "저거", "그거", "이런", "저런", "그런", "여기",
    "저기", "거기", "제가", "내가", "네가", "우리", "저희", "당신", "이게",
    "그게", "저게", "뭔가", "뭐지", "뭐야", "아니", "예요", "이에요", "입니다",
    "합니다", "했다", "한다", "하는", "하면", "해서", "있다", "없다", "같다",
    "같은", "같아요", "있는", "없는", "때문", "때문에", "부터", "까지", "처럼",
    "보다", "정도", "이제", "지금", "오늘", "다시", "계속", "역시", "제발",
    "그럼", "혹시", "많이", "조금", "약간", "다들", "모두", "영상", "댓글",
    "유튜브", "채널", "구독", "좋아요", "ㅋㅋ", "ㅋㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅜㅜ",
}


def tokenize(text: str):
    """한글 2글자 이상 / 영문 3글자 이상 단어 추출"""
    words = re.findall(r"[가-힣]{2,}|[A-Za-z]{3,}", text)
    out = []
    for w in words:
        lw = w.lower()
        if lw in STOPWORDS or w in KR_STOPWORDS:
            continue
        # 조사 꼬리 간단 제거 (은/는/이/가/을/를/도/만/의/에)
        if len(w) >= 3 and re.match(r"[가-힣]+$", w) and w[-1] in "은는이가을를도만의에":
            w = w[:-1]
            if len(w) < 2 or w in KR_STOPWORDS:
                continue
        out.append(w)
    return out


def make_circle_mask(size=800):
    x, y = np.ogrid[:size, :size]
    center = size / 2
    mask = (x - center) ** 2 + (y - center) ** 2 > (center - 10) ** 2
    return 255 * mask.astype(int)


def build_wordcloud(freq: Counter, colormap: str, bg: str, shape: str):
    mask = make_circle_mask() if shape == "원형" else None
    wc = WordCloud(
        font_path=FONT_PATH,
        width=1600,
        height=900 if shape != "원형" else 1600,
        background_color=None if bg == "투명" else bg,
        mode="RGBA" if bg == "투명" else "RGB",
        colormap=colormap,
        mask=mask,
        max_words=200,
        prefer_horizontal=0.92,
        relative_scaling=0.4,
        min_font_size=8,
        random_state=42,
    )
    return wc.generate_from_frequencies(freq)


# ----------------------------------------------------------------------
# 사이드바
# ----------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ 설정")

    if API_KEY:
        st.success("Secrets에서 API 키를 불러왔습니다 ✅")
    else:
        st.warning("Secrets에 키가 없습니다. 아래에 직접 입력하세요.")
        API_KEY = st.text_input("YouTube API Key", type="password")

    url = st.text_input("YouTube 영상 URL", placeholder="https://www.youtube.com/watch?v=...")
    max_comments = st.slider("가져올 댓글 수", 100, 2000, 500, step=100)
    order = st.radio("댓글 정렬", ["relevance", "time"], horizontal=True,
                     format_func=lambda x: "인기순" if x == "relevance" else "최신순")

    st.divider()
    st.subheader("☁️ 워드클라우드 스타일")
    colormap = st.selectbox(
        "색상 테마",
        ["viridis", "plasma", "inferno", "magma", "cool", "spring",
         "autumn", "winter", "rainbow", "tab20", "Set2", "Pastel1"],
        index=1,
    )
    bg = st.selectbox("배경색", ["black", "white", "투명", "#0e1117", "#1a1a2e"])
    shape = st.radio("모양", ["직사각형", "원형"], horizontal=True)

    run = st.button("🚀 분석 시작", use_container_width=True, type="primary")

# ----------------------------------------------------------------------
# 메인 로직
# ----------------------------------------------------------------------
if run:
    if not API_KEY:
        st.error("API 키가 필요합니다. Streamlit Secrets에 `YOUYUBR_API_KRT` 를 등록하세요.")
        st.stop()

    video_id = extract_video_id(url or "")
    if not video_id:
        st.error("올바른 YouTube URL을 입력해 주세요.")
        st.stop()

    try:
        with st.spinner("영상 정보를 가져오는 중..."):
            info = fetch_video_info(API_KEY, video_id)
        if info is None:
            st.error("영상을 찾을 수 없습니다.")
            st.stop()

        with st.spinner(f"댓글 {max_comments}개를 수집하는 중..."):
            df = fetch_comments(API_KEY, video_id, max_comments, order)
    except HttpError as e:
        if e.resp.status == 403:
            st.error("API 할당량 초과 또는 댓글이 비활성화된 영상입니다.")
        else:
            st.error(f"API 오류: {e}")
        st.stop()

    if df.empty:
        st.warning("수집된 댓글이 없습니다.")
        st.stop()

    # ---------------- 영상 정보 ----------------
    col_img, col_meta = st.columns([1, 2])
    with col_img:
        if info["thumbnail"]:
            st.image(info["thumbnail"], use_container_width=True)
    with col_meta:
        st.subheader(info["title"])
        st.caption(f"📺 {info['channel']} · 게시일 {info['published'][:10]}")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("조회수", f"{info['views']:,}")
        m2.metric("좋아요", f"{info['likes']:,}")
        m3.metric("전체 댓글", f"{info['comments']:,}")
        m4.metric("수집한 댓글", f"{len(df):,}")

    st.divider()

    # ---------------- 탭 구성 ----------------
    tab_wc, tab_stats, tab_top, tab_data = st.tabs(
        ["☁️ 워드클라우드", "📊 통계", "🏆 베스트 댓글", "📄 원본 데이터"]
    )

    # ---- 워드클라우드 ----
    with tab_wc:
        all_words = []
        for t in df["text"]:
            all_words.extend(tokenize(t))
        freq = Counter(all_words)

        if not freq:
            st.warning("워드클라우드를 만들 단어가 부족합니다.")
        elif FONT_PATH is None:
            st.error(
                "한글 폰트를 찾지 못했습니다. 저장소 루트에 `packages.txt` 파일을 만들고 "
                "`fonts-nanum` 한 줄을 추가한 뒤 앱을 재배포하세요."
            )
        else:
            wc = build_wordcloud(freq, colormap, bg, shape)
            fig, ax = plt.subplots(figsize=(14, 8))
            fig.patch.set_alpha(0)
            ax.imshow(wc, interpolation="bilinear")
            ax.axis("off")
            st.pyplot(fig, use_container_width=True)

            # PNG 다운로드
            import io
            buf = io.BytesIO()
            wc.to_image().save(buf, format="PNG")
            st.download_button(
                "⬇️ 워드클라우드 PNG 다운로드",
                data=buf.getvalue(),
                file_name="wordcloud.png",
                mime="image/png",
            )

            st.subheader("🔤 최다 등장 단어 TOP 20")
            top_words = pd.DataFrame(freq.most_common(20), columns=["단어", "횟수"])
            fig_bar = px.bar(
                top_words.sort_values("횟수"),
                x="횟수", y="단어", orientation="h",
                color="횟수", color_continuous_scale=colormap.lower() if colormap.islower() else "plasma",
                height=550,
            )
            fig_bar.update_layout(coloraxis_showscale=False, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)

    # ---- 통계 ----
    with tab_stats:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("📅 날짜별 댓글 수")
            daily = df.set_index("published").resample("D").size().reset_index(name="count")
            fig_ts = px.area(daily, x="published", y="count",
                             labels={"published": "날짜", "count": "댓글 수"})
            st.plotly_chart(fig_ts, use_container_width=True)
        with c2:
            st.subheader("🕐 시간대별 댓글 분포")
            hourly = df["published"].dt.hour.value_counts().sort_index().reset_index()
            hourly.columns = ["hour", "count"]
            fig_h = px.bar(hourly, x="hour", y="count",
                           labels={"hour": "시간대 (UTC)", "count": "댓글 수"})
            st.plotly_chart(fig_h, use_container_width=True)

        c3, c4 = st.columns(2)
        with c3:
            st.subheader("👍 좋아요 분포")
            fig_l = px.histogram(df[df["likes"] > 0], x="likes", nbins=30, log_y=True,
                                 labels={"likes": "좋아요 수"})
            st.plotly_chart(fig_l, use_container_width=True)
        with c4:
            st.subheader("✍️ 댓글 길이 분포")
            df["length"] = df["text"].str.len()
            fig_len = px.histogram(df, x="length", nbins=40,
                                   labels={"length": "글자 수"})
            st.plotly_chart(fig_len, use_container_width=True)

    # ---- 베스트 댓글 ----
    with tab_top:
        st.subheader("👍 좋아요 TOP 10")
        top10 = df.nlargest(10, "likes")
        for _, row in top10.iterrows():
            with st.container(border=True):
                st.markdown(f"**{row['author']}** · 👍 {row['likes']:,} · 💬 답글 {row['replies']}")
                st.write(row["text"][:500])

    # ---- 원본 데이터 ----
    with tab_data:
        st.dataframe(df, use_container_width=True, height=500)
        csv = df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("⬇️ CSV 다운로드", csv, "comments.csv", "text/csv")
else:
    st.info("👈 왼쪽 사이드바에서 영상 URL을 입력하고 **분석 시작**을 눌러주세요.")
