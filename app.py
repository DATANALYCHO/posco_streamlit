import datetime as dt
import html
import smtplib
import urllib.parse
import xml.etree.ElementTree as ET
from email.message import EmailMessage

import pandas as pd
from pypdf import PdfReader
import requests
import streamlit as st
import yfinance as yf
from openai import OpenAI


st.set_page_config(
    page_title="국내 주식 대시보드",
    page_icon="📈",
    layout="wide",
)

MODEL_NAME = "gpt-4.1-mini"


def build_report(detail: pd.DataFrame, period_label: str, latest_date, news_df: pd.DataFrame,
                 disclosure_df: pd.DataFrame, api_key: str = "") -> str:
    """Build a Korean market report from the data currently displayed in the app."""
    snapshot = detail.to_csv(index=False)
    news_context = news_df[["제목", "언론사", "게시 시각"]].to_csv(index=False) if not news_df.empty else "뉴스 없음"
    disclosure_context = disclosure_df[["접수일", "보고서명", "공시 유형"]].to_csv(index=False) if not disclosure_df.empty else "공시 없음"

    if api_key:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=MODEL_NAME,
            instructions=(
                "너는 국내 주식 데이터 보고서 작성자다. 제공된 데이터와 뉴스만 근거로 한국어 보고서를 작성하라. "
                "제목, 시장 요약, 종목별 핵심, 뉴스·공시 요약, 유의사항 순서로 작성하고 투자 권유를 하지 마라. "
                "확인할 수 없는 내용은 추정하지 말고 데이터 기준일을 명시하라."
            ),
            input=(f"기준일: {latest_date.strftime('%Y-%m-%d')} / 조회 기간: {period_label}\n\n"
                   f"주가 데이터:\n{snapshot}\n\n뉴스:\n{news_context}\n\n공시:\n{disclosure_context}"),
        )
        return response.output_text

    top = detail.sort_values(f"{period_label} 수익률", ascending=False).head(3)
    lines = [
        f"# 국내 주식 데이터·뉴스 보고서",
        f"기준일: {latest_date.strftime('%Y-%m-%d')} | 조회 기간: {period_label}",
        "",
        "## 시장 요약",
        f"조회 대상 {len(detail)}개 종목의 기간 수익률을 비교했습니다.",
        "",
        "## 수익률 상위 종목",
    ]
    for _, row in top.iterrows():
        lines.append(f"- {row['종목']}: {row[f'{period_label} 수익률']:+.2f}% (현재가 {row['현재가']:,.0f}원)")
    lines += ["", "## 최신 뉴스"]
    if news_df.empty:
        lines.append("- 수집된 뉴스가 없습니다.")
    else:
        for _, row in news_df.head(8).iterrows():
            lines.append(f"- {row['제목']} ({row['언론사'] or '출처 미상'})")
    lines += ["", "## 공시"]
    lines.append(f"- 최근 조회 공시: {len(disclosure_df)}건")
    lines += ["", "## 유의사항", "- 본 보고서는 수집 시점의 참고 자료이며 투자 권유가 아닙니다."]
    return "\n".join(lines)


def send_email_smtp(smtp_host: str, smtp_port: int, sender: str, password: str,
                    recipient: str, subject: str, body: str, security: str = "STARTTLS"):
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    smtp_class = smtplib.SMTP_SSL if security == "SSL" else smtplib.SMTP
    with smtp_class(smtp_host, smtp_port, timeout=30) as server:
        if security == "STARTTLS":
            server.ehlo()
            server.starttls()
            server.ehlo()
        server.login(sender, password)
        server.send_message(message)


def extract_pdf_text(uploaded_file) -> tuple[str, int]:
    """Extract text from an uploaded PDF for use as chatbot context."""
    reader = PdfReader(uploaded_file)
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        if text.strip():
            pages.append(f"[페이지 {page_number}]\n{text.strip()}")
    return "\n\n".join(pages), len(reader.pages)


@st.cache_data(ttl=900, show_spinner=False)
def load_news(stock_name: str, limit: int = 8) -> pd.DataFrame:
    """Fetch recent company news from Yahoo Finance via yfinance."""
    ticker = STOCKS[stock_name]
    rows = []
    for item in (yf.Ticker(ticker).news or [])[:limit]:
        content = item.get("content", item)
        title = content.get("title") or item.get("title")
        if not title:
            continue
        provider = content.get("provider", {})
        publisher = provider.get("displayName") or item.get("publisher") or ""
        link = content.get("canonicalUrl", {}).get("url") or content.get("clickThroughUrl", {}).get("url") or item.get("link") or ""
        published = content.get("pubDate") or item.get("providerPublishTime")
        if isinstance(published, (int, float)):
            published = dt.datetime.fromtimestamp(published).strftime("%Y-%m-%d %H:%M")
        elif published:
            published = str(published).replace("T", " ").replace("Z", "")[:16]
        rows.append({"제목": title, "언론사": publisher, "게시 시각": published or "", "링크": link})

    # Yahoo Finance가 뉴스를 반환하지 않는 경우 Google News RSS로 보완합니다.
    if not rows:
        query = urllib.parse.quote(f"{stock_name} 주식")
        rss_url = f"https://news.google.com/rss/search?q={query}&hl=ko&gl=KR&ceid=KR:ko"
        response = requests.get(rss_url, timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        root = ET.fromstring(response.content)
        for item in root.findall("./channel/item")[:limit]:
            title = html.unescape(item.findtext("title", "")).strip()
            link = item.findtext("link", "").strip()
            published = item.findtext("pubDate", "").strip()
            source = item.findtext("source", "").strip()
            if title:
                rows.append({"제목": title, "언론사": source or "Google News", "게시 시각": published, "링크": link})
    return pd.DataFrame(rows)


@st.cache_data(ttl=900, show_spinner=False)
def load_disclosures(stock_name: str, dart_key: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Fetch recent DART disclosures for a company."""
    response = requests.get(
        "https://opendart.fss.or.kr/api/list.json",
        params={"crtfc_key": dart_key, "corp_code": DART_CORP_CODES[stock_name],
                "bgn_de": start_date, "end_de": end_date, "page_no": 1, "page_count": 30},
        timeout=15,
    )
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "000":
        raise ValueError(f"{payload.get('status', '')} {payload.get('message', 'DART API 오류')}")
    rows = []
    for item in payload.get("list", []):
        receipt = item.get("rcept_no", "")
        rows.append({"접수일": item.get("rcept_dt", ""), "보고서명": item.get("report_nm", ""),
                     "제출인": item.get("flr_nm", ""), "공시 유형": item.get("pblntf_ty_nm", ""),
                     "원문": f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={receipt}"})
    return pd.DataFrame(rows)


STOCKS = {
    "삼성전자": "005930.KS",
    "SK하이닉스": "000660.KS",
    "LG에너지솔루션": "373220.KS",
    "현대차": "005380.KS",
    "삼성바이오로직스": "207940.KS",
    "기아": "000270.KS",
    "셀트리온": "068270.KS",
    "POSCO홀딩스": "005490.KS",
    "NAVER": "035420.KS",
    "카카오": "035720.KS",
}

DART_CORP_CODES = {
    "삼성전자": "00126380", "SK하이닉스": "00164779", "LG에너지솔루션": "01691599",
    "현대차": "00164742", "삼성바이오로직스": "00877059", "기아": "00164788",
    "셀트리온": "00413046", "POSCO홀딩스": "00155319", "NAVER": "01058704", "카카오": "00258801",
}


@st.cache_data(ttl=900, show_spinner=False)
def load_prices(period: str) -> pd.DataFrame:
    """Download adjusted close prices for the default Korean stocks."""
    tickers = list(STOCKS.values())
    data = yf.download(
        tickers=tickers,
        period=period,
        auto_adjust=False,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        return pd.DataFrame()

    # yfinance returns a MultiIndex when multiple tickers are requested.
    if isinstance(data.columns, pd.MultiIndex):
        if "Adj Close" in data.columns.get_level_values(0):
            prices = data["Adj Close"].copy()
        else:
            prices = data["Close"].copy()
    else:
        prices = data[["Adj Close"]].copy()
        prices.columns = [tickers[0]]

    prices = prices.rename(columns={ticker: name for name, ticker in STOCKS.items()})
    return prices.dropna(how="all").sort_index()


def fmt_price(value: float) -> str:
    return f"{value:,.0f}원"


def fmt_pct(value: float) -> str:
    return f"{value:+.2f}%"


st.title("📈 국내 주식 10종 대시보드")
st.caption("Yahoo Finance(yfinance)에서 수집한 국내 주식 데이터를 시각화합니다. 데이터는 최대 15분간 캐시됩니다.")

with st.sidebar:
    st.header("조회 설정")
    period_label = st.selectbox("조회 기간", ["1개월", "3개월", "6개월", "1년", "2년", "5년"], index=2)
    period_map = {"1개월": "1mo", "3개월": "3mo", "6개월": "6mo", "1년": "1y", "2년": "2y", "5년": "5y"}
    selected = st.multiselect("차트에 표시할 종목", list(STOCKS), default=list(STOCKS)[:5])
    refresh = st.button("🔄 데이터 새로고침", use_container_width=True)
    st.divider()
    st.header("🤖 AI 주식 챗봇")
    st.caption("입력한 API 키는 이 Streamlit 세션에서만 사용됩니다.")
    api_key = st.text_input("OpenAI API 키", type="password", placeholder="sk-...")
    dart_key = st.text_input("OpenDART API 키", type="password", placeholder="40자리 인증키")
    uploaded_pdf = st.file_uploader("분석할 PDF 업로드", type=["pdf"])
    st.divider()
    st.header("📧 SMTP 보고서 발송")
    smtp_host = st.text_input("SMTP 서버", value="smtp.gmail.com")
    smtp_port = st.number_input("SMTP 포트", min_value=1, max_value=65535, value=587)
    smtp_security = st.selectbox("보안 방식", ["STARTTLS", "SSL"], help="587 포트는 STARTTLS, 465 포트는 SSL을 사용하세요.")
    smtp_sender = st.text_input("발신 계정", placeholder="example@gmail.com")
    smtp_password = st.text_input("앱 비밀번호", type="password", placeholder="메일 서비스의 앱 비밀번호")
    smtp_recipient = st.text_input("수신 계정", placeholder="recipient@example.com")
    if refresh:
        st.cache_data.clear()
        st.rerun()

try:
    with st.spinner("Yahoo Finance에서 데이터를 불러오는 중입니다..."):
        prices = load_prices(period_map[period_label])
except Exception as exc:
    st.error(f"데이터를 불러오지 못했습니다: {exc}")
    st.stop()

if prices.empty:
    st.warning("조회된 데이터가 없습니다. 잠시 후 다시 시도해 주세요.")
    st.stop()

latest_date = prices.index[-1]
latest = prices.iloc[-1]
previous = prices.iloc[-2] if len(prices) > 1 else latest
change = latest - previous
change_pct = (change / previous.replace(0, pd.NA) * 100).fillna(0)

st.info(f"기준일: {latest_date.strftime('%Y-%m-%d')}  |  마지막 업데이트: {dt.datetime.now().strftime('%Y-%m-%d %H:%M')}")

st.subheader("주요 종목 현황")
cols = st.columns(5)
for col, name in zip(cols, list(STOCKS)[:5]):
    with col:
        value = latest.get(name)
        pct = change_pct.get(name)
        if pd.isna(value):
            st.metric(name, "-", "-")
        else:
            st.metric(name, fmt_price(value), fmt_pct(pct))

cols = st.columns(5)
for col, name in zip(cols, list(STOCKS)[5:]):
    with col:
        value = latest.get(name)
        pct = change_pct.get(name)
        if pd.isna(value):
            st.metric(name, "-", "-")
        else:
            st.metric(name, fmt_price(value), fmt_pct(pct))

st.divider()
left, right = st.columns([1.6, 1])

with left:
    st.subheader("주가 추이")
    chart_data = prices[selected] if selected else prices.iloc[:, :1]
    st.line_chart(chart_data, height=430)

with right:
    st.subheader("기간 수익률")
    start = prices.iloc[0]
    returns = ((latest / start - 1) * 100).dropna().sort_values(ascending=False)
    st.bar_chart(returns, height=430)

st.subheader("상세 데이터")
detail = pd.DataFrame(
    {
        "종목": prices.columns,
        "티커": [STOCKS[name] for name in prices.columns],
        "현재가": [latest.get(name) for name in prices.columns],
        "전일 대비": [change.get(name) for name in prices.columns],
        "등락률": [change_pct.get(name) for name in prices.columns],
        f"{period_label} 수익률": [((latest.get(name) / start.get(name) - 1) * 100) for name in prices.columns],
    }
)
st.dataframe(
    detail.style.format({"현재가": "{:,.0f}원", "전일 대비": "{:+,.0f}원", "등락률": "{:+.2f}%", f"{period_label} 수익률": "{:+.2f}%"}),
    use_container_width=True,
    hide_index=True,
)

st.divider()
st.subheader("📰 기업 뉴스")
news_stock = st.selectbox("뉴스를 볼 종목", list(STOCKS), key="news_stock")
try:
    with st.spinner(f"{news_stock} 관련 최신 뉴스를 불러오는 중입니다..."):
        news_df = load_news(news_stock)
except Exception as exc:
    news_df = pd.DataFrame()
    st.error(f"뉴스를 불러오지 못했습니다: {exc}")

if news_df.empty:
    st.info("현재 Yahoo Finance에서 제공하는 뉴스가 없습니다.")
else:
    for _, article in news_df.iterrows():
        title = article["제목"]
        meta = " · ".join(value for value in [article["언론사"], article["게시 시각"]] if value)
        if article["링크"]:
            st.markdown(f"- [{title}]({article['링크']})  \n  {meta}")
        else:
            st.markdown(f"- {title}  \n  {meta}")

st.divider()
st.subheader("📑 OpenDART 공시")
st.caption("금융감독원 전자공시시스템(OpenDART)의 최근 공시를 조회합니다.")
disclosure_stock = st.selectbox("공시를 볼 종목", list(STOCKS), key="disclosure_stock")
disclosure_df = pd.DataFrame()
if not dart_key:
    st.info("사이드바에 OpenDART API 키를 입력하면 최근 공시를 확인할 수 있습니다.")
else:
    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=90)
    try:
        with st.spinner(f"{disclosure_stock} 공시를 불러오는 중입니다..."):
            disclosure_df = load_disclosures(disclosure_stock, dart_key, start_date.strftime("%Y%m%d"), end_date.strftime("%Y%m%d"))
        if disclosure_df.empty:
            st.info("최근 90일 내 공시가 없습니다.")
        else:
            st.dataframe(disclosure_df, use_container_width=True, hide_index=True,
                         column_config={"원문": st.column_config.LinkColumn("원문")})
    except Exception as exc:
        st.error(f"OpenDART 공시를 불러오지 못했습니다: {exc}")

st.divider()
st.subheader("📄 데이터·뉴스 보고서")
st.caption("현재 조회된 주가, 뉴스, 공시를 바탕으로 보고서를 생성합니다. OpenAI API 키가 있으면 AI 요약을 사용합니다.")
report_col1, report_col2 = st.columns([1, 3])
with report_col1:
    generate_report = st.button("보고서 생성", use_container_width=True)
    send_report = st.button("SMTP로 보고서 보내기", use_container_width=True)

if generate_report or send_report:
    try:
        with st.spinner("보고서를 작성하는 중입니다..."):
            report_text = build_report(detail, period_label, latest_date, news_df, disclosure_df, api_key)
        st.session_state.report_text = report_text
    except Exception as exc:
        st.error(f"보고서 생성에 실패했습니다: {exc}")

if st.session_state.get("report_text"):
    report_text = st.session_state.report_text
    st.text_area("생성된 보고서", value=report_text, height=360)

if send_report:
    missing = [name for name, value in {
        "SMTP 서버": smtp_host, "발신 계정": smtp_sender, "앱 비밀번호": smtp_password,
        "수신 계정": smtp_recipient, "생성된 보고서": st.session_state.get("report_text"),
    }.items() if not value]
    if missing:
        st.warning("다음 항목을 입력하거나 먼저 실행해 주세요: " + ", ".join(missing))
    else:
        try:
            with st.spinner("SMTP 서버로 메일을 보내는 중입니다..."):
                send_email_smtp(
                    smtp_host, int(smtp_port), smtp_sender, smtp_password, smtp_recipient,
                    f"국내 주식 데이터·뉴스 보고서 ({latest_date.strftime('%Y-%m-%d')})",
                    st.session_state.report_text, smtp_security,
                )
            st.success(f"보고서를 {smtp_recipient} 주소로 보냈습니다.")
        except Exception as exc:
            st.error(f"메일 발송에 실패했습니다: {exc}")

st.divider()
st.subheader("🤖 AI 주식 챗봇")
st.caption(f"현재 화면의 {len(prices.columns)}개 종목 데이터를 바탕으로 답변합니다. 사용 모델: `{MODEL_NAME}`")

pdf_text = ""
if uploaded_pdf is not None:
    try:
        pdf_text, page_count = extract_pdf_text(uploaded_pdf)
        if pdf_text.strip():
            st.success(f"PDF 준비 완료: {uploaded_pdf.name} ({page_count}페이지)")
        else:
            st.warning("PDF에서 추출 가능한 텍스트가 없습니다. 스캔 PDF는 OCR 후 업로드해 주세요.")
    except Exception as exc:
        st.error(f"PDF를 읽지 못했습니다: {exc}")

if pdf_text:
    st.caption("PDF 내용은 질문 시 OpenAI API로 전송되어 답변 근거로 사용됩니다.")

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []

for message in st.session_state.chat_messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if not api_key:
    st.info("사이드바에 OpenAI API 키를 입력하면 주식 데이터에 대해 질문할 수 있습니다.")

if prompt := st.chat_input("예: 최근 6개월 수익률이 가장 좋은 종목과 이유를 알려줘"):
    if not api_key:
        st.warning("먼저 사이드바에 OpenAI API 키를 입력해 주세요.")
    else:
        st.session_state.chat_messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 모델이 임의로 현재가를 추정하지 않도록 화면의 최신 데이터와 기간 수익률을 함께 전달합니다.
        snapshot = detail.to_csv(index=False)
        news_context = news_df[["제목", "언론사", "게시 시각"]].to_csv(index=False) if not news_df.empty else "뉴스 없음"
        disclosure_context = disclosure_df[["접수일", "보고서명", "공시 유형"]].to_csv(index=False) if not disclosure_df.empty else "공시 없음"
        # 너무 큰 PDF가 모델 문맥을 과도하게 차지하지 않도록 앞부분을 제한합니다.
        pdf_context = pdf_text[:60000] if pdf_text else "업로드된 PDF 없음"
        conversation = "\n".join(
            f"{item['role']}: {item['content']}" for item in st.session_state.chat_messages[-8:]
        )
        instructions = (
            "너는 국내 주식 데이터 분석 보조 챗봇이다. 반드시 제공된 데이터만 근거로 답변하고, "
            "데이터에 없는 실시간 뉴스·재무정보·미래 주가를 사실처럼 말하지 마라. "
            "투자 조언이 아니라 참고용 분석임을 명시하고, 숫자는 가능한 한 종목명과 기간을 함께 제시하라. "
            "한국어로 간결하고 명확하게 답변하라.\n\n"
            f"현재 대시보드 데이터(기간: {period_label}, 기준일: {latest_date.strftime('%Y-%m-%d')}):\n{snapshot}\n\n"
            f"업로드된 PDF 내용:\n{pdf_context}\n\n"
            f"선택 종목({news_stock}) 최신 뉴스 목록:\n{news_context}\n\n"
            f"선택 종목({disclosure_stock}) 최근 OpenDART 공시:\n{disclosure_context}\n\n"
            f"대화:\n{conversation}"
        )
        try:
            with st.chat_message("assistant"):
                with st.spinner("AI가 데이터를 분석하는 중입니다..."):
                    client = OpenAI(api_key=api_key)
                    response = client.responses.create(
                        model=MODEL_NAME,
                        instructions=instructions,
                        input=prompt,
                    )
                    answer = response.output_text
                    st.markdown(answer)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
        except Exception as exc:
            st.error(f"OpenAI API 호출에 실패했습니다: {exc}")

st.caption("※ 투자 판단의 참고 자료이며, 실시간 시세가 아닐 수 있습니다. 데이터 출처: Yahoo Finance")
