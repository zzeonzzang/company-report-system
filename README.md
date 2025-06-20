# 기업 종합 레포트 시스템

실시간 기업 정보, 재무 데이터, 뉴스 감정 분석, 주가 예측을 제공하는 종합 대시보드

## 주요 기능

- **실시간 기업 정보**: DART API를 통한 공시 정보 조회
- **재무 분석**: 분기별 재무 데이터 및 ML 기반 매출 예측
- **주가 예측**: RandomForest 알고리즘을 이용한 기술적 분석
- **뉴스 감정 분석**: GPT API를 활용한 뉴스 기사 감정 분석
- **채용 정보**: 기업별 채용 현황 및 공고 정보

## 기술 스택

- **Backend**: Node.js, Express.js
- **Frontend**: HTML, CSS, JavaScript, Plotly.js
- **APIs**: DART Open API, Naver News API, OpenAI GPT API
- **ML**: Python RandomForest, 기술적 지표 분석

## 설치 및 실행

1. 레포지토리 클론
```bash
git clone https://github.com/zzeonzzang/company-report-system.git
cd company-report-system
```

2. 의존성 설치
```bash
npm install
```

3. 환경변수 설정
```bash
cp .env.example .env
```
`.env` 파일을 열어 필요한 API 키들을 입력하세요.

4. 서버 실행
```bash
npm start
```

5. 브라우저에서 접속
```
http://localhost:8080
```

## API 키 발급 방법

### DART API
1. [DART 홈페이지](https://opendart.fss.or.kr/) 회원가입
2. API 키 신청 및 발급

### Naver API
1. [Naver Developers](https://developers.naver.com/apps/) 로그인
2. 애플리케이션 등록
3. 검색 API (뉴스) 추가

### OpenAI API
1. [OpenAI Platform](https://platform.openai.com/) 회원가입
2. API 키 생성
3. 사용량에 따른 요금 발생

## 프로젝트 구조

```
company-report-system/
├── public/
│   ├── index.html      # 메인 페이지
│   └── script.js       # 프론트엔드 로직
├── services/
│   ├── dartService.js  # DART API 연동
│   ├── naverService.js # Naver API 및 감정분석
│   ├── financeService.js # 주가 분석
│   ├── jobService.js   # 채용정보
│   └── revenuePredictionModel.py # ML 매출 예측
├── index.js           # 메인 서버
├── package.json       # 의존성 관리
└── .env              # 환경변수 (git에 포함되지 않음)
```

## 사용법

1. 웹페이지에서 기업명 입력 (예: 현대차, 삼성전자)
2. 레포트 생성 버튼 클릭
3. 다음 정보들을 확인:
   - 기업 기본정보
   - 채용정보
   - 분기별 재무정보 및 매출 예측
   - 주가 정보 및 기술적 분석
   - 최신 뉴스 및 감정 분석

## 라이선스

MIT License
