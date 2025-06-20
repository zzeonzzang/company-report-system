const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const dartService = require('./services/dartService');
const naverService = require('./services/naverService');
const financeService = require('./services/financeService');
const jobService = require('./services/jobService');

const app = express();
const PORT = process.env.PORT || 8080;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 테스트 엔드포인트 추가
app.get('/api/test', (req, res) => {
    res.json({ 
        status: 'ok', 
        message: '서버가 정상 작동합니다!',
        timestamp: new Date().toISOString()
    });
});

// 간단한 회사 정보 테스트
app.post('/api/test-company', (req, res) => {
    const { companyName } = req.body;
    res.json({
        companyName: companyName || '테스트회사',
        basicInfo: {
            회사명: companyName || '테스트회사',
            영문명: 'Test Company',
            대표자: '홍길동',
            설립일: '20000101',
            업종: '테스트업종',
            본사주소: '서울시 테스트구'
        },
        status: 'success'
    });
});

app.post('/api/company-report', async (req, res) => {
    try {
        const { companyName, stockCode } = req.body;
        
        if (!companyName) {
            return res.status(400).json({ error: '기업명을 입력해주세요.' });
        }

        console.log(`레포트 생성 시작: ${companyName}, 종목코드: ${stockCode || '없음'}`);

        let resolvedStockCode = stockCode;
        let basicInfo = null;

        if (!stockCode) {
            const searchResults = await dartService.searchCompanyByName(companyName);
            if (searchResults && searchResults.length > 0) {
                resolvedStockCode = searchResults[0].stockCode;
                console.log(`종목코드 자동 검색 결과: ${resolvedStockCode}`);
            }
        }

        if (resolvedStockCode) {
            const corpDict = await dartService.getCorpCodeDict();
            const corpInfo = corpDict[resolvedStockCode];
            if (corpInfo) {
                basicInfo = await dartService.getCompanyInfo(corpInfo.corpCode, resolvedStockCode);
            }
        }

        if (!basicInfo) {
            basicInfo = await getBasicCompanyInfo(companyName);
        }

        // 타임아웃과 함께 각 서비스 호출 (타임아웃 시간 대폭 증가)
        const [financialInfo, newsInfo, stockInfo, jobInfo] = await Promise.allSettled([
            resolvedStockCode ? 
                Promise.race([
                    dartService.getFinancialInfo(resolvedStockCode),
                    new Promise((_, reject) => setTimeout(() => reject(new Error('DART API 타임아웃')), 120000))
                ]) : null,
            Promise.race([
                naverService.getNews(companyName, 10),
                new Promise((_, reject) => setTimeout(() => reject(new Error('Naver API 타임아웃')), 90000))
            ]),
            resolvedStockCode ? 
                Promise.race([
                    financeService.getStockInfo(resolvedStockCode),
                    new Promise((_, reject) => setTimeout(() => reject(new Error('Finance API 타임아웃')), 60000))
                ]) : null,
            Promise.resolve(jobService.getCompanyJobInfo(companyName))
        ]).then(results => results.map(result => 
            result.status === 'fulfilled' ? result.value : 
            { error: result.reason?.message || '데이터 조회 실패' }
        ));

        const report = {
            companyName,
            stockCode: resolvedStockCode,
            basicInfo,
            financialInfo,
            newsInfo,
            stockInfo,
            jobInfo,
            generatedAt: new Date().toISOString(),
            summary: generateReportSummary(financialInfo, newsInfo, stockInfo)
        };

        console.log(`레포트 생성 완료: ${companyName}`);
        res.json(report);
    } catch (error) {
        console.error('Error generating report:', error);
        res.status(500).json({ error: '레포트 생성 중 오류가 발생했습니다.' });
    }
});

app.get('/api/search-company/:name', async (req, res) => {
    try {
        const { name } = req.params;
        const results = await dartService.searchCompanyByName(name);
        res.json(results);
    } catch (error) {
        console.error('Company search error:', error);
        res.status(500).json({ error: '기업 검색 중 오류가 발생했습니다.' });
    }
});

app.get('/api/market-index', async (req, res) => {
    try {
        const marketData = await financeService.getMarketIndex();
        res.json(marketData);
    } catch (error) {
        console.error('Market index error:', error);
        res.status(500).json({ error: '시장지수 조회 중 오류가 발생했습니다.' });
    }
});

async function getBasicCompanyInfo(companyName) {
    return {
        회사명: companyName,
        영문명: '정보 조회 중...',
        대표자: '정보 조회 중...',
        설립일: '정보 조회 중...',
        본사주소: '정보 조회 중...',
        홈페이지: '정보 조회 중...',
        업종: '정보 조회 중...',
        결산월: '정보 조회 중...'
    };
}

function generateReportSummary(financialInfo, newsInfo, stockInfo) {
    const summary = {};
    
    if (financialInfo && !financialInfo.error) {
        summary.financial = {
            status: '양호',
            revenue: dartService.formatAmount(financialInfo.revenue),
            profit: dartService.formatAmount(financialInfo.netIncome)
        };
    }
    
    if (newsInfo && newsInfo.sentimentSummary) {
        const { positiveRatio, negativeRatio } = newsInfo.sentimentSummary;
        summary.news = {
            sentiment: positiveRatio > negativeRatio ? '긍정적' : negativeRatio > positiveRatio ? '부정적' : '중립적',
            totalArticles: newsInfo.total,
            positiveRatio
        };
    }
    
    if (stockInfo && stockInfo.technicalIndicators) {
        summary.stock = {
            currentPrice: financeService.formatCurrency(stockInfo.currentPrice),
            signal: financeService.getTechnicalSignal(stockInfo.technicalIndicators),
            changeRate: financeService.formatPercentage(stockInfo.changeRate)
        };
    }
    
    return summary;
}

app.listen(PORT, 'localhost', () => {
    console.log(`서버가 포트 ${PORT}에서 실행 중입니다.`);
    console.log(`브라우저에서 http://localhost:${PORT} 또는 http://127.0.0.1:${PORT} 으로 접속하세요.`);
});