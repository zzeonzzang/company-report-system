const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const dartService = require('./services/dartService');
const financeService = require('./services/financeService');
const naverService = require('./services/naverService');

const app = express();
const PORT = process.env.PORT || 3001;

// 미들웨어
app.use(cors());
app.use(express.json());
app.use(express.static('public'));

// 홈페이지
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 테스트 엔드포인트
app.get('/test', (req, res) => {
    res.json({ status: 'OK', message: '서버가 정상 작동중입니다!' });
});

// 메인 API - 실제 API 통합
app.post('/api/company-report', async (req, res) => {
    const { companyName, stockCode } = req.body;
    
    if (!companyName) {
        return res.status(400).json({ error: '기업명을 입력해주세요.' });
    }

    console.log(`실제 API 레포트 요청: ${companyName}`);
    
    try {
        const report = { companyName, generatedAt: new Date().toISOString() };
        
        // 1. 기업 기본정보 (종목코드 자동 검색)
        let resolvedStockCode = stockCode;
        if (!resolvedStockCode) {
            const stockCodeResult = await dartService.getStockCodeByName(companyName);
            resolvedStockCode = stockCodeResult;
        }
        
        if (!resolvedStockCode) {
            return res.status(404).json({ error: '해당 기업의 종목코드를 찾을 수 없습니다.' });
        }
        
        report.stockCode = resolvedStockCode;
        
        // 병렬로 모든 API 호출
        const [basicInfo, financialInfo, stockInfo, newsInfo] = await Promise.all([
            dartService.getBasicInfo(resolvedStockCode).catch(err => ({ error: err.message })),
            dartService.getFinancialInfo(resolvedStockCode).catch(err => ({ error: err.message })),
            financeService.getStockInfo(resolvedStockCode).catch(err => ({ error: err.message })),
            naverService.getNews(companyName).catch(err => ({ error: err.message }))
        ]);
        
        report.basicInfo = basicInfo;
        report.financialInfo = financialInfo;
        report.stockInfo = stockInfo;
        report.newsInfo = newsInfo;
        
        res.json(report);
        console.log(`실제 API 레포트 완료: ${companyName}`);
        
    } catch (error) {
        console.error('레포트 생성 오류:', error);
        res.status(500).json({ 
            error: '레포트 생성 중 오류가 발생했습니다.',
            details: error.message 
        });
    }
});

// 서버 시작
app.listen(PORT, '0.0.0.0', () => {
    console.log(`🚀 기업종합레포트 서버가 포트 ${PORT}에서 실행중입니다!`);
    console.log(`📱 브라우저에서 접속하세요:`);
    console.log(`   http://localhost:${PORT}`);
    console.log(`   http://127.0.0.1:${PORT}`);
    console.log(`   http://172.29.184.249:${PORT}`);
});

// 오류 처리
process.on('uncaughtException', (err) => {
    console.error('Uncaught Exception:', err);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('Unhandled Rejection at:', promise, 'reason:', reason);
});