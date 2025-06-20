const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const app = express();
const PORT = process.env.PORT || 8080;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.post('/api/company-report', async (req, res) => {
    try {
        const { companyName, stockCode } = req.body;
        
        if (!companyName) {
            return res.status(400).json({ error: '기업명을 입력해주세요.' });
        }

        console.log(`레포트 생성 시작: ${companyName}`);

        // 모든 서비스를 순차적으로 로드
        const dartService = require('./services/dartService');
        const naverService = require('./services/naverService');
        const financeService = require('./services/financeService');

        let resolvedStockCode = stockCode;
        let basicInfo = null;
        let financialInfo = null;
        let newsInfo = null;
        let stockInfo = null;

        // 1. 기업명으로 종목코드 검색
        if (!stockCode) {
            try {
                const searchResults = await dartService.searchCompanyByName(companyName);
                if (searchResults && searchResults.length > 0) {
                    resolvedStockCode = searchResults[0].stockCode;
                    console.log(`종목코드 검색 결과: ${resolvedStockCode}`);
                }
            } catch (error) {
                console.log('종목코드 검색 실패:', error.message);
            }
        }

        // 2. 기업 기본정보
        if (resolvedStockCode) {
            try {
                const corpDict = await dartService.getCorpCodeDict();
                const corpInfo = corpDict[resolvedStockCode];
                if (corpInfo) {
                    basicInfo = await dartService.getCompanyInfo(corpInfo.corpCode, resolvedStockCode);
                }
            } catch (error) {
                console.log('기업정보 조회 실패:', error.message);
            }
        }

        if (!basicInfo) {
            basicInfo = {
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

        // 3. 재무정보 (시간 제한)
        if (resolvedStockCode) {
            try {
                const timeout = new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Timeout')), 10000)
                );
                financialInfo = await Promise.race([
                    dartService.getFinancialInfo(resolvedStockCode),
                    timeout
                ]);
            } catch (error) {
                console.log('재무정보 조회 실패:', error.message);
                financialInfo = { error: '재무정보 조회 시간 초과' };
            }
        }

        // 4. 뉴스정보 (시간 제한)
        try {
            const timeout = new Promise((_, reject) => 
                setTimeout(() => reject(new Error('Timeout')), 15000)
            );
            newsInfo = await Promise.race([
                naverService.getNewsInfo(companyName, 5),
                timeout
            ]);
        } catch (error) {
            console.log('뉴스정보 조회 실패:', error.message);
            newsInfo = { 
                articles: [], 
                summary: '뉴스 조회 시간 초과',
                sentimentSummary: { positive: 0, neutral: 0, negative: 0 },
                total: 0 
            };
        }

        // 5. 주가정보 (시간 제한)
        if (resolvedStockCode) {
            try {
                const timeout = new Promise((_, reject) => 
                    setTimeout(() => reject(new Error('Timeout')), 15000)
                );
                stockInfo = await Promise.race([
                    financeService.getStockInfo(resolvedStockCode),
                    timeout
                ]);
            } catch (error) {
                console.log('주가정보 조회 실패:', error.message);
                stockInfo = { error: '주가정보 조회 시간 초과' };
            }
        }

        const report = {
            companyName,
            stockCode: resolvedStockCode,
            basicInfo,
            financialInfo,
            newsInfo,
            stockInfo,
            generatedAt: new Date().toISOString()
        };

        console.log(`레포트 생성 완료: ${companyName}`);
        res.json(report);

    } catch (error) {
        console.error('레포트 생성 오류:', error);
        res.status(500).json({ error: '레포트 생성 중 오류가 발생했습니다.' });
    }
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`서버가 포트 ${PORT}에서 실행 중입니다.`);
    console.log(`브라우저에서 http://localhost:${PORT} 으로 접속하세요.`);
});