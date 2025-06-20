const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const dartService = require('./services/dartService');
const financeService = require('./services/financeService');
const naverService = require('./services/naverService');

const app = express();

// 미들웨어
app.use(cors({
    origin: '*',
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'Authorization']
}));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));
app.use(express.static(path.join(__dirname, 'public')));

// 홈페이지
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// 테스트 엔드포인트
app.get('/test', (req, res) => {
    res.json({ 
        status: 'OK', 
        message: '새로운 서버가 정상 작동중입니다!',
        timestamp: new Date().toISOString(),
        port: process.env.PORT || 4000
    });
});

// 헬스체크
app.get('/health', (req, res) => {
    res.json({ 
        status: 'healthy',
        uptime: process.uptime(),
        timestamp: new Date().toISOString()
    });
});

// 메인 API - 실제 API 통합
app.post('/api/company-report', async (req, res) => {
    const startTime = Date.now();
    const { companyName, stockCode } = req.body;
    
    console.log(`[${new Date().toISOString()}] 레포트 요청: ${companyName}`);
    
    if (!companyName) {
        return res.status(400).json({ 
            error: '기업명을 입력해주세요.',
            timestamp: new Date().toISOString()
        });
    }

    try {
        const report = { 
            companyName, 
            requestId: Date.now(),
            generatedAt: new Date().toISOString() 
        };
        
        console.log(`[${report.requestId}] 종목코드 검색 시작...`);
        
        // 1. 기업 기본정보 (종목코드 자동 검색)
        let resolvedStockCode = stockCode;
        if (!resolvedStockCode) {
            try {
                resolvedStockCode = await dartService.getStockCodeByName(companyName);
                console.log(`[${report.requestId}] 종목코드 찾음: ${resolvedStockCode}`);
            } catch (error) {
                console.error(`[${report.requestId}] 종목코드 검색 실패:`, error);
            }
        }
        
        if (!resolvedStockCode) {
            console.log(`[${report.requestId}] 종목코드를 찾을 수 없음`);
            return res.status(404).json({ 
                error: `'${companyName}' 기업의 종목코드를 찾을 수 없습니다. 정확한 기업명을 입력해주세요.`,
                suggestion: '예: 삼성전자, SK하이닉스, LG전자',
                timestamp: new Date().toISOString()
            });
        }
        
        report.stockCode = resolvedStockCode;
        console.log(`[${report.requestId}] API 호출 시작 - 종목코드: ${resolvedStockCode}`);
        
        // 병렬로 모든 API 호출 (타임아웃 설정)
        const apiCalls = [
            Promise.race([
                dartService.getBasicInfo(resolvedStockCode),
                new Promise((_, reject) => setTimeout(() => reject(new Error('기업정보 조회 타임아웃')), 15000))
            ]).catch(err => ({ error: `기업정보: ${err.message}` })),
            
            Promise.race([
                dartService.getFinancialInfo(resolvedStockCode),
                new Promise((_, reject) => setTimeout(() => reject(new Error('재무정보 조회 타임아웃')), 20000))
            ]).catch(err => ({ error: `재무정보: ${err.message}` })),
            
            Promise.race([
                financeService.getStockInfo(resolvedStockCode),
                new Promise((_, reject) => setTimeout(() => reject(new Error('주가정보 조회 타임아웃')), 25000))
            ]).catch(err => ({ error: `주가정보: ${err.message}` })),
            
            Promise.race([
                naverService.getNews(companyName),
                new Promise((_, reject) => setTimeout(() => reject(new Error('뉴스정보 조회 타임아웃')), 15000))
            ]).catch(err => ({ error: `뉴스정보: ${err.message}` }))
        ];
        
        console.log(`[${report.requestId}] 모든 API 호출 중...`);
        const [basicInfo, financialInfo, stockInfo, newsInfo] = await Promise.all(apiCalls);
        
        report.basicInfo = basicInfo;
        report.financialInfo = financialInfo;
        report.stockInfo = stockInfo;
        report.newsInfo = newsInfo;
        report.processingTime = Date.now() - startTime;
        
        console.log(`[${report.requestId}] 레포트 완료 (${report.processingTime}ms)`);
        
        res.json(report);
        
    } catch (error) {
        console.error(`레포트 생성 오류:`, error);
        res.status(500).json({ 
            error: '레포트 생성 중 오류가 발생했습니다.',
            details: error.message,
            timestamp: new Date().toISOString()
        });
    }
});

// 에러 핸들링 미들웨어
app.use((error, req, res, next) => {
    console.error('서버 오류:', error);
    res.status(500).json({
        error: '서버 내부 오류가 발생했습니다.',
        timestamp: new Date().toISOString()
    });
});

// 404 핸들링
app.use((req, res) => {
    res.status(404).json({
        error: '요청한 경로를 찾을 수 없습니다.',
        path: req.path,
        timestamp: new Date().toISOString()
    });
});

// 서버 시작 함수
async function startFreshServer() {
    const PORT = process.env.PORT || 4000;
    
    try {
        const server = app.listen(PORT, '0.0.0.0', () => {
            console.log('\n🆕 ===============================');
            console.log(`🚀 새로운 기업종합레포트 서버 시작!`);
            console.log(`📡 포트: ${PORT}`);
            console.log(`⏰ 시작 시간: ${new Date().toLocaleString('ko-KR')}`);
            console.log(`===============================`);
            console.log(`\n📱 브라우저에서 접속하세요:`);
            console.log(`   ✅ http://localhost:${PORT}`);
            console.log(`   ✅ http://127.0.0.1:${PORT}`);
            console.log(`   ✅ http://172.29.184.249:${PORT}`);
            console.log(`\n🧪 테스트 URL:`);
            console.log(`   http://localhost:${PORT}/test`);
            console.log(`   http://localhost:${PORT}/health\n`);
        });
        
        server.on('error', (error) => {
            if (error.code === 'EADDRINUSE') {
                console.log(`⚠️  포트 ${PORT}가 사용중입니다. 다른 포트로 재시도...`);
                setTimeout(() => startFreshServer(PORT + 1), 1000);
            } else {
                console.error('서버 시작 오류:', error);
            }
        });

        // 우아한 종료
        const gracefulShutdown = (signal) => {
            console.log(`\n${signal} 신호 받음. 서버를 우아하게 종료합니다...`);
            server.close(() => {
                console.log('✅ 서버가 정상적으로 종료되었습니다.');
                process.exit(0);
            });
        };

        process.on('SIGTERM', () => gracefulShutdown('SIGTERM'));
        process.on('SIGINT', () => gracefulShutdown('SIGINT'));

    } catch (error) {
        console.error('❌ 서버 시작 실패:', error);
        process.exit(1);
    }
}

// 프로세스 오류 처리
process.on('uncaughtException', (error) => {
    console.error('❌ Uncaught Exception:', error);
    process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
    console.error('❌ Unhandled Rejection:', reason);
    console.error('Promise:', promise);
});

// 서버 시작
startFreshServer();