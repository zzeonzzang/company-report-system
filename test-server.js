const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = 8080;

app.use(cors());
app.use(express.json());
app.use(express.static('public'));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.get('/test', (req, res) => {
    res.json({ message: '서버가 정상 작동합니다!' });
});

app.post('/api/company-report', (req, res) => {
    const { companyName } = req.body;
    
    // 간단한 테스트 응답
    res.json({
        companyName,
        basicInfo: {
            회사명: companyName,
            업종: '테스트',
            설립일: '2024-01-01'
        },
        message: '테스트 모드입니다'
    });
});

app.listen(PORT, '0.0.0.0', () => {
    console.log(`테스트 서버가 포트 ${PORT}에서 실행 중입니다.`);
    console.log(`http://localhost:${PORT} 로 접속하세요.`);
});