const axios = require('axios');
const xml2js = require('xml2js');
const AdmZip = require('adm-zip');
const { spawn } = require('child_process');
const path = require('path');

class DartService {
    constructor() {
        this.apiKey = process.env.DART_API_KEY;
        this.baseUrl = 'https://opendart.fss.or.kr/api';
    }

    async getCorpCodeDict() {
        try {
            const response = await axios.get(`${this.baseUrl}/corpCode.xml`, {
                params: {
                    crtfc_key: this.apiKey
                },
                responseType: 'arraybuffer'
            });

            const zip = new AdmZip(response.data);
            const xmlData = zip.readAsText('CORPCODE.xml');
            
            const parser = new xml2js.Parser();
            const result = await parser.parseStringPromise(xmlData);
            
            const corpDict = {};
            if (result.result && result.result.list) {
                result.result.list.forEach(item => {
                    const corpName = item.corp_name[0].trim();
                    const stockCode = item.stock_code[0].trim();
                    const corpCode = item.corp_code[0].trim();
                    
                    if (stockCode) {
                        corpDict[stockCode] = { corpCode, corpName };
                    }
                });
            }
            
            return corpDict;
        } catch (error) {
            console.error('DART 기업코드 조회 오류:', error);
            throw error;
        }
    }

    async getCompanyInfo(corpCode, stockCode = null) {
        try {
            const response = await axios.get(`${this.baseUrl}/company.json`, {
                params: {
                    crtfc_key: this.apiKey,
                    corp_code: corpCode
                }
            });

            if (response.data.status === '000') {
                const data = response.data;
                return {
                    회사명: data.corp_name || '정보없음',
                    영문명: data.corp_name_eng || '정보없음',
                    대표자: data.ceo_nm || '정보없음',
                    설립일: data.est_dt || '정보없음',
                    본사주소: data.adres || '정보없음',
                    홈페이지: data.hm_url || '정보없음',
                    업종: data.induty || '정보없음',
                    결산월: data.acc_mt || '정보없음'
                };
            }
            
            return null;
        } catch (error) {
            console.error('DART 기업정보 조회 오류:', error);
            return null;
        }
    }

    async getFinancialInfo(stockCode) {
        if (!this.apiKey) {
            return { error: 'DART API 키가 설정되지 않았습니다.' };
        }

        try {
            const corpDict = await this.getCorpCodeDict();
            const corpInfo = corpDict[stockCode];
            
            if (!corpInfo) {
                return { error: `종목코드 '${stockCode}'에 해당하는 기업을 찾을 수 없습니다.` };
            }

            const { corpCode, corpName } = corpInfo;
            const currentYear = new Date().getFullYear();
            const currentMonth = new Date().getMonth() + 1; // 1-12
            const currentQuarter = Math.ceil(currentMonth / 3);
            
            // 최근 4분기 데이터 조회 
            const queries = [];
            let queryYear = currentYear;
            let queryQuarter = currentQuarter;
            
            // 최근 4분기 생성
            for (let i = 0; i < 4; i++) {
                queries.unshift({
                    year: queryYear,
                    quarter: queryQuarter,
                    reprt_code: this.getQuarterlyReportCode(queryQuarter)
                });
                
                // 이전 분기로 이동
                queryQuarter--;
                if (queryQuarter < 1) {
                    queryQuarter = 4;
                    queryYear--;
                }
            }
            
            const allData = [];
            
            for (const query of queries) {
                try {
                    const response = await axios.get(`${this.baseUrl}/fnlttSinglAcnt.json`, {
                        params: {
                            crtfc_key: this.apiKey,
                            corp_code: corpCode,
                            bsns_year: query.year,
                            reprt_code: query.reprt_code
                        },
                        timeout: 60000  // 60초 타임아웃 설정
                    });

                    if (response.data.status === '000' && response.data.list) {
                        const quarterData = this.parseFinancialData(response.data.list);
                        allData.push({
                            year: query.year,
                            quarter: `Q${query.quarter}`,
                            ...quarterData
                        });
                    }
                    
                    // API 호출 간격 (타임아웃 방지를 위해 줄임)
                    await new Promise(resolve => setTimeout(resolve, 500));
                } catch (error) {
                    console.error(`${query.year}년 재무정보 조회 오류:`, error);
                }
            }

            if (allData.length > 0) {
                const prediction = await this.predictRevenueWithML(allData);
                return {
                    quarterlyData: allData,
                    prediction,
                    companyName: corpName
                };
            } else {
                return { error: '재무정보를 찾을 수 없습니다.' };
            }
        } catch (error) {
            console.error('DART 재무정보 조회 오류:', error);
            return { error: '재무정보 조회 중 오류가 발생했습니다.' };
        }
    }

    async predictRevenueWithML(financialData) {
        try {
            // Python 스크립트에 전달할 데이터 준비
            const inputData = JSON.stringify(financialData);
            const pythonScriptPath = path.join(__dirname, '..', 'services', 'revenuePredictionModel.py');
            
            return new Promise((resolve) => {
                const pythonProcess = spawn('python3', [pythonScriptPath], {
                    stdio: ['pipe', 'pipe', 'pipe']
                });
                
                let output = '';
                let errorOutput = '';
                
                pythonProcess.stdout.on('data', (data) => {
                    output += data.toString();
                });
                
                pythonProcess.stderr.on('data', (data) => {
                    errorOutput += data.toString();
                });
                
                pythonProcess.on('close', (code) => {
                    if (code === 0 && output.trim()) {
                        try {
                            const result = JSON.parse(output.trim());
                            
                            // 성장률 계산
                            const growthRate = this.calculateQuarterlyGrowthRate(financialData);
                            
                            // 결과를 기존 형식에 맞게 변환
                            const formattedResult = {
                                nextYear: 2025,
                                nextQuarter: 'Q2',
                                predictedRevenue: result.predictedRevenue,
                                growthRate: growthRate,
                                modelAccuracy: result.accuracy,
                                analysis: `${result.model} 모델 기반 예측 (신뢰도: ${result.confidence})`,
                                model: result.model,
                                confidence: result.confidence,
                                features_used: result.features_used,
                                data_points: result.data_points,
                                top_features: result.top_features,
                                mae: result.mae
                            };
                            
                            resolve(formattedResult);
                        } catch (parseError) {
                            console.error('Python 출력 파싱 오류:', parseError);
                            resolve(this.getFallbackPrediction(financialData));
                        }
                    } else {
                        console.error('Python 스크립트 실행 오류:', errorOutput);
                        resolve(this.getFallbackPrediction(financialData));
                    }
                });
                
                pythonProcess.on('error', (error) => {
                    console.error('Python 프로세스 오류:', error);
                    resolve(this.getFallbackPrediction(financialData));
                });
                
                // 입력 데이터 전송
                pythonProcess.stdin.write(inputData);
                pythonProcess.stdin.end();
                
                // 타임아웃 설정 (30초)
                setTimeout(() => {
                    pythonProcess.kill();
                    resolve(this.getFallbackPrediction(financialData));
                }, 30000);
            });
        } catch (error) {
            console.error('ML 모델 예측 오류:', error);
            return this.getFallbackPrediction(financialData);
        }
    }

    getFallbackPrediction(financialData) {
        const growthRate = this.calculateQuarterlyGrowthRate(financialData);
        return {
            nextYear: 2025,
            nextQuarter: 'Q2',
            predictedRevenue: null,
            growthRate: growthRate,
            modelAccuracy: 0,
            analysis: '고급 ML 모델 사용 불가로 기본 예측 사용',
            error: 'ML 모델 실행 실패'
        };
    }

    calculateQuarterlyGrowthRate(financialData) {
        try {
            if (!financialData || financialData.length < 2) {
                return null;
            }

            // 최신 2개 분기 데이터 가져오기 (가장 최근이 첫 번째)
            const latestQuarter = financialData[0];
            const previousQuarter = financialData[1];

            if (!latestQuarter || !previousQuarter || !latestQuarter.revenue || !previousQuarter.revenue) {
                return null;
            }

            const latestRevenue = parseFloat(latestQuarter.revenue.toString().replace(/,/g, ''));
            const previousRevenue = parseFloat(previousQuarter.revenue.toString().replace(/,/g, ''));

            if (isNaN(latestRevenue) || isNaN(previousRevenue) || previousRevenue === 0) {
                return null;
            }

            const growthRate = ((latestRevenue - previousRevenue) / previousRevenue) * 100;
            return Math.round(growthRate * 100) / 100; // 소수점 둘째 자리까지
        } catch (error) {
            console.error('성장률 계산 오류:', error);
            return null;
        }
    }

    parseFinancialData(data) {
        const financialInfo = {
            revenue: null,
            operatingProfit: null,
            netIncome: null,
            totalAssets: null,
            totalEquity: null,
            currentAssets: null,
            currentLiabilities: null
        };

        const targetAccounts = [
            { keys: ['매출액', '수익(매출액)'], field: 'revenue' },
            { keys: ['영업이익'], field: 'operatingProfit' },
            { keys: ['당기순이익'], field: 'netIncome' },
            { keys: ['자산총계'], field: 'totalAssets' },
            { keys: ['자본총계'], field: 'totalEquity' },
            { keys: ['유동자산'], field: 'currentAssets' },
            { keys: ['유동부채'], field: 'currentLiabilities' }
        ];

        if (data && data.length > 0) {
            data.forEach(item => {
                targetAccounts.forEach(target => {
                    if (target.keys.includes(item.account_nm)) {
                        financialInfo[target.field] = item.thstrm_amount;
                    }
                });
            });
        }

        return financialInfo;
    }

    async searchCompanyByName(companyName) {
        try {
            // 특정 회사명에 대한 매핑 우선 처리
            const companyMappings = {
                '현대차': '005380',
                '현대자동차': '005380',
                '삼성전자': '005930',
                'SK하이닉스': '000660',
                'LG에너지솔루션': '373220',
                'NAVER': '035420',
                '네이버': '035420',
                'KB금융': '105560',
                '신한지주': '055550'
            };
            
            if (companyMappings[companyName]) {
                const stockCode = companyMappings[companyName];
                const corpDict = await this.getCorpCodeDict();
                const info = corpDict[stockCode];
                if (info) {
                    return [{
                        stockCode,
                        corpCode: info.corpCode,
                        corpName: info.corpName
                    }];
                }
            }
            
            const corpDict = await this.getCorpCodeDict();
            const results = [];
            const exactMatches = [];
            const partialMatches = [];
            
            Object.entries(corpDict).forEach(([stockCode, info]) => {
                const corpName = info.corpName;
                
                // 정확한 매치 우선
                if (corpName === companyName || corpName === companyName + '(주)' || corpName === `${companyName}주식회사`) {
                    exactMatches.push({
                        stockCode,
                        corpCode: info.corpCode,
                        corpName: info.corpName
                    });
                }
                // 부분 매치
                else if (corpName.includes(companyName)) {
                    partialMatches.push({
                        stockCode,
                        corpCode: info.corpCode,
                        corpName: info.corpName
                    });
                }
            });
            
            // 정확한 매치가 있으면 그것만 반환, 없으면 부분 매치 반환
            return exactMatches.length > 0 ? exactMatches : partialMatches.slice(0, 5);
            
        } catch (error) {
            console.error('DART 기업검색 오류:', error);
            throw error;
        }
    }

    getQuarterlyReportCode(quarter) {
        const codes = {
            1: '11013', // 1분기보고서
            2: '11012', // 반기보고서  
            3: '11014', // 3분기보고서
            4: '11011'  // 사업보고서(연간)
        };
        return codes[quarter] || '11011';
    }

    async getStockCodeByName(companyName) {
        if (!this.apiKey) {
            console.error('DART API 키가 설정되지 않았습니다.');
            return null;
        }

        try {
            const corpDict = await this.getCorpCodeDict();
            
            // 정확한 이름 일치 먼저 찾기
            for (const [stockCode, info] of Object.entries(corpDict)) {
                if (info.corpName === companyName) {
                    return stockCode;
                }
            }
            
            // 부분 일치 (포함 관계)
            for (const [stockCode, info] of Object.entries(corpDict)) {
                if (info.corpName.includes(companyName) || companyName.includes(info.corpName)) {
                    return stockCode;
                }
            }
            
            return null;
        } catch (error) {
            console.error('종목코드 검색 오류:', error);
            return null;
        }
    }

    async getBasicInfo(stockCode) {
        if (!this.apiKey) {
            return { error: 'DART API 키가 설정되지 않았습니다.' };
        }

        try {
            const corpDict = await this.getCorpCodeDict();
            const corpInfo = corpDict[stockCode];
            
            if (!corpInfo) {
                return { error: `종목코드 '${stockCode}'에 해당하는 기업을 찾을 수 없습니다.` };
            }

            const companyInfo = await this.getCompanyInfo(corpInfo.corpCode, stockCode);
            
            if (companyInfo) {
                return companyInfo;
            } else {
                // 기본 정보만 반환
                return {
                    회사명: corpInfo.corpName,
                    영문명: '정보없음',
                    대표자: '정보없음',
                    설립일: '정보없음',
                    본사주소: '정보없음',
                    홈페이지: '정보없음',
                    업종: '정보없음',
                    결산월: '정보없음'
                };
            }
        } catch (error) {
            console.error('DART 기업정보 조회 오류:', error);
            return { error: '기업정보 조회 중 오류가 발생했습니다.' };
        }
    }

    formatAmount(amount) {
        if (!amount) return '정보 없음';
        
        const num = parseInt(amount.replace(/,/g, ''));
        if (isNaN(num)) return amount;
        
        if (num >= 1000000000000) {
            return `${(num / 1000000000000).toFixed(1)}조원`;
        } else if (num >= 100000000) {
            return `${(num / 100000000).toFixed(1)}억원`;
        } else {
            return `${num.toLocaleString()}원`;
        }
    }
}

module.exports = new DartService();