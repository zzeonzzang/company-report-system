document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('reportForm');
    const loading = document.getElementById('loading');
    const reportContainer = document.getElementById('reportContainer');
    const testBtn = document.getElementById('testBtn');

    form.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const companyName = document.getElementById('companyName').value.trim();
        const stockCode = document.getElementById('stockCode').value.trim();

        if (!companyName) {
            alert('기업명을 입력해주세요.');
            return;
        }

        showLoading();
        
        try {
            // 타임아웃 추가 (180초)
            const controller = new AbortController();
            const timeoutId = setTimeout(() => controller.abort(), 180000);
            
            const response = await fetch('/api/company-report', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    companyName: companyName,
                    stockCode: stockCode
                }),
                signal: controller.signal
            });

            clearTimeout(timeoutId);
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }

            const data = await response.json();
            displayReport(data);
            
        } catch (error) {
            console.error('Error:', error);
            
            if (error.name === 'AbortError') {
                showError('요청 시간이 초과되었습니다. 다시 시도해주세요.');
            } else if (error.message.includes('HTTP')) {
                showError(`서버 오류: ${error.message}`);
            } else if (error.message.includes('Failed to fetch')) {
                showError('서버에 연결할 수 없습니다. 서버가 실행 중인지 확인해주세요.');
            } else {
                showError(`오류가 발생했습니다: ${error.message}`);
            }
        } finally {
            hideLoading();
        }
    });

    // 테스트 버튼 이벤트
    testBtn.addEventListener('click', async function() {
        try {
            console.log('테스트 시작...');
            
            // 기본 연결 테스트
            const testResponse = await fetch('/api/test');
            const testData = await testResponse.json();
            console.log('기본 연결 테스트:', testData);
            
            // 회사 정보 테스트
            const companyResponse = await fetch('/api/test-company', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ companyName: '테스트회사' })
            });
            const companyData = await companyResponse.json();
            console.log('회사 정보 테스트:', companyData);
            
            alert('✅ 연결 테스트 성공!\n콘솔(F12)에서 상세 결과를 확인하세요.');
            
        } catch (error) {
            console.error('테스트 실패:', error);
            alert('❌ 테스트 실패: ' + error.message);
        }
    });

    function showLoading() {
        loading.style.display = 'block';
        reportContainer.style.display = 'none';
        form.querySelector('button').disabled = true;
    }

    function hideLoading() {
        loading.style.display = 'none';
        form.querySelector('button').disabled = false;
    }

    function showError(message) {
        const errorDiv = document.createElement('div');
        errorDiv.className = 'error';
        errorDiv.textContent = message;
        
        const container = document.querySelector('.container');
        container.insertBefore(errorDiv, reportContainer);
        
        setTimeout(() => {
            errorDiv.remove();
        }, 5000);
    }

    function displayReport(data) {
        displayBasicInfo(data.basicInfo, data.companyName);
        displayJobInfo(data.jobInfo);
        displayFinancialInfo(data.financialInfo);
        displayStockInfo(data.stockInfo);
        displayNewsInfo(data.newsInfo);
        
        reportContainer.style.display = 'block';
    }

    function displayBasicInfo(basicInfo, companyName) {
        const container = document.getElementById('basicInfo');
        const koreanName = companyName || '정보없음';
        const englishName = basicInfo?.영문명 || '';
        const combinedName = englishName && englishName !== '정보없음' ? 
            `${koreanName} (${englishName})` : koreanName;
            
        container.innerHTML = `
            <div class="info-grid">
                <div class="info-item">
                    <div class="info-label">기업명</div>
                    <div class="info-value">${combinedName}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">대표자</div>
                    <div class="info-value">${basicInfo?.대표자 || '정보없음'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">설립일</div>
                    <div class="info-value">${basicInfo?.설립일 || '정보없음'}</div>
                </div>
                <div class="info-item">
                    <div class="info-label">본사주소</div>
                    <div class="info-value">${basicInfo?.본사주소 || '정보없음'}</div>
                </div>
            </div>
        `;
    }

    function displayFinancialInfo(financialInfo) {
        const container = document.getElementById('financialInfo');
        
        if (!financialInfo) {
            container.innerHTML = '<p>종목코드가 제공되지 않아 재무정보를 조회할 수 없습니다.</p>';
            return;
        }

        if (financialInfo.error) {
            container.innerHTML = `<p class="error">${financialInfo.error}</p>`;
            return;
        }

        let html = '';

        // 분기별 재무정보 표
        if (financialInfo.quarterlyData && financialInfo.quarterlyData.length > 0) {
            html += `
                <h3 style="margin-bottom: 1rem; color: #333;">분기별 재무정보 (최근 4분기)</h3>
                <div style="overflow-x: auto; margin-bottom: 2rem;">
                    <table style="width: 100%; border-collapse: collapse; margin: 1rem 0; background-color: white; border-radius: 8px; overflow: hidden; font-size: 0.9rem;">
                        <thead>
                            <tr style="background-color: #f0f4f8;">
                                <th style="padding: 12px 15px; text-align: left; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #e0e0e0;">분기</th>
                                <th style="padding: 12px 15px; text-align: right; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #e0e0e0;">매출액</th>
                                <th style="padding: 12px 15px; text-align: right; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #e0e0e0;">영업이익</th>
                                <th style="padding: 12px 15px; text-align: right; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #e0e0e0;">당기순이익</th>
                                <th style="padding: 12px 15px; text-align: right; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #e0e0e0;">총자산</th>
                                <th style="padding: 12px 15px; text-align: right; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #e0e0e0;">총자본</th>
                            </tr>
                        </thead>
                        <tbody>`;
            
            financialInfo.quarterlyData.forEach((quarterData, index) => {
                html += `
                    <tr style="border-bottom: 1px solid #e0e0e0; ${index % 2 === 1 ? 'background-color: #f8fafc;' : ''}">
                        <td style="padding: 10px 15px; color: #333; font-weight: 600;">${quarterData.year}년 ${quarterData.quarter}</td>
                        <td style="padding: 10px 15px; color: #333; text-align: right;">${formatAmount(quarterData.revenue)}</td>
                        <td style="padding: 10px 15px; color: #333; text-align: right;">${formatAmount(quarterData.operatingProfit)}</td>
                        <td style="padding: 10px 15px; color: #333; text-align: right;">${formatAmount(quarterData.netIncome)}</td>
                        <td style="padding: 10px 15px; color: #333; text-align: right;">${formatAmount(quarterData.totalAssets)}</td>
                        <td style="padding: 10px 15px; color: #333; text-align: right;">${formatAmount(quarterData.totalEquity)}</td>
                    </tr>`;
            });
            
            html += `
                        </tbody>
                    </table>
                </div>`;
        }

        // 매출액 예측 섹션
        if (financialInfo.prediction && !financialInfo.prediction.error) {
            const prediction = financialInfo.prediction;
            const changeSymbol = prediction.growthRate > 0 ? '▲' : prediction.growthRate < 0 ? '▼' : '';
            const changeColor = prediction.growthRate > 0 ? '#ff6b6b' : prediction.growthRate < 0 ? '#4dabf7' : '#666';
            
            html += `
                <div style="margin-top: 2rem; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
                    <h3 style="margin-bottom: 1rem; color: #333;">📈 분기별 매출액 예측</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <div class="info-label">${prediction.nextYear}년 ${prediction.nextQuarter} 예상 매출액</div>
                            <div class="info-value" style="font-size: 1.3rem; font-weight: bold; color: #0064ff;">
                                ${formatAmount(prediction.predictedRevenue?.toString())}
                            </div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">전분기 대비 성장률</div>
                            <div class="info-value" style="color: ${changeColor}; font-weight: bold;">
                                ${changeSymbol} ${Math.abs(prediction.growthRate || 0).toFixed(1)}%
                            </div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">예측 정확도</div>
                            <div class="info-value">${Math.round(prediction.modelAccuracy || 0)}%</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">예측 모델</div>
                            <div class="info-value">분기별 성장률 기반</div>
                        </div>
                    </div>
                    
                    <div style="margin-top: 1rem; padding: 1rem; background: white; border-radius: 8px; border-left: 4px solid #0064ff;">
                        <div style="font-size: 0.9rem; color: #666;">
                            <strong>분석:</strong> 
                            ${prediction.growthRate > 5 ? '높은 성장률을 보이고 있어 긍정적인 전망입니다.' : 
                              prediction.growthRate < -5 ? '매출 감소 추세로 주의가 필요합니다.' : 
                              '안정적인 성장세를 유지하고 있습니다.'}
                        </div>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;
    }

    function displayStockInfo(stockInfo) {
        const container = document.getElementById('stockInfo');
        
        if (!stockInfo) {
            container.innerHTML = '<p>종목코드가 제공되지 않아 주가정보를 조회할 수 없습니다.</p>';
            return;
        }

        if (stockInfo.error) {
            container.innerHTML = `<p class="error">${stockInfo.error}</p>`;
            return;
        }

        const changeColor = stockInfo.changeRate > 0 ? '#ff6b6b' : stockInfo.changeRate < 0 ? '#4dabf7' : '#666';
        const changeSymbol = stockInfo.changeRate > 0 ? '▲' : stockInfo.changeRate < 0 ? '▼' : '';

        let technicalSection = '';
        if (stockInfo.technicalIndicators && stockInfo.technicalIndicators.rsi !== null) {
            const indicators = stockInfo.technicalIndicators;
            const signal = getTechnicalSignal(indicators);
            const signalColor = signal === '매수' ? '#51cf66' : signal === '매도' ? '#ff6b6b' : '#ffd43b';
            
            technicalSection = `
                <div style="margin-top: 2rem; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
                    <h3 style="margin-bottom: 1rem; color: #333;">💹 기술적 분석</h3>
                    
                    <!-- 기술적 지표 카드 -->
                    <div style="display: flex; justify-content: space-between; gap: 15px; margin-bottom: 1.5rem;">
                        <div style="flex: 1; padding: 15px; background: white; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                            <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">20일 이평선</div>
                            <div style="font-size: 1.2rem; color: #0064ff; font-weight: 600;">${indicators.ma20 ? indicators.ma20.toLocaleString() + '원' : '계산불가'}</div>
                            <div style="font-size: 0.8rem; color: #999; margin-top: 4px;">
                                ${stockInfo.currentPrice && indicators.ma20 ? 
                                    (stockInfo.currentPrice > indicators.ma20 ? '현재가 > 이평선' : '현재가 < 이평선') : ''}
                            </div>
                        </div>
                        <div style="flex: 1; padding: 15px; background: white; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                            <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">RSI</div>
                            <div style="font-size: 1.2rem; color: #0064ff; font-weight: 600;">${indicators.rsi || '계산불가'}</div>
                            <div style="font-size: 0.8rem; margin-top: 4px; color: ${indicators.rsi > 70 ? '#ff6b6b' : indicators.rsi < 30 ? '#51cf66' : '#ffd43b'};">
                                ${indicators.rsi > 70 ? '과매수' : indicators.rsi < 30 ? '과매도' : '중립'}
                            </div>
                        </div>
                        <div style="flex: 1; padding: 15px; background: white; border-radius: 8px; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.05);">
                            <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">볼린저밴드</div>
                            <div style="font-size: 1.2rem; color: #0064ff; font-weight: 600;">
                                ${indicators.bollinger ? 
                                    (stockInfo.currentPrice > indicators.bollinger.upper ? '상단돌파' : 
                                     stockInfo.currentPrice < indicators.bollinger.lower ? '하단이탈' : '중립') : '계산불가'}
                            </div>
                            <div style="font-size: 0.8rem; color: #999; margin-top: 4px;">
                                ${indicators.bollinger ? `${indicators.bollinger.lower.toLocaleString()} ~ ${indicators.bollinger.upper.toLocaleString()}` : ''}
                            </div>
                        </div>
                    </div>

                    <!-- 종합 투자신호 -->
                    <div style="padding: 15px; background: white; border-radius: 8px; text-align: center; margin-bottom: 1.5rem;">
                        <div style="font-size: 0.9rem; color: #666; margin-bottom: 8px;">종합 투자신호</div>
                        <div style="font-size: 1.4rem; color: ${signalColor}; font-weight: bold;">${signal}</div>
                    </div>
                </div>
            `;
        }

        // 주가 예측 및 시각화 섹션
        let predictionSection = '';
        if (stockInfo.predictions && stockInfo.predictions.predictions && stockInfo.predictions.predictions.length > 0) {
            const pred = stockInfo.predictions;
            predictionSection = `
                <div style="margin-top: 2rem; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
                    <h3 style="margin-bottom: 1rem; color: #333;">📊 RandomForest 주가 예측 분석</h3>
                    
                    <div class="info-grid" style="margin-bottom: 1.5rem;">
                        <div class="info-item">
                            <div class="info-label">예측 모델</div>
                            <div class="info-value">${pred.modelInfo || 'Random Forest (LightGBM 대안)'}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">예측 정확도</div>
                            <div class="info-value">${Math.round(pred.accuracy || 0)}%</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">RMSE</div>
                            <div class="info-value">${pred.rmse ? pred.rmse.toLocaleString() + '원' : '계산불가'}</div>
                        </div>
                        <div class="info-item">
                            <div class="info-label">예측 데이터</div>
                            <div class="info-value">${pred.predictions.length}일</div>
                        </div>
                    </div>
                    
                    <!-- 시계열 차트 -->
                    <div style="background: white; padding: 15px; border-radius: 8px; margin-bottom: 1.5rem;">
                        <h4 style="margin-bottom: 15px; color: #333;">📈 시계열 예측 차트</h4>
                        <div id="timeSeriesChart" style="width: 100%; height: 400px;"></div>
                    </div>
                    
                    <!-- 최근 예측 결과 테이블 -->
                    <div style="background: white; padding: 15px; border-radius: 8px;">
                        <h4 style="margin-bottom: 10px; color: #333;">최근 예측 vs 실제</h4>
                        <div style="overflow-x: auto;">
                            <table style="width: 100%; border-collapse: collapse; font-size: 0.85rem;">
                                <thead>
                                    <tr style="background-color: #f8f9fa;">
                                        <th style="padding: 8px; text-align: left; border-bottom: 1px solid #e0e0e0;">날짜</th>
                                        <th style="padding: 8px; text-align: right; border-bottom: 1px solid #e0e0e0;">실제가</th>
                                        <th style="padding: 8px; text-align: right; border-bottom: 1px solid #e0e0e0;">예측가</th>
                                        <th style="padding: 8px; text-align: right; border-bottom: 1px solid #e0e0e0;">오차</th>
                                        <th style="padding: 8px; text-align: center; border-bottom: 1px solid #e0e0e0;">정확도</th>
                                    </tr>
                                </thead>
                                <tbody>
                                    ${pred.predictions.slice(-8).map(p => {
                                        if (!p.actual || !p.predicted) {
                                            return `
                                            <tr style="border-bottom: 1px solid #f0f0f0;">
                                                <td style="padding: 8px; color: #333;">${p.date}</td>
                                                <td style="padding: 8px; text-align: right; color: #999;">미래예측</td>
                                                <td style="padding: 8px; text-align: right; color: #0064ff;">${p.predicted ? p.predicted.toLocaleString() + '원' : '-'}</td>
                                                <td style="padding: 8px; text-align: right; color: #999;">-</td>
                                                <td style="padding: 8px; text-align: center; color: #999;">예측값</td>
                                            </tr>`;
                                        }
                                        
                                        const error = Math.abs(p.actual - p.predicted);
                                        const errorRate = (error / p.actual) * 100;
                                        const isAccurate = errorRate < 5;
                                        return `
                                        <tr style="border-bottom: 1px solid #f0f0f0;">
                                            <td style="padding: 8px; color: #333;">${p.date}</td>
                                            <td style="padding: 8px; text-align: right; color: #333;">${p.actual.toLocaleString()}원</td>
                                            <td style="padding: 8px; text-align: right; color: #0064ff;">${p.predicted.toLocaleString()}원</td>
                                            <td style="padding: 8px; text-align: right; color: ${isAccurate ? '#51cf66' : '#ff6b6b'};">${error.toLocaleString()}원</td>
                                            <td style="padding: 8px; text-align: center;">
                                                <span style="color: ${isAccurate ? '#51cf66' : '#ff6b6b'};">
                                                    ${isAccurate ? '✓' : '✗'} ${(100 - errorRate).toFixed(1)}%
                                                </span>
                                            </td>
                                        </tr>`;
                                    }).join('')}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            `;
        }

        // GPT 분석 섹션
        let gptSection = '';
        if (stockInfo.gptAnalysis) {
            gptSection = `
                <div style="margin-top: 2rem; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
                    <h3 style="margin-bottom: 1rem; color: #333;">🤖 AI 투자 분석</h3>
                    <div style="background: white; padding: 15px; border-radius: 8px; line-height: 1.6; color: #333;">
                        ${stockInfo.gptAnalysis.split('\n').map(line => line.trim()).filter(line => line).map(line => `<p style="margin-bottom: 10px;">${line}</p>`).join('')}
                    </div>
                </div>
            `;
        }

        container.innerHTML = `
            ${technicalSection}
            ${predictionSection}
            ${gptSection}
            ${stockInfo.note ? `<p style="margin-top: 1rem; color: #666; font-style: italic;">${stockInfo.note}</p>` : ''}
        `;
        
        // 시계열 차트 그리기
        if (stockInfo.predictions && stockInfo.predictions.chartData) {
            setTimeout(() => {
                drawTimeSeriesChart(stockInfo.predictions.chartData);
            }, 100);
        }
    }

    function getTechnicalSignal(indicators) {
        if (!indicators || !indicators.rsi) return '분석불가';

        const { rsi, ma5, ma20 } = indicators;
        let signals = [];

        if (rsi > 70) signals.push('과매수');
        else if (rsi < 30) signals.push('과매도');
        else signals.push('중립');

        if (ma5 && ma20) {
            if (ma5 > ma20) signals.push('상승추세');
            else signals.push('하락추세');
        }

        const buySignals = signals.filter(s => ['과매도', '상승추세'].includes(s)).length;
        const sellSignals = signals.filter(s => ['과매수', '하락추세'].includes(s)).length;

        if (buySignals > sellSignals) return '매수';
        if (sellSignals > buySignals) return '매도';
        return '관망';
    }

    function formatVolume(volume) {
        if (!volume || volume === 0 || volume === null || volume === undefined) return '정보 없음';
        
        const numVolume = Number(volume);
        if (isNaN(numVolume)) return '정보 없음';
        
        if (numVolume >= 10000) {
            return `${Math.round(numVolume / 10000).toLocaleString()}만주`;
        }
        return `${numVolume.toLocaleString()}주`;
    }

    function displayNewsInfo(newsInfo) {
        const container = document.getElementById('newsInfo');
        
        if (!newsInfo || newsInfo.error) {
            container.innerHTML = `<p class="error">${newsInfo ? newsInfo.error : '뉴스 정보를 조회할 수 없습니다.'}</p>`;
            return;
        }

        let html = '';

        // 감정분석 요약 표시
        if (newsInfo.sentimentSummary && newsInfo.total > 0) {
            const { positive, neutral, negative, positiveRatio, neutralRatio, negativeRatio, total } = newsInfo.sentimentSummary;
            html += `
                <div style="margin-bottom: 1.5rem; padding: 1.5rem; background: #f8f9fa; border-radius: 10px;">
                    <h3 style="margin-bottom: 1rem; color: #333;">뉴스 감정분석 요약</h3>
                    <div style="display: flex; gap: 1rem; margin-bottom: 1rem;">
                        <div style="flex: 1; text-align: center; padding: 1rem; background: #e8f5e8; border-radius: 8px;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #51cf66;">${positive}건</div>
                            <div style="color: #666;">긍정 (${positiveRatio}%)</div>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 1rem; background: #fff3cd; border-radius: 8px;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #ffd43b;">${neutral}건</div>
                            <div style="color: #666;">중립 (${neutralRatio}%)</div>
                        </div>
                        <div style="flex: 1; text-align: center; padding: 1rem; background: #f8d7da; border-radius: 8px;">
                            <div style="font-size: 1.5rem; font-weight: bold; color: #ff6b6b;">${negative}건</div>
                            <div style="color: #666;">부정 (${negativeRatio}%)</div>
                        </div>
                    </div>
                    <div style="text-align: center; color: #666;">
                        총 ${total}건의 뉴스 분석 완료
                    </div>
                </div>
            `;
        }

        // 뉴스 요약
        html += `<div style="margin-bottom: 1.5rem; padding: 1rem; background: #e3f2fd; border-radius: 8px;">
            <strong>📰 뉴스 요약:</strong> ${newsInfo.summary}
        </div>`;

        // 뉴스 테이블
        if (newsInfo.articles && newsInfo.articles.length > 0) {
            html += `
                <h3 style="margin-bottom: 1rem; color: #333;">📰 최근 뉴스 (감정분석)</h3>
                <div style="overflow-x: auto;">
                    <table style="width: 100%; border-collapse: collapse; margin: 1rem 0; background-color: white; border-radius: 8px; overflow: hidden; font-size: 0.9rem;">
                        <thead>
                            <tr style="background-color: #f0f4f8;">
                                <th style="padding: 12px 15px; text-align: left; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #e0e0e0; width: 25%;">제목</th>
                                <th style="padding: 12px 15px; text-align: center; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #e0e0e0; width: 12%; white-space: nowrap;">감정</th>
                                <th style="padding: 12px 15px; text-align: left; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #e0e0e0; width: 50%;">분석 이유</th>
                                <th style="padding: 12px 15px; text-align: center; font-weight: 600; color: #1a1a1a; border-bottom: 1px solid #e0e0e0; width: 13%; white-space: nowrap;">날짜</th>
                            </tr>
                        </thead>
                        <tbody>`;
            
            newsInfo.articles.slice(0, 10).forEach((article, index) => {
                const sentimentColor = article.sentiment === '긍정' ? '#51cf66' : 
                                     article.sentiment === '부정' ? '#ff6b6b' : '#ffd43b';
                const sentimentIcon = article.sentiment === '긍정' ? '😊' : 
                                     article.sentiment === '부정' ? '😟' : '😐';
                
                html += `
                    <tr style="border-bottom: 1px solid #e0e0e0; ${index % 2 === 1 ? 'background-color: #f8fafc;' : ''}" onmouseover="this.style.backgroundColor='#f0f4f8'" onmouseout="this.style.backgroundColor='${index % 2 === 1 ? '#f8fafc' : 'white'}'">
                        <td style="padding: 10px 15px; color: #333; line-height: 1.4;">
                            <a href="${article.link}" target="_blank" style="color: #0064ff; text-decoration: none; font-weight: 500;">${article.title}</a>
                        </td>
                        <td style="padding: 10px 15px; text-align: center;">
                            <span style="padding: 0.3rem 0.6rem; background: ${sentimentColor}; color: white; border-radius: 12px; font-size: 0.8rem; font-weight: 500;">
                                ${sentimentIcon} ${article.sentiment}
                            </span>
                        </td>
                        <td style="padding: 10px 15px; color: #666; line-height: 1.4; font-size: 0.85rem;">
                            ${article.reason || '분석 정보 없음'}
                        </td>
                        <td style="padding: 10px 15px; color: #999; text-align: center; font-size: 0.85rem; white-space: nowrap;">
                            ${formatDate(article.pubDate)}
                        </td>
                    </tr>`;
            });
            
            html += `
                        </tbody>
                    </table>
                </div>`;
        } else {
            html += '<p>관련 뉴스를 찾을 수 없습니다.</p>';
        }

        container.innerHTML = html;
    }

    function drawTimeSeriesChart(chartData) {
        if (!chartData || !document.getElementById('timeSeriesChart')) return;
        
        const traces = [];
        
        // 실제 주가 (과거 데이터만)
        const actualDates = [];
        const actualValues = [];
        chartData.dates.forEach((date, i) => {
            if (chartData.actual[i] !== null) {
                actualDates.push(date);
                actualValues.push(chartData.actual[i]);
            }
        });
        
        if (actualValues.length > 0) {
            traces.push({
                x: actualDates,
                y: actualValues,
                type: 'scatter',
                mode: 'lines',
                name: '실제 주가',
                line: { color: '#0064ff', width: 2 }
            });
        }
        
        // 과거 예측 주가 (검증용)
        const pastPredictedDates = [];
        const pastPredictedValues = [];
        chartData.dates.forEach((date, i) => {
            if (chartData.predicted[i] !== null && chartData.actual[i] !== null) {
                pastPredictedDates.push(date);
                pastPredictedValues.push(chartData.predicted[i]);
            }
        });
        
        if (pastPredictedValues.length > 0) {
            traces.push({
                x: pastPredictedDates,
                y: pastPredictedValues,
                type: 'scatter',
                mode: 'lines',
                name: '예측값',
                line: { color: '#ff9f43', width: 1.5, dash: 'dot' }
            });
        }
        
        
        // 5일 이동평균선 (과거 데이터만)
        const ma5Dates = [];
        const ma5Values = [];
        chartData.dates.forEach((date, i) => {
            if (chartData.ma5[i] !== null) {
                ma5Dates.push(date);
                ma5Values.push(chartData.ma5[i]);
            }
        });
        
        if (ma5Values.length > 0) {
            traces.push({
                x: ma5Dates,
                y: ma5Values,
                type: 'scatter',
                mode: 'lines',
                name: '5일 이평선',
                line: { color: '#ffd43b', width: 1 },
                opacity: 0.7
            });
        }
        
        // 20일 이동평균선 (과거 데이터만)
        const ma20Dates = [];
        const ma20Values = [];
        chartData.dates.forEach((date, i) => {
            if (chartData.ma20[i] !== null) {
                ma20Dates.push(date);
                ma20Values.push(chartData.ma20[i]);
            }
        });
        
        if (ma20Values.length > 0) {
            traces.push({
                x: ma20Dates,
                y: ma20Values,
                type: 'scatter',
                mode: 'lines',
                name: '20일 이평선',
                line: { color: '#51cf66', width: 1 },
                opacity: 0.7
            });
        }
        
        // 최근 6개월만 표시하도록 날짜 범위 설정
        const today = new Date();
        const sixMonthsAgo = new Date(today);
        sixMonthsAgo.setMonth(today.getMonth() - 6);
        
        const layout = {
            title: {
                text: '주가예측 시계열차트',
                font: { size: 16, color: '#333' }
            },
            xaxis: {
                title: '날짜',
                gridcolor: '#e0e0e0',
                tickformat: '%m/%d',
                dtick: 'M1'
            },
            yaxis: {
                title: '주가 (원)',
                gridcolor: '#e0e0e0'
            },
            plot_bgcolor: '#fafafa',
            paper_bgcolor: 'white',
            legend: {
                x: 0,
                y: 1,
                bgcolor: 'rgba(255,255,255,0.9)',
                bordercolor: '#ddd',
                borderwidth: 1
            },
            margin: { l: 70, r: 30, t: 60, b: 70 }
        };
        
        const config = {
            responsive: true,
            displayModeBar: true,
            modeBarButtonsToRemove: ['pan2d', 'lasso2d', 'select2d'],
            displaylogo: false
        };
        
        Plotly.newPlot('timeSeriesChart', traces, layout, config);
    }

    function formatAmount(amount) {
        if (!amount || amount === '정보 조회 중...' || amount === null || amount === undefined) {
            return '정보 없음';
        }
        
        // 문자열이 아닌 경우 문자열로 변환
        const amountStr = String(amount);
        const num = parseInt(amountStr.replace(/,/g, ''));
        
        if (isNaN(num) || num === 0) {
            return amountStr === '0' ? '0원' : '정보 없음';
        }
        
        if (num >= 1000000000000) {
            return `${(num / 1000000000000).toFixed(1)}조원`;
        } else if (num >= 100000000) {
            return `${(num / 100000000).toFixed(1)}억원`;
        } else {
            return `${num.toLocaleString()}원`;
        }
    }

    function formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('ko-KR', {
            year: 'numeric',
            month: 'long',
            day: 'numeric'
        });
    }

    function displayJobInfo(jobInfo) {
        const container = document.getElementById('jobInfo');
        
        if (!jobInfo || jobInfo.error) {
            container.innerHTML = `
                <div class="error">채용정보를 가져올 수 없습니다.</div>
            `;
            return;
        }

        // 단일 라인 형태로 임직원 수, 신입 채용, 경력 채용 정보만 표시
        const html = `
            <div style="padding: 1.5rem; background: #f8f9fa; border-radius: 8px; text-align: center;">
                <div style="font-size: 1.2rem; color: #333; font-weight: 600;">
                    👥 임직원 ${jobInfo.employeeCount || '정보없음'} | 
                    🎓 ${jobInfo.newGradJobs || '신입 0건'} | 
                    💼 ${jobInfo.experiencedJobs || '경력 0건'}
                </div>
            </div>
        `;

        container.innerHTML = html;
    }

    function formatEmployeeCount(count) {
        if (!count) return '정보없음';
        if (count >= 10000) {
            return `${Math.floor(count / 10000)}만 ${(count % 10000).toLocaleString()}명`;
        }
        return `${count.toLocaleString()}명`;
    }

    function formatMarketCap(marketCap) {
        if (!marketCap) return '정보없음';
        if (marketCap >= 10000) {
            return `${Math.floor(marketCap / 10000)}조원`;
        }
        return `${marketCap.toLocaleString()}억원`;
    }
});