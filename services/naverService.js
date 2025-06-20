const axios = require('axios');
const cheerio = require('cheerio');

class NaverService {
    constructor() {
        this.clientId = process.env.NAVER_API_CLIENT_ID;
        this.clientSecret = process.env.NAVER_API_CLIENT_SECRET;
        this.openaiApiKey = process.env.OPENAI_API_KEY;
    }

    async getNews(companyName, maxResults = 10) {
        if (!this.clientId || !this.clientSecret) {
            return { 
                articles: [], 
                summary: 'Naver API 키가 설정되지 않았습니다.',
                error: 'API 키 필요',
                sentimentSummary: { positive: 0, neutral: 0, negative: 0, total: 0 },
                total: 0
            };
        }

        try {
            const searchUrl = `https://openapi.naver.com/v1/search/news.json`;
            
            let allArticles = [];
            let startIndex = 1;
            
            while (allArticles.length < maxResults && startIndex <= 100) {
                const response = await axios.get(searchUrl, {
                    params: {
                        query: companyName,
                        display: Math.min(100, maxResults - allArticles.length),
                        start: startIndex,
                        sort: 'date'
                    },
                    headers: {
                        'X-Naver-Client-Id': this.clientId,
                        'X-Naver-Client-Secret': this.clientSecret
                    },
                    timeout: 60000 // 60초 타임아웃
                });

                if (response.data.items && response.data.items.length > 0) {
                    const filteredArticles = response.data.items.filter(item => {
                        const title = this.removeHtmlTags(item.title);
                        const description = this.removeHtmlTags(item.description);
                        
                        // 회사명 관련 키워드 확장
                        const keywords = [companyName];
                        if (companyName === '현대차') {
                            keywords.push('현대자동차', '현대모비스', '현대');
                        } else if (companyName === '삼성전자') {
                            keywords.push('삼성', 'Samsung');
                        } else if (companyName === 'SK하이닉스') {
                            keywords.push('SK', '하이닉스');
                        }
                        
                        // 제목이나 설명에 관련 키워드가 포함되어 있으면 포함
                        return keywords.some(keyword => 
                            title.includes(keyword) || description.includes(keyword)
                        );
                    });

                    allArticles.push(...filteredArticles);
                    startIndex += response.data.items.length;
                    
                    if (response.data.items.length < 100) break;
                } else {
                    break;
                }
            }

            if (allArticles.length > 0) {
                const processedArticles = allArticles.slice(0, maxResults).map(item => ({
                    title: this.removeHtmlTags(item.title),
                    description: this.removeHtmlTags(item.description),
                    link: item.link,
                    pubDate: this.formatDate(item.pubDate),
                    sentiment: null
                }));

                const articlesWithSentiment = await this.analyzeSentimentIndividually(processedArticles);
                const sentimentSummary = this.generateSentimentSummary(articlesWithSentiment);

                return {
                    articles: articlesWithSentiment,
                    summary: this.generateNewsSummary(articlesWithSentiment),
                    sentimentSummary,
                    total: articlesWithSentiment.length
                };
            }

            return { 
                articles: [], 
                summary: '관련 뉴스를 찾을 수 없습니다.',
                sentimentSummary: { positive: 0, neutral: 0, negative: 0, total: 0 },
                total: 0
            };
        } catch (error) {
            console.error('네이버 뉴스 조회 오류:', error);
            return { 
                articles: [], 
                summary: '뉴스 정보 조회 중 오류가 발생했습니다.',
                error: error.message,
                sentimentSummary: { positive: 0, neutral: 0, negative: 0, total: 0 },
                total: 0
            };
        }
    }

    async analyzeSentimentIndividually(articles) {
        // 배치 크기 제한으로 성능 개선
        const batchSize = 3;
        const results = [];
        
        for (let i = 0; i < articles.length; i += batchSize) {
            const batch = articles.slice(i, i + batchSize);
            
            // 병렬 처리로 성능 향상
            const batchPromises = batch.map(async (article) => {
                try {
                    return await this.analyzeSingleArticle(article);
                } catch (error) {
                    console.error('개별 기사 분석 오류:', error);
                    const fallbackResult = this.getEnhancedSentiment(article.title, article.description);
                    return {
                        ...article,
                        sentiment: fallbackResult.sentiment,
                        reason: fallbackResult.reason
                    };
                }
            });
            
            const batchResults = await Promise.all(batchPromises);
            results.push(...batchResults);
            
            // 배치 간 간격 (rate limit 방지)
            if (i + batchSize < articles.length) {
                await this.sleep(200);
            }
        }
        
        return results;
    }

    async analyzeSingleArticle(article) {
        try {
            const response = await axios.post('https://api.openai.com/v1/chat/completions', {
                model: 'gpt-3.5-turbo',
                messages: [
                    {
                        role: 'system',
                        content: '당신은 전문 금융 분석가입니다. 주어진 뉴스 기사를 분석하여 해당 기업의 주가와 비즈니스에 미치는 구체적인 영향을 평가하세요. 투자자 관점에서 이 뉴스가 왜 긍정적/부정적/중립적인지 명확하고 구체적인 근거를 제시해야 합니다.'
                    },
                    {
                        role: 'user',
                        content: `뉴스를 분석하여 JSON으로 답변하세요:
제목: ${article.title}

{
  "sentiment": "긍정|중립|부정",
  "reason": "주가 영향을 한 문장으로 설명"
}`
                    }
                ],
                temperature: 0.1,
                max_tokens: 150
            }, {
                headers: {
                    'Authorization': `Bearer ${this.openaiApiKey}`,
                    'Content-Type': 'application/json'
                },
                timeout: 30000
            });

            const analysis = response.data.choices[0].message.content;
            return this.parseSingleSentimentResult(article, analysis);
            
        } catch (error) {
            console.error('OpenAI API 개별 분석 오류:', error);
            const fallbackResult = this.getEnhancedSentiment(article.title, article.description);
            return {
                ...article,
                sentiment: fallbackResult.sentiment,
                reason: fallbackResult.reason
            };
        }
    }

    parseSingleSentimentResult(article, analysis) {
        let sentiment = '중립';
        let reason = '분석 결과 없음';
        
        try {
            // JSON 파싱 시도
            const jsonMatch = analysis.match(/\{[\s\S]*\}/);
            if (jsonMatch) {
                const result = JSON.parse(jsonMatch[0]);
                if (result.sentiment && result.reason) {
                    sentiment = result.sentiment;
                    reason = result.reason;
                    
                    // 문장 완성 확인
                    if (!reason.match(/[.!?]$/)) {
                        reason += '.';
                    }
                }
            }
        } catch (e) {
            // JSON 파싱 실패시 기존 방식으로 fallback
            const sentimentMatch = analysis.match(/감정:\s*(긍정|중립|부정)/);
            const reasonMatch = analysis.match(/이유:\s*(.+?)(?=\n\n|$)/s);
            
            if (sentimentMatch) sentiment = sentimentMatch[1];
            if (reasonMatch) {
                reason = reasonMatch[1].trim().replace(/\n/g, ' ').replace(/\s+/g, ' ');
                
                // 문장 완성
                if (reason.length > 15 && !reason.match(/[.!?]$/)) {
                    const words = reason.split(' ');
                    if (words.length > 3) {
                        const lastWord = words[words.length - 1];
                        if (lastWord.length < 3 || !lastWord.match(/[가-힣]{2,}/)) {
                            words.pop();
                        }
                        reason = words.join(' ') + '.';
                    } else {
                        reason += '.';
                    }
                }
            }
        }
        
        // 분석 실패 시 fallback 사용
        if (reason === '분석 결과 없음' || reason.length < 15) {
            const fallbackResult = this.getEnhancedSentiment(article.title, article.description);
            sentiment = fallbackResult.sentiment;
            reason = fallbackResult.reason;
        }
        
        return {
            ...article,
            sentiment,
            reason
        };
    }

    async analyzeSentimentBatch(articles) {
        if (!this.openaiApiKey) {
            console.warn('OpenAI API 키가 설정되지 않아 기본 감정분석을 사용합니다.');
            return articles.map(article => {
                const sentimentResult = this.getEnhancedSentiment(article.title, article.description);
                return {
                    ...article, 
                    sentiment: sentimentResult.sentiment,
                    reason: sentimentResult.reason
                };
            });
        }

        // 먼저 기사 본문을 병렬로 크롤링
        const articlesWithContent = await this.fetchArticleContents(articles);
        
        const batchSize = 2; // 배치 크기를 줄여서 안정성 향상
        const results = [];

        for (let i = 0; i < articlesWithContent.length; i += batchSize) {
            const batch = articlesWithContent.slice(i, i + batchSize);
            
            try {
                const sentimentResults = await this.processSentimentBatchWithContent(batch);
                results.push(...sentimentResults);
            } catch (error) {
                console.error('감정분석 배치 처리 오류:', error);
                const defaultResults = batch.map(article => {
                    const sentimentResult = this.getEnhancedSentiment(article.title, article.description);
                    return { 
                        ...article, 
                        sentiment: sentimentResult.sentiment, 
                        reason: sentimentResult.reason
                    };
                });
                results.push(...defaultResults);
            }
            
            await this.sleep(1000); // 간격을 늘려서 안정성 향상
        }

        return results;
    }

    async processSentimentBatch(batch) {
        try {
            const response = await axios.post('https://api.openai.com/v1/chat/completions', {
                model: 'gpt-3.5-turbo',
                messages: [
                    {
                        role: 'system',
                        content: '당신은 전문 금융 분석가입니다. 뉴스 기사를 분석하여 해당 기업의 주가와 비즈니스에 미치는 구체적인 영향을 평가합니다. 매출, 수익성, 시장 점유율, 경쟁력, 규제 환경, 투자자 심리 등을 종합적으로 고려하여 투자 관점에서 정확한 분석을 제공하세요. 각 뉴스의 비즈니스 임팩트를 명확하고 구체적으로 설명하며, 단순한 키워드 분석이 아닌 깊이 있는 인사이트를 제공해야 합니다.'
                    },
                    {
                        role: 'user',
                        content: this.createBatchPrompt(batch)
                    }
                ],
                temperature: 0.2,
                max_tokens: 1500
            }, {
                headers: {
                    'Authorization': `Bearer ${this.openaiApiKey}`,
                    'Content-Type': 'application/json'
                },
                timeout: 20000 // 20초 타임아웃
            });

            const analysis = response.data.choices[0].message.content;
            return this.parseBatchSentimentResult(batch, analysis);
        } catch (error) {
            console.error('OpenAI API 호출 오류:', error);
            return batch.map(article => ({ 
                ...article, 
                sentiment: '중립', 
                reason: 'API 오류' 
            }));
        }
    }

    createBatchPrompt(batch) {
        let prompt = '다음 뉴스 기사들을 **기업의 주가와 사업 성과**에 미치는 영향을 중심으로 감정을 분석해주세요.\n\n';
        prompt += '**분석 기준:**\n';
        prompt += '- 긍정: 기업 가치 상승 요인 (매출/수익 증가, 신사업 확장, 기술혁신, 시장점유율 확대, 전략적 파트너십, 신제품 출시, 정부 지원, 투자 유치 등)\n';
        prompt += '- 부정: 기업 가치 하락 요인 (매출/수익 감소, 시장 축소, 경쟁 심화, 규제 강화, 법적 리스크, 제품 리콜, 경영진 교체, 구조조정 등)\n';
        prompt += '- 중립: 주가에 직접적 영향이 제한적인 일반적 기업 활동이나 정보성 내용\n\n';
        
        batch.forEach((article, index) => {
            prompt += `${index + 1}. 제목: ${article.title}\n`;
            prompt += `   내용: ${article.description}\n\n`;
        });
        
        prompt += '**중요: 기사의 전체 내용을 분석하여 기업에게 미치는 실질적 영향을 판단하세요.**\n';
        prompt += '각 기사에 대해 다음 형식으로 답변해주세요:\n';
        prompt += '1. 감정: (긍정/중립/부정), 이유: (이 뉴스가 해당 기업의 주가/실적에 미치는 구체적 영향을 한 문장으로 설명)\n';
        prompt += '2. 감정: (긍정/중립/부정), 이유: (이 뉴스가 해당 기업의 주가/실적에 미치는 구체적 영향을 한 문장으로 설명)\n';
        
        return prompt;
    }

    parseBatchSentimentResult(batch, analysis) {
        const lines = analysis.split('\n').filter(line => line.trim());
        const results = [];
        
        batch.forEach((article, index) => {
            const line = lines.find(l => l.startsWith(`${index + 1}.`));
            let sentiment = '중립';
            let reason = '분석 결과 없음';
            
            if (line) {
                const sentimentMatch = line.match(/감정:\s*(긍정|중립|부정)/);
                const reasonMatch = line.match(/이유:\s*(.+?)(?=\n|$)/);
                
                if (sentimentMatch) sentiment = sentimentMatch[1];
                if (reasonMatch) {
                    reason = reasonMatch[1].trim();
                    // 너무 짧거나 의미없는 답변 필터링
                    if (reason.length < 10 || ['네이버', 'API', '분석', '오류'].includes(reason)) {
                        reason = '분석 결과 없음';
                    }
                }
            }
            
            // OpenAI 분석 실패 시 개선된 기본 분석으로 대체
            if (reason === '분석 결과 없음') {
                const fallbackResult = this.getEnhancedSentiment(article.title, article.description);
                sentiment = fallbackResult.sentiment;
                reason = fallbackResult.reason;
            }
            
            results.push({
                ...article,
                sentiment,
                reason
            });
        });
        
        return results;
    }

    generateSentimentSummary(articles) {
        const summary = { positive: 0, neutral: 0, negative: 0 };
        
        articles.forEach(article => {
            switch (article.sentiment) {
                case '긍정':
                    summary.positive++;
                    break;
                case '부정':
                    summary.negative++;
                    break;
                default:
                    summary.neutral++;
                    break;
            }
        });
        
        const total = articles.length;
        return {
            ...summary,
            total,
            positiveRatio: total > 0 ? Math.round((summary.positive / total) * 100) : 0,
            neutralRatio: total > 0 ? Math.round((summary.neutral / total) * 100) : 0,
            negativeRatio: total > 0 ? Math.round((summary.negative / total) * 100) : 0
        };
    }

    removeHtmlTags(text) {
        return text.replace(/<[^>]*>/g, '');
    }

    formatDate(dateString) {
        const date = new Date(dateString);
        return date.toLocaleDateString('ko-KR', {
            year: 'numeric',
            month: '2-digit',
            day: '2-digit'
        });
    }

    generateNewsSummary(articles) {
        if (!articles || articles.length === 0) {
            return '최근 관련 뉴스가 없습니다.';
        }

        const titles = articles.slice(0, 3).map(article => article.title);
        return `최근 주요 뉴스 ${articles.length}건: ${titles.join(', ')}`;
    }

    async crawlNaverNews(companyName) {
        try {
            const searchUrl = `https://search.naver.com/search.naver?where=news&query=${encodeURIComponent(companyName)}`;
            const response = await axios.get(searchUrl, {
                headers: {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                },
                timeout: 10000
            });

            const $ = cheerio.load(response.data);
            const articles = [];

            $('.news_tit').each((index, element) => {
                if (index < 10) {
                    const title = $(element).text().trim();
                    const link = $(element).attr('href');
                    if (title && link) {
                        articles.push({ title, link });
                    }
                }
            });

            return articles;
        } catch (error) {
            console.error('네이버 뉴스 크롤링 오류:', error);
            return [];
        }
    }

    async crawlArticleContent(url) {
        try {
            const response = await axios.get(url, {
                headers: {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                },
                timeout: 10000
            });

            const $ = cheerio.load(response.data);
            
            // 다양한 뉴스 사이트 선택자
            const contentSelectors = [
                'div#newsct_article',  // 네이버 뉴스
                '.news_end',           // 일반 뉴스
                '._article_content',   // 뉴스 콘텐츠
                'article',             // HTML5 article 태그
                '.article-content',    // 기사 내용
                '.article_body',       // 기사 본문
                '#content'             // 일반 콘텐츠
            ];
            
            let content = null;
            for (const selector of contentSelectors) {
                const element = $(selector);
                if (element.length > 0) {
                    content = element.text().trim().replace(/\s+/g, ' ').substring(0, 2000); // 2000자로 제한
                    if (content.length > 100) { // 최소 길이 확인
                        break;
                    }
                }
            }
            
            return content || null;
        } catch (error) {
            console.error('기사 내용 크롤링 오류:', error);
            return null;
        }
    }

    getEnhancedSentiment(title, description = '') {
        const content = `${title} ${description}`.toLowerCase();
        
        // 비즈니스 임팩트 기반 키워드 분석
        const businessPositive = [
            { keywords: ['매출', '수익', '이익', '실적'], impact: '매출 성과 개선으로 주가에 긍정적 영향' },
            { keywords: ['투자', '확대', '진출', '확장'], impact: '사업 확장으로 성장 가능성 증대' },
            { keywords: ['계약', '체결', '파트너십', '제휴'], impact: '새로운 사업 기회 창출로 긍정적 전망' },
            { keywords: ['개발', '출시', '신제품', '혁신'], impact: '신기술/제품 개발로 경쟁력 강화' },
            { keywords: ['성장', '증가', '상승', '호조'], impact: '성장세 지속으로 투자자 신뢰도 상승' }
        ];
        
        const businessNegative = [
            { keywords: ['손실', '적자', '감소', '하락'], impact: '재무 성과 악화로 주가 하락 요인' },
            { keywords: ['리콜', '결함', '문제', '사고'], impact: '제품 품질 이슈로 기업 신뢰도 하락' },
            { keywords: ['소송', '제재', '규제', '조사'], impact: '법적 리스크로 경영 불확실성 증가' },
            { keywords: ['경쟁', '점유율', '밀려', '추락'], impact: '시장 경쟁력 약화로 성장성 우려' },
            { keywords: ['중단', '연기', '취소', '철회'], impact: '사업 계획 차질로 성장 동력 약화' }
        ];
        
        // 긍정적 영향 분석
        for (const category of businessPositive) {
            const foundKeywords = category.keywords.filter(keyword => content.includes(keyword));
            if (foundKeywords.length > 0) {
                return { 
                    sentiment: '긍정', 
                    reason: category.impact 
                };
            }
        }
        
        // 부정적 영향 분석
        for (const category of businessNegative) {
            const foundKeywords = category.keywords.filter(keyword => content.includes(keyword));
            if (foundKeywords.length > 0) {
                return { 
                    sentiment: '부정', 
                    reason: category.impact 
                };
            }
        }
        
        // 중립적 판단
        const neutralIndicators = ['발표', '예정', '계획', '검토', '논의', '회의'];
        const hasNeutralContent = neutralIndicators.some(keyword => content.includes(keyword));
        
        if (hasNeutralContent) {
            return { 
                sentiment: '중립', 
                reason: '기업 활동 관련 정보성 내용으로 주가 영향 제한적' 
            };
        }
        
        return { 
            sentiment: '중립', 
            reason: '기업 실적에 직접적 영향이 불분명하여 중립적 판단' 
        };
    }

    getBasicSentiment(title) {
        // 기존 메서드는 호환성을 위해 유지하되 새로운 메서드로 리다이렉트
        return this.getEnhancedSentiment(title, '');
    }

    async fetchArticleContents(articles) {
        const articlesWithContent = [];
        
        for (const article of articles) {
            try {
                const content = await this.crawlArticleContent(article.link);
                articlesWithContent.push({
                    ...article,
                    content: content || article.description || article.title
                });
            } catch (error) {
                console.error('기사 본문 가져오기 실패:', error);
                articlesWithContent.push({
                    ...article,
                    content: article.description || article.title
                });
            }
            
            // 크롤링 간격
            await this.sleep(300);
        }
        
        return articlesWithContent;
    }

    async processSentimentBatchWithContent(batch) {
        try {
            const messages = [
                {
                    role: 'system',
                    content: '당신은 금융 분석가입니다. 뉴스 기사가 기업의 주가와 사업 실적에 미치는 영향을 정확하게 분석하여 투자자 관점에서 감정을 분류하고, 구체적인 비즈니스 영향을 설명합니다.'
                }
            ];

            let userPrompt = '다음 뉴스 기사들을 분석해서 기업에 미치는 영향을 판단해주세요.\n\n';
            userPrompt += '**분석 기준:**\n';
            userPrompt += '- 긍정: 매출/수익 증가, 투자 유치, 신제품 출시, 파트너십 체결 등 주가 상승 요인\n';
            userPrompt += '- 부정: 손실/적자, 리콜/결함, 소송/제재, 경쟁 심화 등 주가 하락 요인\n';
            userPrompt += '- 중립: 일반 정보, 단순 발표, 직접적 영향이 불분명한 내용\n\n';

            batch.forEach((article, index) => {
                userPrompt += `${index + 1}. 제목: ${article.title}\n`;
                userPrompt += `   요약: ${article.description}\n`;
                userPrompt += `   본문: ${article.content.substring(0, 1000)}\n\n`;
            });

            userPrompt += '각 기사에 대해 다음 형식으로 답변하되, 반드시 완전한 문장으로 마무리해주세요:\n';
            userPrompt += '1. 감정: (긍정/중립/부정), 이유: (이 뉴스가 해당 기업의 주가나 실적에 미치는 구체적인 영향을 한 문장으로 명확히 설명)\n';
            userPrompt += '2. 감정: (긍정/중립/부정), 이유: (이 뉴스가 해당 기업의 주가나 실적에 미치는 구체적인 영향을 한 문장으로 명확히 설명)\n';

            messages.push({
                role: 'user',
                content: userPrompt
            });

            const response = await axios.post('https://api.openai.com/v1/chat/completions', {
                model: 'gpt-3.5-turbo',
                messages: messages,
                temperature: 0.1,
                max_tokens: 1000,
                presence_penalty: 0.1,
                frequency_penalty: 0.1
            }, {
                headers: {
                    'Authorization': `Bearer ${this.openaiApiKey}`,
                    'Content-Type': 'application/json'
                },
                timeout: 30000
            });

            const analysis = response.data.choices[0].message.content;
            return this.parseBatchSentimentResult(batch, analysis);
        } catch (error) {
            console.error('OpenAI API 호출 오류:', error);
            return batch.map(article => ({ 
                ...article, 
                sentiment: '중립', 
                reason: 'API 분석 실패로 중립 처리' 
            }));
        }
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

module.exports = new NaverService();