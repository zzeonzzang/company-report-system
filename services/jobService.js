class JobService {
    constructor() {
        this.companyDatabase = this.initializeCompanyDatabase();
    }

    initializeCompanyDatabase() {
        return {
            '현대차': {
                stockCode: '005380',
                market: 'KOSPI',
                sector: '자동차',
                employeeCount: 70439,
                marketCap: 30000000,
                revenue: 117611000,
                listingDate: '1974-10-02',
                careerSite: 'https://careers.hyundai.com/'
            },
            '현대자동차': {
                stockCode: '005380',
                market: 'KOSPI',
                sector: '자동차',
                employeeCount: 70439,
                marketCap: 30000000,
                revenue: 117611000,
                listingDate: '1974-10-02',
                careerSite: 'https://careers.hyundai.com/'
            },
            '삼성전자': {
                stockCode: '005930',
                market: 'KOSPI', 
                sector: '반도체',
                employeeCount: 267937,
                marketCap: 400000000,
                revenue: 279651000,
                listingDate: '1975-06-11',
                careerSite: 'https://www.samsung.com/sec/about-us/careers/'
            },
            'SK하이닉스': {
                stockCode: '000660',
                market: 'KOSPI',
                sector: '반도체', 
                employeeCount: 29415,
                marketCap: 80000000,
                revenue: 44819000,
                listingDate: '1996-12-26',
                careerSite: 'https://careers.skhynix.com/'
            },
            '기아': {
                stockCode: '000270',
                market: 'KOSPI',
                sector: '자동차',
                employeeCount: 52713,
                marketCap: 25000000,
                revenue: 89094000,
                listingDate: '1973-07-10',
                careerSite: 'https://careers.kia.com/'
            },
            'LG에너지솔루션': {
                stockCode: '373220',
                market: 'KOSPI',
                sector: '배터리',
                employeeCount: 26586,
                marketCap: 70000000,
                revenue: 27307000,
                listingDate: '2022-01-27',
                careerSite: 'https://www.lgensol.com/careers'
            },
            'NAVER': {
                stockCode: '035420',
                market: 'KOSPI',
                sector: '인터넷',
                employeeCount: 3793,
                marketCap: 35000000,
                revenue: 8487000,
                listingDate: '2002-10-29',
                careerSite: 'https://career.navercorp.com/'
            },
            '카카오': {
                stockCode: '035720',
                market: 'KOSPI',
                sector: '인터넷',
                employeeCount: 4479,
                marketCap: 25000000,
                revenue: 6671000,
                listingDate: '2017-07-10',
                careerSite: 'https://careers.kakao.com/'
            }
        };
    }

    getCompanyJobInfo(companyName) {
        const companyInfo = this.companyDatabase[companyName];
        
        if (!companyInfo) {
            return {
                employeeCount: '정보없음',
                newGradJobs: '신입 0건',
                experiencedJobs: '경력 0건'
            };
        }
        
        return {
            employeeCount: this.formatEmployeeCount(companyInfo.employeeCount),
            newGradJobs: '신입 2건',
            experiencedJobs: '경력 3건'
        };
    }

    generateJobPostings(companyName) {
        const today = new Date();
        const formatDate = (date) => {
            return date.toISOString().split('T')[0];
        };

        const basicPositions = [
            {
                title: '신입사원 공개채용',
                experience: '신입',
                location: '본사',
                summary: '각 부문별 신입사원 모집'
            },
            {
                title: '경력직 채용',
                experience: '3년 이상',
                location: '전국',
                summary: '전문 분야 경력직 모집'
            },
            {
                title: '연구개발직',
                experience: '석사 이상',
                location: '연구소',
                summary: 'R&D 연구원 모집'
            },
            {
                title: '영업/마케팅',
                experience: '2년 이상',
                location: '전국',
                summary: '영업 및 마케팅 전문가 모집'
            },
            {
                title: 'IT/개발직',
                experience: '경력무관',
                location: '본사',
                summary: '소프트웨어 개발자 모집'
            }
        ];

        return basicPositions.map((position, index) => ({
            id: index + 1,
            title: `${companyName} ${position.title}`,
            company: companyName,
            location: position.location,
            experience: position.experience,
            source: '추정정보',
            url: `https://careers.${companyName.toLowerCase().replace(/\s/g, '')}.com/`,
            postedDate: formatDate(new Date(today.setDate(today.getDate() - Math.floor(Math.random() * 30)))),
            summary: position.summary
        }));
    }

    getJobPlatformLinks() {
        return [
            {
                name: '사람인',
                url: 'https://www.saramin.co.kr/',
                description: '국내 최대 채용 플랫폼'
            },
            {
                name: '잡코리아',
                url: 'https://www.jobkorea.co.kr/',
                description: '대기업 채용 정보'
            },
            {
                name: '원티드',
                url: 'https://www.wanted.co.kr/',
                description: 'IT/스타트업 전문'
            },
            {
                name: '링크드인',
                url: 'https://www.linkedin.com/',
                description: '글로벌 네트워킹'
            }
        ];
    }

    formatEmployeeCount(count) {
        if (count >= 10000) {
            return `${Math.floor(count / 10000)}만 ${count % 10000}명`;
        }
        return `${count.toLocaleString()}명`;
    }

    formatMarketCap(marketCap) {
        if (marketCap >= 10000) {
            return `${Math.floor(marketCap / 10000)}조원`;
        }
        return `${marketCap.toLocaleString()}억원`;
    }
}

module.exports = new JobService();