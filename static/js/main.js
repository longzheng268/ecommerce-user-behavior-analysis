/**
 * E-Commerce Analysis Dashboard - Main JavaScript
 * Handles AJAX calls, chart initialization, and animations
 */

// ============================================================
// Chart Theme Configuration
// ============================================================
const CHART_COLORS = {
    primary: '#667eea',
    secondary: '#764ba2',
    accent: '#f093fb',
    success: '#38ef7d',
    warning: '#ffc107',
    danger: '#f5576c',
    info: '#4facfe',
    cyan: '#00f2fe',
    text: '#9b97b0',
    textLight: '#6b6880',
    gridLine: 'rgba(102, 126, 234, 0.08)',
    bg: 'transparent'
};

const PALETTE = ['#667eea', '#764ba2', '#f093fb', '#38ef7d', '#ffc107', '#f5576c', '#4facfe', '#00f2fe', '#ff9a9e', '#a18cd1'];

function getGradientColor(chart, color1, color2) {
    const ctx = chart.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, chart.getHeight ? chart.getHeight() : 320);
    gradient.addColorStop(0, color1);
    gradient.addColorStop(1, color2);
    return gradient;
}

// ============================================================
// Utility Functions
// ============================================================
function formatNumber(num) {
    if (num >= 10000) return (num / 10000).toFixed(1) + '万';
    return num.toLocaleString('zh-CN');
}

function showSpinner(containerId) {
    const container = document.getElementById(containerId);
    if (container) {
        container.innerHTML = '<div class="loading-spinner"><div class="spinner"></div><span class="spinner-text">加载中...</span></div>';
    }
}

function fetchJSON(url) {
    return fetch(url).then(res => res.json());
}

// ============================================================
// Intersection Observer for lazy loading charts
// ============================================================
const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('visible');
            observer.unobserve(entry.target);
        }
    });
}, { threshold: 0.1 });

// ============================================================
// Initialize Dashboard
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    // Observe all chart cards
    document.querySelectorAll('.chart-card').forEach(card => {
        observer.observe(card);
    });

    // Update time
    updateTime();
    setInterval(updateTime, 1000);

    // Load all data
    loadOverview();
    loadBehaviorStats();
    loadHourlyStats();
    loadCategoryStats();
    loadRFMStats();
    loadClusterStats();
    loadARIMAStats();
    loadMLStats();
    loadMarketingStats();
    loadAprioriStats();
});

function updateTime() {
    const el = document.getElementById('current-time');
    if (el) {
        const now = new Date();
        el.textContent = now.toLocaleString('zh-CN', {
            year: 'numeric', month: '2-digit', day: '2-digit',
            hour: '2-digit', minute: '2-digit', second: '2-digit'
        });
    }
}

// ============================================================
// 1. Overview KPI Cards
// ============================================================
function loadOverview() {
    fetchJSON('/api/overview').then(data => {
        document.getElementById('kpi-users').textContent = formatNumber(data.total_users);
        document.getElementById('kpi-orders').textContent = formatNumber(data.total_orders);
        document.getElementById('kpi-conversion').textContent = data.conversion_rate + '%';
        document.getElementById('kpi-aov').textContent = '¥' + data.avg_order_value.toFixed(1);
        document.getElementById('kpi-revenue').textContent = '¥' + formatNumber(Math.round(data.total_revenue));
        document.getElementById('kpi-purchases').textContent = formatNumber(data.total_purchases);
    });
}

// ============================================================
// 2. Behavior Distribution (Pie Chart) + 24h Trend (Line Chart)
// ============================================================
function loadBehaviorStats() {
    showSpinner('behavior-chart');
    fetchJSON('/api/behavior_stats').then(data => {
        const chart = echarts.init(document.getElementById('behavior-chart'));
        const option = {
            tooltip: {
                trigger: 'item',
                backgroundColor: 'rgba(15, 12, 41, 0.95)',
                borderColor: 'rgba(102, 126, 234, 0.3)',
                textStyle: { color: '#e8e6f0', fontSize: 12 },
                formatter: '{b}: {c} ({d}%)'
            },
            legend: {
                orient: 'vertical',
                right: 10,
                top: 'center',
                textStyle: { color: CHART_COLORS.text, fontSize: 11 },
                itemWidth: 10,
                itemHeight: 10,
                itemGap: 12
            },
            series: [{
                type: 'pie',
                radius: ['45%', '72%'],
                center: ['40%', '50%'],
                avoidLabelOverlap: true,
                itemStyle: {
                    borderRadius: 6,
                    borderColor: 'rgba(15, 12, 41, 0.8)',
                    borderWidth: 2
                },
                label: {
                    show: false
                },
                emphasis: {
                    label: {
                        show: true,
                        fontSize: 14,
                        fontWeight: 'bold',
                        color: '#e8e6f0'
                    },
                    itemStyle: {
                        shadowBlur: 20,
                        shadowColor: 'rgba(102, 126, 234, 0.5)'
                    }
                },
                data: data.map((d, i) => ({
                    name: d.name,
                    value: d.value,
                    itemStyle: { color: PALETTE[i % PALETTE.length] }
                })),
                animationType: 'scale',
                animationEasing: 'elasticOut',
                animationDelay: (idx) => idx * 100
            }]
        };
        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());
    });
}

function loadHourlyStats() {
    showSpinner('hourly-chart');
    fetchJSON('/api/hourly_stats').then(data => {
        const chart = echarts.init(document.getElementById('hourly-chart'));
        const hours = data.map(d => d.hour + ':00');
        const counts = data.map(d => d.count);
        const option = {
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(15, 12, 41, 0.95)',
                borderColor: 'rgba(102, 126, 234, 0.3)',
                textStyle: { color: '#e8e6f0', fontSize: 12 },
                axisPointer: { type: 'cross', crossStyle: { color: '#667eea' } }
            },
            grid: { left: 50, right: 20, top: 20, bottom: 30 },
            xAxis: {
                type: 'category',
                data: hours,
                axisLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLabel: { color: CHART_COLORS.textLight, fontSize: 10, interval: 2 },
                axisTick: { show: false }
            },
            yAxis: {
                type: 'value',
                splitLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLabel: { color: CHART_COLORS.textLight, fontSize: 10 }
            },
            series: [{
                type: 'line',
                data: counts,
                smooth: true,
                symbol: 'circle',
                symbolSize: 4,
                lineStyle: { width: 2.5, color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                    { offset: 0, color: '#667eea' },
                    { offset: 1, color: '#f093fb' }
                ])},
                areaStyle: {
                    color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                        { offset: 0, color: 'rgba(102, 126, 234, 0.3)' },
                        { offset: 1, color: 'rgba(102, 126, 234, 0.01)' }
                    ])
                },
                itemStyle: { color: '#667eea' },
                animationDuration: 1500,
                animationEasing: 'cubicOut'
            }]
        };
        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());
    });
}

// ============================================================
// 3. Category Conversion Rate
// ============================================================
function loadCategoryStats() {
    showSpinner('category-chart');
    fetchJSON('/api/category_stats').then(data => {
        const chart = echarts.init(document.getElementById('category-chart'));
        const categories = data.map(d => d.category);
        const conversions = data.map(d => d.conversion);
        const option = {
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(15, 12, 41, 0.95)',
                borderColor: 'rgba(102, 126, 234, 0.3)',
                textStyle: { color: '#e8e6f0', fontSize: 12 },
                axisPointer: { type: 'shadow' },
                formatter: (params) => {
                    const d = data[params[0].dataIndex];
                    return `<strong>${d.category}</strong><br/>浏览: ${d.views}<br/>购买: ${d.purchases}<br/>转化率: ${d.conversion}%`;
                }
            },
            grid: { left: 110, right: 60, top: 10, bottom: 20 },
            xAxis: {
                type: 'value',
                splitLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLabel: { color: CHART_COLORS.textLight, fontSize: 10, formatter: '{value}%' }
            },
            yAxis: {
                type: 'category',
                data: categories,
                axisLine: { show: false },
                axisTick: { show: false },
                axisLabel: { color: CHART_COLORS.text, fontSize: 11 }
            },
            series: [{
                type: 'bar',
                data: conversions,
                barWidth: 16,
                itemStyle: {
                    borderRadius: [0, 4, 4, 0],
                    color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                        { offset: 0, color: '#667eea' },
                        { offset: 1, color: '#764ba2' }
                    ])
                },
                emphasis: {
                    itemStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 1, 0, [
                            { offset: 0, color: '#f093fb' },
                            { offset: 1, color: '#667eea' }
                        ])
                    }
                },
                animationDuration: 1500,
                animationEasing: 'cubicOut'
            }]
        };
        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());
    });
}

// ============================================================
// 4. RFM Analysis (Scatter Chart)
// ============================================================
function loadRFMStats() {
    showSpinner('rfm-chart');
    fetchJSON('/api/rfm_stats').then(data => {
        const chart = echarts.init(document.getElementById('rfm-chart'));
        const option = {
            tooltip: {
                backgroundColor: 'rgba(15, 12, 41, 0.95)',
                borderColor: 'rgba(102, 126, 234, 0.3)',
                textStyle: { color: '#e8e6f0', fontSize: 12 },
                formatter: (params) => `Recency: ${params.data[0]}天<br/>Frequency: ${params.data[1]}次<br/>Monetary: ¥${params.data[2]}`
            },
            grid: { left: 55, right: 75, top: 20, bottom: 40 },
            xAxis: {
                name: 'Recency (天)',
                nameTextStyle: { color: CHART_COLORS.textLight, fontSize: 11 },
                splitLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLabel: { color: CHART_COLORS.textLight, fontSize: 10 }
            },
            yAxis: {
                name: 'Frequency',
                nameTextStyle: { color: CHART_COLORS.textLight, fontSize: 11 },
                splitLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLabel: { color: CHART_COLORS.textLight, fontSize: 10 }
            },
            visualMap: {
                min: Math.min(...data.map(d => d.monetary)),
                max: Math.max(...data.map(d => d.monetary)),
                dimension: 2,
                orient: 'vertical',
                right: 0,
                top: 'center',
                text: ['高消费', '低消费'],
                textStyle: { color: CHART_COLORS.text, fontSize: 10 },
                inRange: {
                    color: ['#667eea', '#764ba2', '#f093fb', '#f5576c', '#ffc107']
                },
                itemWidth: 12,
                itemHeight: 140
            },
            series: [{
                type: 'scatter',
                data: data.map(d => [d.recency, d.frequency, d.monetary]),
                symbolSize: 6,
                itemStyle: {
                    opacity: 0.7
                },
                emphasis: {
                    itemStyle: {
                        opacity: 1,
                        shadowBlur: 10,
                        shadowColor: 'rgba(102, 126, 234, 0.5)'
                    }
                },
                animationDuration: 1500
            }]
        };
        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());
    });
}

// ============================================================
// 5. K-Means Clustering (Radar Chart)
// ============================================================
function loadClusterStats() {
    showSpinner('cluster-chart');
    fetchJSON('/api/cluster_stats').then(data => {
        // Radar chart
        const chart = echarts.init(document.getElementById('cluster-chart'));
        const colors = PALETTE;
        const series = data.clusters.map((c, i) => ({
            name: c.name,
            value: c.radar_values,
            symbol: 'circle',
            symbolSize: 5,
            lineStyle: { width: 2, color: colors[i % colors.length] },
            areaStyle: { color: colors[i % colors.length], opacity: 0.1 },
            itemStyle: { color: colors[i % colors.length] }
        }));

        const option = {
            tooltip: {
                backgroundColor: 'rgba(15, 12, 41, 0.95)',
                borderColor: 'rgba(102, 126, 234, 0.3)',
                textStyle: { color: '#e8e6f0', fontSize: 12 }
            },
            legend: {
                bottom: 0,
                textStyle: { color: CHART_COLORS.text, fontSize: 11 },
                itemWidth: 10,
                itemHeight: 10,
                itemGap: 16
            },
            radar: {
                indicator: data.indicators.map(name => ({ name, max: 1 })),
                shape: 'polygon',
                splitNumber: 4,
                axisName: { color: CHART_COLORS.text, fontSize: 11 },
                splitLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                splitArea: { show: false },
                axisLine: { lineStyle: { color: CHART_COLORS.gridLine } }
            },
            series: [{
                type: 'radar',
                data: series,
                animationDuration: 1500
            }]
        };
        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());

        // Cluster cards
        const cardsContainer = document.getElementById('cluster-cards');
        if (cardsContainer) {
            cardsContainer.innerHTML = data.clusters.map(c => `
                <div class="cluster-item">
                    <div class="cluster-name">${c.name}</div>
                    <div class="cluster-stat"><span>用户数</span><span>${c.count}</span></div>
                    <div class="cluster-stat"><span>平均消费</span><span>¥${c.avg_monetary.toFixed(0)}</span></div>
                    <div class="cluster-stat"><span>平均频率</span><span>${c.avg_frequency.toFixed(1)}次</span></div>
                    <div class="cluster-stat"><span>平均Recency</span><span>${c.avg_recency.toFixed(0)}天</span></div>
                </div>
            `).join('');
        }
    });
}

// ============================================================
// 6. ARIMA Forecast (Area Chart with Confidence Band)
// ============================================================
function loadARIMAStats() {
    showSpinner('arima-chart');
    fetchJSON('/api/arima_stats').then(data => {
        const chart = echarts.init(document.getElementById('arima-chart'));

        // Sample historical data (show last 60 days)
        const historical = data.historical.slice(-60);
        const fitted = data.fitted.slice(-60);
        const predictions = data.predictions;

        const allDates = [...historical.map(d => d.date), ...predictions.map(d => d.date)];

        // Historical line
        const historicalValues = historical.map(d => d.value);
        // Fitted line (aligned with historical)
        const fittedValues = fitted.map(d => d.value);
        // Prediction line
        const predictionValues = [...new Array(historical.length).fill(null), ...predictions.map(d => d.value)];
        // Confidence band
        const upperBand = [...new Array(historical.length).fill(null), ...predictions.map(d => d.upper)];
        const lowerBand = [...new Array(historical.length).fill(null), ...predictions.map(d => d.lower)];

        const option = {
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(15, 12, 41, 0.95)',
                borderColor: 'rgba(102, 126, 234, 0.3)',
                textStyle: { color: '#e8e6f0', fontSize: 12 }
            },
            legend: {
                data: ['实际值', '拟合值', '预测值'],
                top: 0,
                textStyle: { color: CHART_COLORS.text, fontSize: 11 },
                itemWidth: 16,
                itemHeight: 8,
                itemGap: 20
            },
            grid: { left: 50, right: 20, top: 40, bottom: 30 },
            xAxis: {
                type: 'category',
                data: allDates,
                axisLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLabel: {
                    color: CHART_COLORS.textLight,
                    fontSize: 9,
                    rotate: 30,
                    interval: Math.floor(allDates.length / 10)
                },
                axisTick: { show: false }
            },
            yAxis: {
                type: 'value',
                splitLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLabel: { color: CHART_COLORS.textLight, fontSize: 10 }
            },
            series: [
                {
                    name: '实际值',
                    type: 'line',
                    data: [...historicalValues, ...new Array(predictions.length).fill(null)],
                    smooth: true,
                    lineStyle: { width: 1.5, color: '#4facfe' },
                    itemStyle: { color: '#4facfe' },
                    symbol: 'none',
                    animationDuration: 1500
                },
                {
                    name: '拟合值',
                    type: 'line',
                    data: [...fittedValues, ...new Array(predictions.length).fill(null)],
                    smooth: true,
                    lineStyle: { width: 1.5, color: '#ffc107', type: 'dashed' },
                    itemStyle: { color: '#ffc107' },
                    symbol: 'none',
                    animationDuration: 1500
                },
                {
                    name: '预测值',
                    type: 'line',
                    data: predictionValues,
                    smooth: true,
                    lineStyle: { width: 2.5, color: '#f5576c' },
                    itemStyle: { color: '#f5576c' },
                    symbol: 'circle',
                    symbolSize: 3,
                    animationDuration: 2000
                },
                {
                    name: '置信上界',
                    type: 'line',
                    data: upperBand,
                    lineStyle: { opacity: 0 },
                    areaStyle: { color: 'rgba(245, 87, 108, 0.08)' },
                    stack: 'confidence',
                    symbol: 'none',
                    silent: true
                },
                {
                    name: '置信下界',
                    type: 'line',
                    data: lowerBand,
                    lineStyle: { opacity: 0 },
                    areaStyle: { color: 'rgba(245, 87, 108, 0.12)' },
                    stack: 'confidence-upper',
                    symbol: 'none',
                    silent: true
                }
            ]
        };
        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());

        // Update ARIMA info
        const infoEl = document.getElementById('arima-info');
        if (infoEl) {
            infoEl.innerHTML = `ARIMA(${data.order.join(',')})  |  RMSE: ${data.rmse}  |  MAE: ${data.mae}`;
        }
    });
}

// ============================================================
// 7. Machine Learning ROC Curve
// ============================================================
function loadMLStats() {
    showSpinner('ml-chart');
    fetchJSON('/api/ml_stats').then(data => {
        const chart = echarts.init(document.getElementById('ml-chart'));

        const option = {
            tooltip: {
                backgroundColor: 'rgba(15, 12, 41, 0.95)',
                borderColor: 'rgba(102, 126, 234, 0.3)',
                textStyle: { color: '#e8e6f0', fontSize: 12 }
            },
            legend: {
                data: [
                    `Random Forest (AUC=${data.rf.auc})`,
                    `MLP Neural Net (AUC=${data.mlp.auc})`
                ],
                top: 0,
                textStyle: { color: CHART_COLORS.text, fontSize: 11 },
                itemWidth: 16,
                itemHeight: 8
            },
            grid: { left: 50, right: 20, top: 40, bottom: 40 },
            xAxis: {
                type: 'value',
                name: 'False Positive Rate',
                nameTextStyle: { color: CHART_COLORS.textLight, fontSize: 11 },
                min: 0,
                max: 1,
                splitLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLabel: { color: CHART_COLORS.textLight, fontSize: 10 }
            },
            yAxis: {
                type: 'value',
                name: 'True Positive Rate',
                nameTextStyle: { color: CHART_COLORS.textLight, fontSize: 11 },
                min: 0,
                max: 1,
                splitLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLabel: { color: CHART_COLORS.textLight, fontSize: 10 }
            },
            series: [
                {
                    name: `Random Forest (AUC=${data.rf.auc})`,
                    type: 'line',
                    data: data.rf.roc.fpr.map((fpr, i) => [fpr, data.rf.roc.tpr[i]]),
                    smooth: true,
                    lineStyle: { width: 2.5, color: '#667eea' },
                    itemStyle: { color: '#667eea' },
                    symbol: 'none',
                    animationDuration: 1500
                },
                {
                    name: `MLP Neural Net (AUC=${data.mlp.auc})`,
                    type: 'line',
                    data: data.mlp.roc.fpr.map((fpr, i) => [fpr, data.mlp.roc.tpr[i]]),
                    smooth: true,
                    lineStyle: { width: 2.5, color: '#f093fb' },
                    itemStyle: { color: '#f093fb' },
                    symbol: 'none',
                    animationDuration: 1500
                },
                {
                    name: 'Random',
                    type: 'line',
                    data: [[0, 0], [1, 1]],
                    lineStyle: { width: 1, color: '#6b6880', type: 'dashed' },
                    symbol: 'none',
                    silent: true
                }
            ]
        };
        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());

        // Update ML metrics
        const rfEl = document.getElementById('rf-accuracy');
        const mlpEl = document.getElementById('mlp-accuracy');
        if (rfEl) rfEl.textContent = (data.rf.accuracy * 100).toFixed(2) + '%';
        if (mlpEl) mlpEl.textContent = (data.mlp.accuracy * 100).toFixed(2) + '%';
    });
}

// ============================================================
// 8. Marketing Campaign Results (Grouped Bar Chart)
// ============================================================
function loadMarketingStats() {
    showSpinner('marketing-chart');
    fetchJSON('/api/marketing_stats').then(data => {
        const chart = echarts.init(document.getElementById('marketing-chart'));
        const campaigns = data.map(d => d.name);
        const beforeData = data.map(d => d.before_purchases);
        const duringData = data.map(d => d.during_purchases);
        const option = {
            tooltip: {
                trigger: 'axis',
                backgroundColor: 'rgba(15, 12, 41, 0.95)',
                borderColor: 'rgba(102, 126, 234, 0.3)',
                textStyle: { color: '#e8e6f0', fontSize: 12 },
                axisPointer: { type: 'shadow' },
                formatter: (params) => {
                    const idx = params[0].dataIndex;
                    const d = data[idx];
                    return `<strong>${d.name}</strong> (${d.type})<br/>` +
                        `活动前购买量: ${d.before_purchases}<br/>` +
                        `活动中购买量: ${d.during_purchases}<br/>` +
                        `提升率: ${d.lift > 0 ? '+' : ''}${d.lift}%<br/>` +
                        `A/B组: ${d.group}`;
                }
            },
            legend: {
                data: ['活动前', '活动中'],
                top: 0,
                textStyle: { color: CHART_COLORS.text, fontSize: 11 },
                itemWidth: 16,
                itemHeight: 8
            },
            grid: { left: 50, right: 20, top: 40, bottom: 60 },
            xAxis: {
                type: 'category',
                data: campaigns,
                axisLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLabel: { color: CHART_COLORS.textLight, fontSize: 10, rotate: 20 },
                axisTick: { show: false }
            },
            yAxis: {
                type: 'value',
                splitLine: { lineStyle: { color: CHART_COLORS.gridLine } },
                axisLabel: { color: CHART_COLORS.textLight, fontSize: 10 }
            },
            series: [
                {
                    name: '活动前',
                    type: 'bar',
                    data: beforeData,
                    barWidth: 20,
                    itemStyle: {
                        color: 'rgba(102, 126, 234, 0.4)',
                        borderRadius: [4, 4, 0, 0]
                    },
                    animationDuration: 1500
                },
                {
                    name: '活动中',
                    type: 'bar',
                    data: duringData,
                    barWidth: 20,
                    itemStyle: {
                        color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                            { offset: 0, color: '#667eea' },
                            { offset: 1, color: '#764ba2' }
                        ]),
                        borderRadius: [4, 4, 0, 0]
                    },
                    animationDuration: 1500,
                    animationDelay: 300
                }
            ]
        };
        chart.setOption(option);
        window.addEventListener('resize', () => chart.resize());
    });
}

// ============================================================
// 9. Association Rules Table
// ============================================================
function loadAprioriStats() {
    fetchJSON('/api/apriori_stats').then(data => {
        const tbody = document.getElementById('apriori-tbody');
        if (!tbody) return;

        if (data.length === 0) {
            tbody.innerHTML = '<tr><td colspan="5" style="text-align:center;color:#6b6880;padding:40px;">暂无关联规则数据</td></tr>';
            return;
        }

        tbody.innerHTML = data.map((rule, i) => {
            const liftClass = rule.lift >= 3 ? 'rule-badge-high' : rule.lift >= 1.5 ? 'rule-badge-med' : 'rule-badge-low';
            return `
                <tr>
                    <td>${i + 1}</td>
                    <td>${rule.antecedents}</td>
                    <td>${rule.consequents}</td>
                    <td>${rule.support}</td>
                    <td>${rule.confidence}</td>
                    <td><span class="rule-badge ${liftClass}">${rule.lift}</span></td>
                </tr>
            `;
        }).join('');
    });
}
