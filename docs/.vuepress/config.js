const { getSidebar } = require('../../js/utils');

module.exports = {
    markdown: {
        lineNumbers: false
    },
    title: '工具箱的深度学习记事簿',
    description: '这里包含了我从入门到依然在入门的过程中接触到的大部分知识。翻翻目录，也许能找到有用的',
    head: [
        ['link', {
            rel: 'icon',
            href: '/statics/logo.svg'
        }],
        ['meta', {
            name: 'keywords',
            content: 'Akasaki,Deep learning,Machine learning,工具箱,工具箱的深度学习记事簿,Akasaki的深度学习记事簿'
        }],
        ['meta', {
            name: 'google-site-verification',
            content: 'VVNYs0bXM_EKTgxJ8XIfvXShjHsksGNv3YNedxBGFjU'
        }]
    ],
    themeConfig: {
        lastUpdated: '上次更新时间', // 上次更新
        smoothScroll: true, // 页面滚动
        sidebarDepth: 6,
        // 添加导航栏
        nav: [{
            text: 'View on Github',
            // 这里是下拉列表展现形式。
            link: 'https://github.com/VisualDust/ml.akasaki.space'
        },{
            text: '工具箱',
            // 这里是下拉列表展现形式。
            link: 'https://github.com/VisualDust'
        }],
        // 为以下路由添加侧边栏
        sidebar: {
            '/': [
                {
                    title: '第零章：在开始之前',
                    children: getSidebar('ch0'),
                },
                {
                    title: '第一章上：HelloWorld',
                    children: getSidebar('ch1p1'),
                },
                {
                    title: '第一章下：深度学习基础',
                    children: getSidebar('ch1p2'),
                },
                {
                    title: '第二章上：卷积神经网络',
                    children: getSidebar('ch2p1'),
                },
                {
                    title: '第二章下：经典卷积神经网络',
                    children: getSidebar('ch2p2'),
                },
                {
                    title: '第三章上：谈一些计算机视觉方向',
                    children: getSidebar('ch3p1'),
                },
                {
                    title: '第三章下：了解更高级的技术',
                    children: getSidebar('ch3p2'),
                },
                {
                    title: '附录：永远是你的好朋友',
                    children: getSidebar('appendix'),
                },{
                    title: '第五章：Playground',
                    children: getSidebar('ch5'),
                },
                {
                    title: '第-1章：TensorFlow编程策略',
                    children: getSidebar('ch-1'),
                },
                {
                    title: '第-2章：数字信号处理（DSP）',
                    children: getSidebar('ch-2'),
                },
                {
                    title: '魔法部日志（又名论文阅读日志）',
                    children: getSidebar('unlimited-paper-works'),
                },
            ]
        }
    },
    plugins: [
        [
            "md-enhance",
            {
                // 启用 TeX 支持
                tex: true,
            },
            "vuepress-plugin-element-tabs",
        ],
    ],
    locales: {
        '/': {
            lang: 'zh-CN'
        }
    }
}
