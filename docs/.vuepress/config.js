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
            href: '/logo.png'
        }],
        ['meta', {
            name: 'keywords',
            content: 'Akasaki,Deep learning,Machine learning,工具箱,工具箱的深度学习记事簿,Akasaki的深度学习记事簿'
        }]
    ],
    themeConfig: {
        // 添加导航栏
        nav: [{
            text: 'GitHub',
            // 这里是下拉列表展现形式。
            link: 'https://github.com/VisualDust/ml.akasaki.space'
        }],
        // 为以下路由添加侧边栏
        sidebar: {
            '/': [
                {
                    title: '第零章：在开始之前',
                    children: getSidebar('ch0')
                },
                {
                    title: '第一章上：这HelloWorld有点长啊',
                    children: getSidebar('ch1p1')
                },
                {
                    title: '第一章下：深度学习基础——多层感知机',
                    children: getSidebar('ch1p2')
                },
                {
                    title: '第二章上：卷积神经网络及其要素',
                    children: getSidebar('ch2p1')
                },
                {
                    title: '第二章下：经典卷积神经网络',
                    children: getSidebar('ch2p2')
                },
                {
                    title: '第三章上：谈一些计算机视觉的方向',
                    children: getSidebar('ch3p1')
                },
                {
                    title: '第三章下：尝试一些计算机视觉任务',
                    children: getSidebar('ch3p2')
                },
                {
                    title: '附录',
                    children: getSidebar('appendix')
                },
                {
                    title: '第-1章：TensorFlow编程策略',
                    children: getSidebar('ch-1')
                },
                {
                    title: '第-2章：数字信号处理（DSP）',
                    children: getSidebar('ch-2')
                },
                {
                    title: '无尽模式',
                    children: getSidebar('unlimited-paper-works')
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
        ],
    ],
}