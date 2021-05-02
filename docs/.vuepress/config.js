const { getSidebar } = require('../../js/index');

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
        }]
    ],
    themeConfig: {
        // 添加导航栏
        nav: [{
                text: 'GitHub',
                // 这里是下拉列表展现形式。
                link: 'https://github.com/VisualDust/ml.akasaki.space'
            }
        ],
        // 为以下路由添加侧边栏
        sidebar: {
            // '/database/': getSidebar('database'),
            // '/algorithm/': getSidebar('algorithm'),
            '/ch0/':getSidebar('ch0'),
            '/ch1p1/':getSidebar('ch1p1'),
            '/ch1p2/':getSidebar('ch1p2'),
            '/ch2p1/':getSidebar('ch2p1'),
            '/ch2p2/':getSidebar('ch2p2'),
            '/ch3p1/':getSidebar('ch3p1'),
            '/ch3p2/':getSidebar('ch3p2'),
            '/ch4/':getSidebar('ch4'),
            '/appendix/':getSidebar('appendix'),
            '/ch-1/':getSidebar('ch-1'),
            '/ch-2/':getSidebar('ch-2')
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