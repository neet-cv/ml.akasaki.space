module.exports = {
    markdown: {
        lineNumbers: false,
        extendMarkdown: md => {
            md.use(require('markdown-it-footnote'))
        }
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
            '/': require('./theindex').getSidebarIndex()
        }
    },
    plugins: [
        [
            "vuepress-plugin-element-tabs",
            "md-enhance",
            {
                // 启用 TeX 支持
                tex: true,
            },
        ],
    ],
    locales: {
        '/': {
            lang: 'zh-CN'
        }
    }
}