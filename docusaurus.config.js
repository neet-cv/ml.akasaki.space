// @ts-check
// Note: type annotations allow type checking and IDEs autocompletion

const lightCodeTheme = require("prism-react-renderer/themes/github");
const darkCodeTheme = require("prism-react-renderer/themes/dracula");

const math = require("remark-math");
const katex = require("rehype-katex");

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: "工具箱的深度学习记事簿",
  tagline: "Dinosaurs are cool",
  url: "https://ml.akasaki.space/",
  baseUrl: "/",
  onBrokenLinks: "warn",
  onBrokenMarkdownLinks: "warn",
  favicon: "img/logo.svg",
  organizationName: "neet-cv", // Usually your GitHub org/user name.
  projectName: "ml.akasaki.space", // Usually your repo name.
  i18n: {
    defaultLocale: "zh-cn",
    locales: ["zh-cn"],
  },

  presets: [
    [
      "@docusaurus/preset-classic",
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          sidebarPath: require.resolve("./sidebars.js"),
          // Please change this to your repo.
          // editUrl: 'https://github.com/facebook/docusaurus/edit/main/website/',
          // routeBasePath: '/'
          remarkPlugins: [math],
          rehypePlugins: [[katex, { strict: false }]],
          id: "docs",
          routeBasePath: "/",
        },
        blog: false,
        // pages: false,
        theme: {
          customCss: require.resolve("./src/css/custom.css"),
        },
      }),
    ],
  ],

  plugins: [
    [
      require.resolve("@easyops-cn/docusaurus-search-local"),
      {
        // ... Your options.
        // `hashed` is recommended as long-term-cache of index file is possible.
        hashed: true,
        // For Docs using Chinese, The `language` is recommended to set to:
        // ```
        language: ["en", "zh"],
        indexDocs:true,
        docsRouteBasePath:"/",
        docsDir:"docs",
        // ```
        // When applying `zh` in language, please install `nodejieba` in your project.
        translations:{
          "search_placeholder": "Search",
          "see_all_results": "See all results",
          "no_results": "No results.",
          "search_results_for": "Search results for \"{{ keyword }}\"",
          "search_the_documentation": "Search the documentation",
          "count_documents_found": "{{ count }} document found",
          "count_documents_found_plural": "{{ count }} documents found",
          "no_documents_were_found": "No documents were found"
        }
      },
    ],
  ],

  webpack: {
    jsLoader: (isServer) => ({
      loader: require.resolve("esbuild-loader"),
      options: {
        loader: "tsx",
        format: isServer ? "cjs" : undefined,
        target: isServer ? "node12" : "es2017",
      },
    }),
  },

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      hideableSidebar: true,
      navbar: {
        title: "工具箱的深度学习记事簿",
        logo: {
          alt: "Logo",
          src: "img/logo.svg",
        },
        items: [
          {
            label: "魔法部日志",
            to: "/unlimited-paper-works/",
            position: "left",
            activeBaseRegex: "/unlimited-paper-works/",
          },
          {
            href: "https://github.com/neet-cv/ml.akasaki.space",
            label: "GitHub",
            position: "left",
          },
          {
            label: "Authors & About",
            to: "/about",
            position: "right",
          },
          // {to: '/blog', label: 'Blog', position: 'left'},
          {
            href: "https://focus.akasaki.space",
            label: "Akasaki's blogs",
            position: "right",
          },
        ],
      },
      footer: {
        style: "dark",
        links: [
          // {
          //   title: 'Docs',
          //   items: [
          //     {
          //       label: 'Tutorial',
          //       to: '/docs/intro',
          //     },
          //   ],
          // },
          // {
          //   title: 'Community',
          //   items: [
          //     {
          //       label: 'Stack Overflow',
          //       href: 'https://stackoverflow.com/questions/tagged/docusaurus',
          //     },
          //     {
          //       label: 'Discord',
          //       href: 'https://discordapp.com/invite/docusaurus',
          //     },
          //     {
          //       label: 'Twitter',
          //       href: 'https://twitter.com/docusaurus',
          //     },
          //   ],
          // },
          // {
          //   title: 'More',
          //   items: [
          //     {
          //       label: 'Blog',
          //       to: '/blog',
          //     },
          //     {
          //       label: 'GitHub',
          //       href: 'https://github.com/facebook/docusaurus',
          //     },
          //   ],
          // },
        ],
        copyright: `Copyright © ${new Date().getFullYear()} neet-cv. Built with Docusaurus.`,
      },
      prism: {
        theme: lightCodeTheme,
        darkTheme: darkCodeTheme,
      },
    }),
  stylesheets: [
    {
      href: "https://cdn.jsdelivr.net/npm/katex@0.13.20/dist/katex.min.css",
      crossorigin: "anonymous",
    },
  ],
};

module.exports = config;
