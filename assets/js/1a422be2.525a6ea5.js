"use strict";(self.webpackChunkml_notebook=self.webpackChunkml_notebook||[]).push([[892],{3905:(e,t,a)=>{a.d(t,{Zo:()=>m,kt:()=>d});var n=a(67294);function o(e,t,a){return t in e?Object.defineProperty(e,t,{value:a,enumerable:!0,configurable:!0,writable:!0}):e[t]=a,e}function r(e,t){var a=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),a.push.apply(a,n)}return a}function l(e){for(var t=1;t<arguments.length;t++){var a=null!=arguments[t]?arguments[t]:{};t%2?r(Object(a),!0).forEach((function(t){o(e,t,a[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(a)):r(Object(a)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(a,t))}))}return e}function i(e,t){if(null==e)return{};var a,n,o=function(e,t){if(null==e)return{};var a,n,o={},r=Object.keys(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||(o[a]=e[a]);return o}(e,t);if(Object.getOwnPropertySymbols){var r=Object.getOwnPropertySymbols(e);for(n=0;n<r.length;n++)a=r[n],t.indexOf(a)>=0||Object.prototype.propertyIsEnumerable.call(e,a)&&(o[a]=e[a])}return o}var s=n.createContext({}),p=function(e){var t=n.useContext(s),a=t;return e&&(a="function"==typeof e?e(t):l(l({},t),e)),a},m=function(e){var t=p(e.components);return n.createElement(s.Provider,{value:t},e.children)},c="mdxType",u={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},h=n.forwardRef((function(e,t){var a=e.components,o=e.mdxType,r=e.originalType,s=e.parentName,m=i(e,["components","mdxType","originalType","parentName"]),c=p(a),h=o,d=c["".concat(s,".").concat(h)]||c[h]||u[h]||r;return a?n.createElement(d,l(l({ref:t},m),{},{components:a})):n.createElement(d,l({ref:t},m))}));function d(e,t){var a=arguments,o=t&&t.mdxType;if("string"==typeof e||o){var r=a.length,l=new Array(r);l[0]=h;var i={};for(var s in t)hasOwnProperty.call(t,s)&&(i[s]=t[s]);i.originalType=e,i[c]="string"==typeof e?e:o,l[1]=i;for(var p=2;p<r;p++)l[p]=a[p];return n.createElement.apply(null,l)}return n.createElement.apply(null,a)}h.displayName="MDXCreateElement"},15902:(e,t,a)=>{a.r(t),a.d(t,{assets:()=>k,contentTitle:()=>b,default:()=>y,frontMatter:()=>d,metadata:()=>g,toc:()=>v});var n=a(3905),o=Object.defineProperty,r=Object.defineProperties,l=Object.getOwnPropertyDescriptors,i=Object.getOwnPropertySymbols,s=Object.prototype.hasOwnProperty,p=Object.prototype.propertyIsEnumerable,m=(e,t,a)=>t in e?o(e,t,{enumerable:!0,configurable:!0,writable:!0,value:a}):e[t]=a,c=(e,t)=>{for(var a in t||(t={}))s.call(t,a)&&m(e,a,t[a]);if(i)for(var a of i(t))p.call(t,a)&&m(e,a,t[a]);return e},u=(e,t)=>r(e,l(t)),h=(e,t)=>{var a={};for(var n in e)s.call(e,n)&&t.indexOf(n)<0&&(a[n]=e[n]);if(null!=e&&i)for(var n of i(e))t.indexOf(n)<0&&p.call(e,n)&&(a[n]=e[n]);return a};const d={title:"DeepLab Series",authors:["visualdust"],tags:["segmentation","decoder","atrous-convolution","backbone"]},b=void 0,g={permalink:"/blog/[06]DeepLab-Series",editUrl:"https://github.dev/neet-cv/ml.akasaki.space/blob/master/blog/[06]DeepLab-Series.md",source:"@site/blog/[06]DeepLab-Series.md",title:"DeepLab Series",description:"DeepLab\u7cfb\u5217\u4e2d\u5305\u542b\u4e86\u4e09\u7bc7\u8bba\u6587\uff1aDeepLab-v1\u3001DeepLab-v2\u3001DeepLab-v3\u3002",date:"2023-12-31T09:31:53.000Z",formattedDate:"2023\u5e7412\u670831\u65e5",tags:[{label:"segmentation",permalink:"/blog/tags/segmentation"},{label:"decoder",permalink:"/blog/tags/decoder"},{label:"atrous-convolution",permalink:"/blog/tags/atrous-convolution"},{label:"backbone",permalink:"/blog/tags/backbone"}],readingTime:9.385,hasTruncateMarker:!0,authors:[{name:"Gavin Gong",title:"Rubbish CVer | Poor LaTex speaker | Half stack developer | \u952e\u5708\u8eba\u5c38\u7816\u5bb6",url:"https://gong.host",email:"gavin@gong.host",imageURL:"https://github.yuuza.net/visualDust.png",key:"visualdust"}],frontMatter:{title:"DeepLab Series",authors:["visualdust"],tags:["segmentation","decoder","atrous-convolution","backbone"]},prevItem:{title:"HLA-Face Joint High-Low Adaptation for Low Light Face Detection",permalink:"/blog/[05]HLA-Face-Joint-High-Low-Adaptation-for-Low-Light-Face-Detection"},nextItem:{title:"Cross-Dataset Collaborative Learning for Semantic Segmentation",permalink:"/blog/[07]Cross-Dataset-Collaborative-Learning-for-Semantic-Segmentation"}},k={authorsImageUrls:[void 0]},v=[{value:"DeepLab-v1",id:"deeplab-v1",level:2},{value:"\u7a7a\u6d1e\u5377\u79ef",id:"\u7a7a\u6d1e\u5377\u79ef",level:3},{value:"\u7a7a\u6d1e\u5377\u79ef\u7684\u4f18\u52bf",id:"\u7a7a\u6d1e\u5377\u79ef\u7684\u4f18\u52bf",level:4},{value:"\u7a7a\u6d1e\u5377\u79ef\u7684\u95ee\u9898",id:"\u7a7a\u6d1e\u5377\u79ef\u7684\u95ee\u9898",level:4},{value:"\u6df7\u5408\u81a8\u80c0\u5377\u79ef\uff08Hybrid Dilated Convolution, HDC\uff09",id:"\u6df7\u5408\u81a8\u80c0\u5377\u79efhybrid-dilated-convolution-hdc",level:4},{value:"\u6761\u4ef6\u968f\u673a\u573a",id:"\u6761\u4ef6\u968f\u673a\u573a",level:3},{value:"DeepLab-v2",id:"deeplab-v2",level:2},{value:"DeepLab-v3",id:"deeplab-v3",level:2}],f={toc:v},N="wrapper";function y(e){var t=e,{components:o}=t,r=h(t,["components"]);return(0,n.kt)(N,u(c(c({},f),r),{components:o,mdxType:"MDXLayout"}),(0,n.kt)("p",null,"DeepLab\u7cfb\u5217\u4e2d\u5305\u542b\u4e86\u4e09\u7bc7\u8bba\u6587\uff1aDeepLab-v1\u3001DeepLab-v2\u3001DeepLab-v3\u3002"),(0,n.kt)("p",null,"DeepLab-v1\uff1a",(0,n.kt)("a",c({parentName:"p"},{href:"https://arxiv.org/abs/1412.7062"}),"Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs")),(0,n.kt)("p",null,"DeepLab-v2\uff1a",(0,n.kt)("a",c({parentName:"p"},{href:"https://arxiv.org/abs/1606.00915"}),"Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs")),(0,n.kt)("p",null,"DeepLab-v3\uff1a",(0,n.kt)("a",c({parentName:"p"},{href:"https://arxiv.org/pdf/1706.05587.pdf"}),"Rethinking Atrous Convolution for Semantic Image Segmentation")),(0,n.kt)("p",null,"\u5728\u8fd9\u91cc\u6211\u4eec\u5c06\u8fd9\u4e09\u7bc7\u653e\u5728\u4e00\u8d77\u9605\u8bfb\u3002"),(0,n.kt)("p",null,"\u540e\u6765\u751a\u81f3\u8fd8\u51fa\u73b0\u4e86\u540e\u7eed\uff1a"),(0,n.kt)("p",null,"DeepLab-v3+\uff1a",(0,n.kt)("a",c({parentName:"p"},{href:"https://arxiv.org/abs/1802.02611"}),"Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation")),(0,n.kt)("p",null,"\u4e0d\u8fc7\u6682\u65f6\u6ca1\u6709\u5199\u8fdb\u6765\u7684\u6253\u7b97\u3002"),(0,n.kt)("h2",c({},{id:"deeplab-v1"}),"DeepLab-v1"),(0,n.kt)("p",null,"DeepLab-v1\u7684\u539f\u8bba\u6587\u662f",(0,n.kt)("a",c({parentName:"p"},{href:"https://arxiv.org/abs/1412.7062"}),"Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs"),"\u3002"),(0,n.kt)("blockquote",null,(0,n.kt)("p",{parentName:"blockquote"},"In this work we address the task of semantic image segmentation with Deep Learning and make three main contributions that are experimentally shown to have substantial practical merit. First, we highlight convolution with upsampled filters, or 'atrous convolution', as a powerful tool in dense prediction tasks. Atrous convolution allows us to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks. It also allows us to effectively enlarge the field of view of filters to incorporate larger context without increasing the number of parameters or the amount of computation. Second, we propose atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales. ASPP probes an incoming convolutional feature layer with filters at multiple sampling rates and effective fields-of-views, thus capturing objects as well as image context at multiple scales. Third, we improve the localization of object boundaries by combining methods from DCNNs and probabilistic graphical models. The commonly deployed combination of max-pooling and downsampling in DCNNs achieves invariance but has a toll on localization accuracy. We overcome this by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF), which is shown both qualitatively and quantitatively to improve localization performance. Our proposed \"DeepLab\" system sets the new state-of-art at the PASCAL VOC-2012 semantic image segmentation task, reaching 79.7% mIOU in the test set, and advances the results on three other datasets: PASCAL-Context, PASCAL-Person-Part, and Cityscapes. All of our code is made publicly available online.")),(0,n.kt)("p",null,"\u5728\u4e4b\u524d\u7684\u8bed\u4e49\u5206\u5272\u7f51\u7edc\u4e2d\uff0c\u5206\u5272\u7ed3\u679c\u5f80\u5f80\u6bd4\u8f83\u7c97\u7cd9\uff0c\u539f\u56e0\u4e3b\u8981\u6709\u4e24\u4e2a\uff0c\u4e00\u662f\u56e0\u4e3a\u6c60\u5316\u5bfc\u81f4\u7a7a\u95f4\u4fe1\u606f\u4e22\u5931\uff0c\u4e8c\u662f\u6ca1\u6709\u5229\u7528\u4e34\u8fd1\u50cf\u7d20\u70b9\u7c7b\u522b\u4e4b\u95f4\u7684\u6982\u7387\u5173\u7cfb\uff0c\u9488\u5bf9\u8fd9\u4e24\u70b9\uff0c\u4f5c\u8005\u63d0\u51fa\u4e86\u9488\u5bf9\u6027\u7684\u6539\u8fdb\u3002\u9996\u5148\u4f7f\u7528",(0,n.kt)("strong",{parentName:"p"},"\u7a7a\u6d1e\u5377\u79ef\uff08Atrous Convolution\uff09"),"\uff0c\u907f\u514d\u6c60\u5316\u5e26\u6765\u7684\u4fe1\u606f\u635f\u5931\uff0c\u7136\u540e\u4f7f\u7528",(0,n.kt)("strong",{parentName:"p"},"\u6761\u4ef6\u968f\u673a\u573a\uff08CRF\uff09"),"\uff0c\u8fdb\u4e00\u6b65\u4f18\u5316\u5206\u5272\u7cbe\u5ea6\u3002\u9605\u8bfb\u8fd9\u7bc7\u8bba\u6587\u5e94\u5173\u6ce8\u7684\u91cd\u70b9\u95ee\u9898\u5c31\u662f\u7a7a\u6d1e\u5377\u79ef\u548c\u6761\u4ef6\u968f\u673a\u573a\u3002"),(0,n.kt)("h3",c({},{id:"\u7a7a\u6d1e\u5377\u79ef"}),"\u7a7a\u6d1e\u5377\u79ef"),(0,n.kt)("p",null,"\u7a7a\u6d1e\u5377\u79ef\uff08Dilated/Atrous Convolution\u6216\u662fConvolution with holes \uff09\u7684\u4e3b\u8981\u4f5c\u7528\u662f\u5728\u589e\u5927\u611f\u53d7\u91ce\u7684\u540c\u65f6\uff0c\u4e0d\u589e\u52a0\u53c2\u6570\u6570\u91cf\uff0c\u800c\u4e14VGG\u4e2d\u63d0\u51fa\u7684\u591a\u4e2a\u5c0f\u5377\u79ef\u6838\u4ee3\u66ff\u5927\u5377\u79ef\u6838\u7684\u65b9\u6cd5\uff0c\u53ea\u80fd\u4f7f\u611f\u53d7\u91ce\u7ebf\u6027\u589e\u957f\uff0c\u800c\u591a\u4e2a\u7a7a\u6d1e\u5377\u79ef\u4e32\u8054\uff0c\u53ef\u4ee5\u5b9e\u73b0\u6307\u6570\u589e\u957f\u3002"),(0,n.kt)("h4",c({},{id:"\u7a7a\u6d1e\u5377\u79ef\u7684\u4f18\u52bf"}),"\u7a7a\u6d1e\u5377\u79ef\u7684\u4f18\u52bf"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},"\u8fd9\u79cd\u7ed3\u6784\u4ee3\u66ff\u4e86\u6c60\u5316\uff0c\u5b83\u53ef\u4ee5\u4fdd\u6301\u50cf\u7d20\u7a7a\u95f4\u4fe1\u606f\u3002"),(0,n.kt)("li",{parentName:"ul"},"\u5b83\u7531\u4e8e\u53ef\u4ee5\u6269\u5927\u611f\u53d7\u91ce\u56e0\u800c\u53ef\u4ee5\u5f88\u597d\u5730\u6574\u5408\u4e0a\u4e0b\u6587\u4fe1\u606f\u3002")),(0,n.kt)("p",null,"Convolution with holes \u5b57\u5982\u5176\u540d\uff0c\u662f\u5728\u6807\u51c6\u7684\u5377\u79ef\u6838\u4e2d\u6ce8\u5165\u7a7a\u6d1e\uff0c\u4ee5\u6b64\u6765\u589e\u52a0\u611f\u53d7\u91ce\u3002\u76f8\u6bd4\u4e8e\u666e\u901a\u7684\u5377\u79ef\uff0c\u7a7a\u6d1e\u5377\u79ef\u591a\u4e86\u4e00\u4e2a\u8d85\u53c2\u6570\u79f0\u4e4b\u4e3a\u7a7a\u6d1e\u7387\uff08dilation rate\uff09\u6307\u7684\u662fkernel\u7684\u95f4\u9694\u7684\u50cf\u7d20\u6570\u91cf\u3002"),(0,n.kt)("p",null,(0,n.kt)("img",{alt:"Atrous_conv",src:a(50251).Z,width:"1423",height:"675"})),(0,n.kt)("p",null,"\u4e0a\u56fe\u662f\u4e00\u5f20\u7a7a\u6d1e\u5377\u79ef\u7684\u793a\u610f\u56fe\u3002\u5728\u4e0a\u56fe\u4e2d\uff0c\u4e09\u4e2a\u7a7a\u6d1e\u5377\u79ef\u7684\u5927\u5c0f\u90fd\u662f",(0,n.kt)("span",c({parentName:"p"},{className:"math math-inline"}),(0,n.kt)("span",c({parentName:"span"},{className:"katex"}),(0,n.kt)("span",c({parentName:"span"},{className:"katex-mathml"}),(0,n.kt)("math",c({parentName:"span"},{xmlns:"http://www.w3.org/1998/Math/MathML"}),(0,n.kt)("semantics",{parentName:"math"},(0,n.kt)("mrow",{parentName:"semantics"},(0,n.kt)("mn",{parentName:"mrow"},"3"),(0,n.kt)("mo",{parentName:"mrow"},"\xd7"),(0,n.kt)("mn",{parentName:"mrow"},"3")),(0,n.kt)("annotation",c({parentName:"semantics"},{encoding:"application/x-tex"}),"3\\times 3")))),(0,n.kt)("span",c({parentName:"span"},{className:"katex-html","aria-hidden":"true"}),(0,n.kt)("span",c({parentName:"span"},{className:"base"}),(0,n.kt)("span",c({parentName:"span"},{className:"strut",style:{height:"0.7278em",verticalAlign:"-0.0833em"}})),(0,n.kt)("span",c({parentName:"span"},{className:"mord"}),"3"),(0,n.kt)("span",c({parentName:"span"},{className:"mspace",style:{marginRight:"0.2222em"}})),(0,n.kt)("span",c({parentName:"span"},{className:"mbin"}),"\xd7"),(0,n.kt)("span",c({parentName:"span"},{className:"mspace",style:{marginRight:"0.2222em"}}))),(0,n.kt)("span",c({parentName:"span"},{className:"base"}),(0,n.kt)("span",c({parentName:"span"},{className:"strut",style:{height:"0.6444em"}})),(0,n.kt)("span",c({parentName:"span"},{className:"mord"}),"3"))))),"\uff0c\u800c\u5b83\u4eec\u7684\u7a7a\u6d1e\u7387\u5206\u522b\u662f1\u30016\u548c24\uff0c\u6240\u4ee5\u80fd\u7528\u76f8\u540c\u5927\u5c0f\u7684\u5377\u79ef\u6838\u5f97\u5230\u4e0d\u540c\u7684\u611f\u53d7\u91ce\u3002"),(0,n.kt)("h4",c({},{id:"\u7a7a\u6d1e\u5377\u79ef\u7684\u95ee\u9898"}),"\u7a7a\u6d1e\u5377\u79ef\u7684\u95ee\u9898"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("p",{parentName:"li"},"\u7f51\u683c\u6548\u5e94\uff08The Gridding Effect\uff09"),(0,n.kt)("p",{parentName:"li"},"\u7a7a\u6d1e\u5377\u79ef\u5c42\u5e76\u4e0d\u80fd\u968f\u610f\u8bbe\u8ba1\uff0c\u4f8b\u5982\uff0c\u6211\u4eec\u7b80\u5355\u5730\u5806\u53e0\u7a7a\u6d1e\u7387\u4e3a2\u7684",(0,n.kt)("span",c({parentName:"p"},{className:"math math-inline"}),(0,n.kt)("span",c({parentName:"span"},{className:"katex"}),(0,n.kt)("span",c({parentName:"span"},{className:"katex-mathml"}),(0,n.kt)("math",c({parentName:"span"},{xmlns:"http://www.w3.org/1998/Math/MathML"}),(0,n.kt)("semantics",{parentName:"math"},(0,n.kt)("mrow",{parentName:"semantics"},(0,n.kt)("mn",{parentName:"mrow"},"3"),(0,n.kt)("mo",{parentName:"mrow"},"\xd7"),(0,n.kt)("mn",{parentName:"mrow"},"3")),(0,n.kt)("annotation",c({parentName:"semantics"},{encoding:"application/x-tex"}),"3\\times 3")))),(0,n.kt)("span",c({parentName:"span"},{className:"katex-html","aria-hidden":"true"}),(0,n.kt)("span",c({parentName:"span"},{className:"base"}),(0,n.kt)("span",c({parentName:"span"},{className:"strut",style:{height:"0.7278em",verticalAlign:"-0.0833em"}})),(0,n.kt)("span",c({parentName:"span"},{className:"mord"}),"3"),(0,n.kt)("span",c({parentName:"span"},{className:"mspace",style:{marginRight:"0.2222em"}})),(0,n.kt)("span",c({parentName:"span"},{className:"mbin"}),"\xd7"),(0,n.kt)("span",c({parentName:"span"},{className:"mspace",style:{marginRight:"0.2222em"}}))),(0,n.kt)("span",c({parentName:"span"},{className:"base"}),(0,n.kt)("span",c({parentName:"span"},{className:"strut",style:{height:"0.6444em"}})),(0,n.kt)("span",c({parentName:"span"},{className:"mord"}),"3"))))),"\u7684\u7a7a\u6d1e\u5377\u79ef\u6838\uff0c\u90a3\u4e48\u8fde\u7eed\u4e09\u5c42\u5377\u79ef\u6838\u5728\u539f\u56fe\u4e0a\u7684\u540c\u4e2a\u50cf\u7d20\u4f4d\u7f6e\u6240\u5bf9\u5e94\u7684\u611f\u53d7\u91ce\u5982\u4e0b\u56fe\u6240\u793a\uff1a"),(0,n.kt)("p",{parentName:"li"},(0,n.kt)("img",{alt:"image-20210514145720970",src:a(77253).Z,width:"1157",height:"374"})),(0,n.kt)("p",{parentName:"li"},"\u5f88\u660e\u663e\uff0c\u6807\u5706\u5708\u7684\u4f4d\u7f6e\u4e00\u76f4\u6ca1\u6709\u53c2\u4e0e\u8be5\u4f4d\u7f6e\u7684\u5377\u79ef\u8fd0\u7b97\u3002\u4e5f\u5c31\u662f\u5e76\u4e0d\u662f\u6240\u6709\u7684\u50cf\u7d20\u90fd\u7528\u6765\u8ba1\u7b97\u4e86\uff0c\u8fd9\u4f1a\u5bfc\u81f4\u4fe1\u606f\u7684\u8fde\u7eed\u6027\u635f\u5931\u3002\u8fd9\u5bf9\u5bc6\u96c6\u9884\u6d4b\uff08\u9010\u50cf\u7d20\uff09\u7684\u89c6\u89c9\u4efb\u52a1\u6765\u8bf4\u662f\u81f4\u547d\u7684\u3002")),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("p",{parentName:"li"},"\u76f8\u5173\u6027\u4e22\u5931"),(0,n.kt)("p",{parentName:"li"},"\u539f\u8bba\u6587\u4e2d\u63cf\u8ff0\u95ee\u9898\u7684\u8bdd\u662f\uff1a"),(0,n.kt)("blockquote",{parentName:"li"},(0,n.kt)("p",{parentName:"blockquote"},"Long-ranged information might be not relevant.")),(0,n.kt)("p",{parentName:"li"},"\u4e5f\u5c31\u662f\u8bf4\uff0c\u6211\u4eec\u4ece dilated convolution \u7684\u8bbe\u8ba1\u80cc\u666f\u6765\u770b\u5c31\u80fd\u63a8\u6d4b\u51fa\u8fd9\u6837\u7684\u8bbe\u8ba1\u662f\u7528\u6765\u83b7\u53d6 long-ranged information\u3002\u7136\u800c\u4ec5\u91c7\u7528\u5927 dilation rate \u7684\u4fe1\u606f\u6216\u8bb8\u53ea\u5bf9\u4e00\u4e9b\u5927\u7269\u4f53\u5206\u5272\u6709\u6548\u679c\uff0c\u800c\u5bf9\u5c0f\u7269\u4f53\u6765\u8bf4\u53ef\u80fd\u5219\u6709\u5f0a\u65e0\u5229\u4e86\u3002\u5982\u4f55\u540c\u65f6\u5904\u7406\u4e0d\u540c\u5927\u5c0f\u7684\u7269\u4f53\u7684\u5173\u7cfb\uff0c\u5219\u662f\u8bbe\u8ba1\u597d dilated convolution \u7f51\u7edc\u7684\u5173\u952e\u3002"))),(0,n.kt)("h4",c({},{id:"\u6df7\u5408\u81a8\u80c0\u5377\u79efhybrid-dilated-convolution-hdc"}),"\u6df7\u5408\u81a8\u80c0\u5377\u79ef\uff08Hybrid Dilated Convolution, HDC\uff09"),(0,n.kt)("p",null,"\u5bf9\u4e8e\u521a\u624d\u63d0\u5230\u7684\u7a7a\u6d1e\u5377\u79ef\u7684\u95ee\u9898\uff0c\u8bba\u6587\u4e2d\u63d0\u51fa\u4e86\u4e00\u79cd\u79f0\u4e3aHDC\u7684\u7ed3\u6784\u4f5c\u4e3a\u89e3\u51b3\u65b9\u6848\u3002\u8fd9\u4e2a\u65b9\u6848\u5177\u6709\u4ee5\u4e0b\u7279\u6027\uff1a"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},"\u5bf9\u4e8e\u6bcf\u5c42\u7a7a\u6d1e\u5377\u79ef\uff0c\u5176\u6700\u5927\u7a7a\u6d1e\u5377\u79ef\u7387\u7684\u6700\u5c0f\u516c\u56e0\u5b50\u4e0d\u80fd\u4e3a1\u3002"),(0,n.kt)("li",{parentName:"ul"})),(0,n.kt)("h3",c({},{id:"\u6761\u4ef6\u968f\u673a\u573a"}),"\u6761\u4ef6\u968f\u673a\u573a"),(0,n.kt)("p",null,"\u6761\u4ef6\u968f\u673a\u573a\uff0c\u7b80\u5355\u6765\u8bb2\u5c31\u662f\u6bcf\u4e2a\u50cf\u7d20\u70b9\u4f5c\u4e3a\u8282\u70b9\uff0c\u50cf\u7d20\u4e0e\u50cf\u7d20\u95f4\u7684\u5173\u7cfb\u4f5c\u4e3a\u8fb9\uff0c\u5373\u6784\u6210\u4e86\u4e00\u4e2a\u6761\u4ef6\u968f\u673a\u573a\u3002\u901a\u8fc7\u4e8c\u5143\u52bf\u51fd\u6570\u63cf\u8ff0\u50cf\u7d20\u70b9\u4e0e\u50cf\u7d20\u70b9\u4e4b\u95f4\u7684\u5173\u7cfb\uff0c\u9f13\u52b1\u76f8\u4f3c\u50cf\u7d20\u5206\u914d\u76f8\u540c\u7684\u6807\u7b7e\uff0c\u800c\u76f8\u5dee\u8f83\u5927\u7684\u50cf\u7d20\u5206\u914d\u4e0d\u540c\u6807\u7b7e\uff0c\u800c\u8fd9\u4e2a\u201c\u8ddd\u79bb\u201d\u7684\u5b9a\u4e49\u4e0e\u989c\u8272\u503c\u548c\u5b9e\u9645\u76f8\u5bf9\u8ddd\u79bb\u6709\u5173\u3002\u6240\u4ee5\u8fd9\u6837CRF\u80fd\u591f\u4f7f\u56fe\u7247\u5728\u5206\u5272\u7684\u8fb9\u754c\u51fa\u53d6\u5f97\u6bd4\u8f83\u597d\u7684\u6548\u679c\u3002"),(0,n.kt)("h2",c({},{id:"deeplab-v2"}),"DeepLab-v2"),(0,n.kt)("p",null,"DeepLab-v2\u7684\u539f\u8bba\u6587\u662f",(0,n.kt)("a",c({parentName:"p"},{href:"https://arxiv.org/abs/1606.00915"}),"Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs"),"\u3002"),(0,n.kt)("blockquote",null,(0,n.kt)("p",{parentName:"blockquote"},'Deep Convolutional Neural Networks (DCNNs) have recently shown state of the art performance in high level vision tasks, such as image classification and object detection. This work brings together methods from DCNNs and probabilistic graphical models for addressing the task of pixel-level classification (also called "semantic image segmentation"). We show that responses at the final layer of DCNNs are not sufficiently localized for accurate object segmentation. This is due to the very invariance properties that make DCNNs good for high level tasks. We overcome this poor localization property of deep networks by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF). Qualitatively, our "DeepLab" system is able to localize segment boundaries at a level of accuracy which is beyond previous methods. Quantitatively, our method sets the new state-of-art at the PASCAL VOC-2012 semantic image segmentation task, reaching 71.6% IOU accuracy in the test set. We show how these results can be obtained efficiently: Careful network re-purposing and a novel application of the \'hole\' algorithm from the wavelet community allow dense computation of neural net responses at 8 frames per second on a modern GPU.')),(0,n.kt)("p",null,"DeepLab-v2\u5bf9DeepLab-v1\u7684\u6539\u8fdb\u662f\uff1a"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},"\u4f7f\u7528\u4e86\u91d1\u5b57\u5854\u591a\u5c3a\u5ea6\u7279\u5f81\u83b7\u5f97\u66f4\u597d\u7684\u5206\u5272\u6548\u679c\u3002"),(0,n.kt)("li",{parentName:"ul"},"\u5c06\u9aa8\u5e72\u7f51\u7edc\u7531VGG\u66ff\u6362\u4e3a\u4e86ResNet\u3002"),(0,n.kt)("li",{parentName:"ul"},"\u7a0d\u5fae\u4fee\u6539\u4e86learning-rate\u3002")),(0,n.kt)("p",null,"\u5176\u4e2dASPP\u7684\u5f15\u5165\u662f\u6700\u5927\u4e5f\u662f\u6700\u91cd\u8981\u7684\u6539\u53d8\u3002\u591a\u5c3a\u5ea6\u4e3b\u8981\u662f\u4e3a\u4e86\u89e3\u51b3\u76ee\u6807\u5728\u56fe\u50cf\u4e2d\u8868\u73b0\u4e3a\u4e0d\u540c\u5927\u5c0f\u65f6\u4ecd\u80fd\u591f\u6709\u5f88\u597d\u7684\u5206\u5272\u7ed3\u679c\uff0c\u6bd4\u5982\u540c\u6837\u7684\u7269\u4f53\uff0c\u5728\u8fd1\u5904\u62cd\u6444\u65f6\u7269\u4f53\u663e\u5f97\u5927\uff0c\u8fdc\u5904\u62cd\u6444\u65f6\u663e\u5f97\u5c0f\u3002\u5177\u4f53\u505a\u6cd5\u662f\u5e76\u884c\u7684\u91c7\u7528\u591a\u4e2a\u91c7\u6837\u7387\u7684\u7a7a\u6d1e\u5377\u79ef\u63d0\u53d6\u7279\u5f81\uff0c\u518d\u5c06\u7279\u5f81\u878d\u5408\uff0c\u7c7b\u4f3c\u4e8e\u7a7a\u95f4\u91d1\u5b57\u5854\u7ed3\u6784\uff0c\u5f62\u8c61\u7684\u79f0\u4e3aAtrous Spatial Pyramid Pooling (ASPP)\u3002"),(0,n.kt)("h2",c({},{id:"deeplab-v3"}),"DeepLab-v3"),(0,n.kt)("p",null,"DeepLab-v3\u7684\u539f\u8bba\u6587\u662f",(0,n.kt)("a",c({parentName:"p"},{href:"https://arxiv.org/abs/1706.05587"}),"Rethinking Atrous Convolution for Semantic Image Segmentation"),"\u3002"),(0,n.kt)("blockquote",null,(0,n.kt)("p",{parentName:"blockquote"},"In this work, we revisit atrous convolution, a powerful tool to explicitly adjust filter's field-of-view as well as control the resolution of feature responses computed by Deep Convolutional Neural Networks, in the application of semantic image segmentation. To handle the problem of segmenting objects at multiple scales, we design modules which employ atrous convolution in cascade or in parallel to capture multi-scale context by adopting multiple atrous rates. Furthermore, we propose to augment our previously proposed Atrous Spatial Pyramid Pooling module, which probes convolutional features at multiple scales, with image-level features encoding global context and further boost performance. We also elaborate on implementation details and share our experience on training our system. The proposed `DeepLabv3' system significantly improves over our previous DeepLab versions without DenseCRF post-processing and attains comparable performance with other state-of-art models on the PASCAL VOC 2012 semantic image segmentation benchmark.")),(0,n.kt)("p",null,"DeepLab-v3\u7684\u6539\u8fdb\u662f\uff1a"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},"\u63d0\u51fa\u4e86\u66f4\u901a\u7528\u7684\u6846\u67b6\uff0c\u9002\u7528\u4e8e\u4efb\u4f55\u7f51\u7edc\u3002"),(0,n.kt)("li",{parentName:"ul"},"\u5c06ResNet\u6700\u540e\u7684\u4e00\u4e9b\u6a21\u5757\u66ff\u6362\u4e3a\u4f7f\u7528\u7a7a\u6d1e\u5377\u79ef\u8fdb\u884c\u7684\u7ea7\u8054\u3002"),(0,n.kt)("li",{parentName:"ul"},"\u5728ASPP\u4e2d\u4f7f\u7528\u4e86Batch Normolization\u5c42\u3002"),(0,n.kt)("li",{parentName:"ul"},"\u53bb\u9664\u4e86\u6761\u4ef6\u968f\u673a\u573a\u3002")))}y.isMDXComponent=!0},50251:(e,t,a)=>{a.d(t,{Z:()=>n});const n=a.p+"assets/images/Atrous_conv-8bbc9b569825f67962e113f732b01b61.png"},77253:(e,t,a)=>{a.d(t,{Z:()=>n});const n=a.p+"assets/images/image-20210514145720970-e129c46568b26824bc4946847ff0323c.png"}}]);