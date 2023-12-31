"use strict";(self.webpackChunkml_notebook=self.webpackChunkml_notebook||[]).push([[5730],{3905:(e,a,t)=>{t.d(a,{Zo:()=>c,kt:()=>k});var n=t(67294);function r(e,a,t){return a in e?Object.defineProperty(e,a,{value:t,enumerable:!0,configurable:!0,writable:!0}):e[a]=t,e}function s(e,a){var t=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);a&&(n=n.filter((function(a){return Object.getOwnPropertyDescriptor(e,a).enumerable}))),t.push.apply(t,n)}return t}function p(e){for(var a=1;a<arguments.length;a++){var t=null!=arguments[a]?arguments[a]:{};a%2?s(Object(t),!0).forEach((function(a){r(e,a,t[a])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(t)):s(Object(t)).forEach((function(a){Object.defineProperty(e,a,Object.getOwnPropertyDescriptor(t,a))}))}return e}function l(e,a){if(null==e)return{};var t,n,r=function(e,a){if(null==e)return{};var t,n,r={},s=Object.keys(e);for(n=0;n<s.length;n++)t=s[n],a.indexOf(t)>=0||(r[t]=e[t]);return r}(e,a);if(Object.getOwnPropertySymbols){var s=Object.getOwnPropertySymbols(e);for(n=0;n<s.length;n++)t=s[n],a.indexOf(t)>=0||Object.prototype.propertyIsEnumerable.call(e,t)&&(r[t]=e[t])}return r}var m=n.createContext({}),o=function(e){var a=n.useContext(m),t=a;return e&&(t="function"==typeof e?e(a):p(p({},a),e)),t},c=function(e){var a=o(e.components);return n.createElement(m.Provider,{value:a},e.children)},i="mdxType",u={inlineCode:"code",wrapper:function(e){var a=e.children;return n.createElement(n.Fragment,{},a)}},d=n.forwardRef((function(e,a){var t=e.components,r=e.mdxType,s=e.originalType,m=e.parentName,c=l(e,["components","mdxType","originalType","parentName"]),i=o(t),d=r,k=i["".concat(m,".").concat(d)]||i[d]||u[d]||s;return t?n.createElement(k,p(p({ref:a},c),{},{components:t})):n.createElement(k,p({ref:a},c))}));function k(e,a){var t=arguments,r=a&&a.mdxType;if("string"==typeof e||r){var s=t.length,p=new Array(s);p[0]=d;var l={};for(var m in a)hasOwnProperty.call(a,m)&&(l[m]=a[m]);l.originalType=e,l[i]="string"==typeof e?e:r,p[1]=l;for(var o=2;o<s;o++)p[o]=t[o];return n.createElement.apply(null,p)}return n.createElement.apply(null,t)}d.displayName="MDXCreateElement"},59315:(e,a,t)=>{t.r(a),t.d(a,{assets:()=>N,contentTitle:()=>f,default:()=>g,frontMatter:()=>k,metadata:()=>h,toc:()=>y});var n=t(3905),r=Object.defineProperty,s=Object.defineProperties,p=Object.getOwnPropertyDescriptors,l=Object.getOwnPropertySymbols,m=Object.prototype.hasOwnProperty,o=Object.prototype.propertyIsEnumerable,c=(e,a,t)=>a in e?r(e,a,{enumerable:!0,configurable:!0,writable:!0,value:t}):e[a]=t,i=(e,a)=>{for(var t in a||(a={}))m.call(a,t)&&c(e,t,a[t]);if(l)for(var t of l(a))o.call(a,t)&&c(e,t,a[t]);return e},u=(e,a)=>s(e,p(a)),d=(e,a)=>{var t={};for(var n in e)m.call(e,n)&&a.indexOf(n)<0&&(t[n]=e[n]);if(null!=e&&l)for(var n of l(e))a.indexOf(n)<0&&o.call(e,n)&&(t[n]=e[n]);return t};const k={},f="\u591a\u5c42\u611f\u77e5\u673a\u7684\u4ee3\u7801\u5b9e\u73b0",h={unversionedId:"ch1p2/[2]multilayer-perceptron-code",id:"ch1p2/[2]multilayer-perceptron-code",title:"\u591a\u5c42\u611f\u77e5\u673a\u7684\u4ee3\u7801\u5b9e\u73b0",description:"\u6211\u4eec\u5df2\u7ecf\u4ece\u4e0a\u4e00\u8282\u91cc\u4e86\u89e3\u4e86\u591a\u5c42\u611f\u77e5\u673a\u7684\u539f\u7406\u3002\u4e0b\u9762\uff0c\u6211\u4eec\u4e00\u8d77\u6765\u52a8\u624b\u5b9e\u73b0\u4e00\u4e2a\u591a\u5c42\u611f\u77e5\u673a\u3002\u9996\u5148\u5bfc\u5165\u5b9e\u73b0\u6240\u9700\u7684\u5305\u6216\u6a21\u5757\u3002",source:"@site/docs/ch1p2/[2]multilayer-perceptron-code.mdx",sourceDirName:"ch1p2",slug:"/ch1p2/[2]multilayer-perceptron-code",permalink:"/ch1p2/[2]multilayer-perceptron-code",draft:!1,tags:[],version:"current",frontMatter:{},sidebar:"docsSidebar",previous:{title:"\u591a\u5c42\u611f\u77e5\u673a",permalink:"/ch1p2/[1]multilayer-perceptron"},next:{title:"\u6fc0\u6d3b\u51fd\u6570",permalink:"/ch1p2/[3]activation-functions"}},N={},y=[{value:"\u624b\u52a8\u5b9e\u73b0",id:"\u624b\u52a8\u5b9e\u73b0",level:2},{value:"\u83b7\u53d6\u548c\u8bfb\u53d6\u6570\u636e",id:"\u83b7\u53d6\u548c\u8bfb\u53d6\u6570\u636e",level:3},{value:"\u5b9a\u4e49\u6a21\u578b\u53c2\u6570",id:"\u5b9a\u4e49\u6a21\u578b\u53c2\u6570",level:3},{value:"\u5b9a\u4e49\u6fc0\u6d3b\u51fd\u6570",id:"\u5b9a\u4e49\u6fc0\u6d3b\u51fd\u6570",level:3},{value:"\u5b9a\u4e49\u6a21\u578b",id:"\u5b9a\u4e49\u6a21\u578b",level:3},{value:"\u5b9a\u4e49\u635f\u5931\u51fd\u6570",id:"\u5b9a\u4e49\u635f\u5931\u51fd\u6570",level:3},{value:"\u8bad\u7ec3\u6a21\u578b",id:"\u8bad\u7ec3\u6a21\u578b",level:3},{value:"\u4f7f\u7528\u6846\u67b6\u5b9e\u73b0",id:"\u4f7f\u7528\u6846\u67b6\u5b9e\u73b0",level:2},{value:"\u5b9a\u4e49\u6a21\u578b",id:"\u5b9a\u4e49\u6a21\u578b-1",level:3},{value:"\u8bfb\u53d6\u6570\u636e\u5e76\u8bad\u7ec3\u6a21\u578b",id:"\u8bfb\u53d6\u6570\u636e\u5e76\u8bad\u7ec3\u6a21\u578b",level:3}],_={toc:y},v="wrapper";function g(e){var a=e,{components:t}=a,r=d(a,["components"]);return(0,n.kt)(v,u(i(i({},_),r),{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h1",i({},{id:"\u591a\u5c42\u611f\u77e5\u673a\u7684\u4ee3\u7801\u5b9e\u73b0"}),"\u591a\u5c42\u611f\u77e5\u673a\u7684\u4ee3\u7801\u5b9e\u73b0"),(0,n.kt)("p",null,"\u6211\u4eec\u5df2\u7ecf\u4ece\u4e0a\u4e00\u8282\u91cc\u4e86\u89e3\u4e86\u591a\u5c42\u611f\u77e5\u673a\u7684\u539f\u7406\u3002\u4e0b\u9762\uff0c\u6211\u4eec\u4e00\u8d77\u6765\u52a8\u624b\u5b9e\u73b0\u4e00\u4e2a\u591a\u5c42\u611f\u77e5\u673a\u3002\u9996\u5148\u5bfc\u5165\u5b9e\u73b0\u6240\u9700\u7684\u5305\u6216\u6a21\u5757\u3002"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{className:"language-python"}),'import tensorflow as tf\nimport numpy as np\nimport sys\nsys.path.append("..") # \u4e3a\u4e86\u5bfc\u5165\u4e0a\u5c42\u76ee\u5f55\u7684d2lzh_tensorflow\nimport d2lzh_tensorflow2 as d2l\nprint(tf.__version__)\n')),(0,n.kt)("h2",i({},{id:"\u624b\u52a8\u5b9e\u73b0"}),"\u624b\u52a8\u5b9e\u73b0"),(0,n.kt)("h3",i({},{id:"\u83b7\u53d6\u548c\u8bfb\u53d6\u6570\u636e"}),"\u83b7\u53d6\u548c\u8bfb\u53d6\u6570\u636e"),(0,n.kt)("p",null,"\u8fd9\u91cc\u7ee7\u7eed\u4f7f\u7528Fashion-MNIST\u6570\u636e\u96c6\u3002\u6211\u4eec\u5c06\u4f7f\u7528\u591a\u5c42\u611f\u77e5\u673a\u5bf9\u56fe\u50cf\u8fdb\u884c\u5206\u7c7b\u3002"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{className:"language-python"}),"from tensorflow.keras.datasets import fashion_mnist\n(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()\nbatch_size = 256\nx_train = tf.cast(x_train, tf.float32)\nx_test = tf.cast(x_test, tf.float32)\nx_train = x_train/255.0\nx_test = x_test/255.0\ntrain_iter = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)\ntest_iter = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)\n")),(0,n.kt)("h3",i({},{id:"\u5b9a\u4e49\u6a21\u578b\u53c2\u6570"}),"\u5b9a\u4e49\u6a21\u578b\u53c2\u6570"),(0,n.kt)("p",null,"\u6211\u4eec\u57283.6\u8282\uff08softmax\u56de\u5f52\u7684\u4ece\u96f6\u5f00\u59cb\u5b9e\u73b0\uff09\u91cc\u5df2\u7ecf\u4ecb\u7ecd\u4e86\uff0cFashion-MNIST\u6570\u636e\u96c6\u4e2d\u56fe\u50cf\u5f62\u72b6\u4e3a ",(0,n.kt)("span",i({parentName:"p"},{className:"math math-inline"}),(0,n.kt)("span",i({parentName:"span"},{className:"katex"}),(0,n.kt)("span",i({parentName:"span"},{className:"katex-mathml"}),(0,n.kt)("math",i({parentName:"span"},{xmlns:"http://www.w3.org/1998/Math/MathML"}),(0,n.kt)("semantics",{parentName:"math"},(0,n.kt)("mrow",{parentName:"semantics"},(0,n.kt)("mn",{parentName:"mrow"},"28"),(0,n.kt)("mo",{parentName:"mrow"},"\xd7"),(0,n.kt)("mn",{parentName:"mrow"},"28")),(0,n.kt)("annotation",i({parentName:"semantics"},{encoding:"application/x-tex"}),"28 \\times 28")))),(0,n.kt)("span",i({parentName:"span"},{className:"katex-html","aria-hidden":"true"}),(0,n.kt)("span",i({parentName:"span"},{className:"base"}),(0,n.kt)("span",i({parentName:"span"},{className:"strut",style:{height:"0.7278em",verticalAlign:"-0.0833em"}})),(0,n.kt)("span",i({parentName:"span"},{className:"mord"}),"28"),(0,n.kt)("span",i({parentName:"span"},{className:"mspace",style:{marginRight:"0.2222em"}})),(0,n.kt)("span",i({parentName:"span"},{className:"mbin"}),"\xd7"),(0,n.kt)("span",i({parentName:"span"},{className:"mspace",style:{marginRight:"0.2222em"}}))),(0,n.kt)("span",i({parentName:"span"},{className:"base"}),(0,n.kt)("span",i({parentName:"span"},{className:"strut",style:{height:"0.6444em"}})),(0,n.kt)("span",i({parentName:"span"},{className:"mord"}),"28"))))),"\uff0c\u7c7b\u522b\u6570\u4e3a10\u3002\u672c\u8282\u4e2d\u6211\u4eec\u4f9d\u7136\u4f7f\u7528\u957f\u5ea6\u4e3a ",(0,n.kt)("span",i({parentName:"p"},{className:"math math-inline"}),(0,n.kt)("span",i({parentName:"span"},{className:"katex"}),(0,n.kt)("span",i({parentName:"span"},{className:"katex-mathml"}),(0,n.kt)("math",i({parentName:"span"},{xmlns:"http://www.w3.org/1998/Math/MathML"}),(0,n.kt)("semantics",{parentName:"math"},(0,n.kt)("mrow",{parentName:"semantics"},(0,n.kt)("mn",{parentName:"mrow"},"28"),(0,n.kt)("mo",{parentName:"mrow"},"\xd7"),(0,n.kt)("mn",{parentName:"mrow"},"28"),(0,n.kt)("mo",{parentName:"mrow"},"="),(0,n.kt)("mn",{parentName:"mrow"},"784")),(0,n.kt)("annotation",i({parentName:"semantics"},{encoding:"application/x-tex"}),"28 \\times 28 = 784")))),(0,n.kt)("span",i({parentName:"span"},{className:"katex-html","aria-hidden":"true"}),(0,n.kt)("span",i({parentName:"span"},{className:"base"}),(0,n.kt)("span",i({parentName:"span"},{className:"strut",style:{height:"0.7278em",verticalAlign:"-0.0833em"}})),(0,n.kt)("span",i({parentName:"span"},{className:"mord"}),"28"),(0,n.kt)("span",i({parentName:"span"},{className:"mspace",style:{marginRight:"0.2222em"}})),(0,n.kt)("span",i({parentName:"span"},{className:"mbin"}),"\xd7"),(0,n.kt)("span",i({parentName:"span"},{className:"mspace",style:{marginRight:"0.2222em"}}))),(0,n.kt)("span",i({parentName:"span"},{className:"base"}),(0,n.kt)("span",i({parentName:"span"},{className:"strut",style:{height:"0.6444em"}})),(0,n.kt)("span",i({parentName:"span"},{className:"mord"}),"28"),(0,n.kt)("span",i({parentName:"span"},{className:"mspace",style:{marginRight:"0.2778em"}})),(0,n.kt)("span",i({parentName:"span"},{className:"mrel"}),"="),(0,n.kt)("span",i({parentName:"span"},{className:"mspace",style:{marginRight:"0.2778em"}}))),(0,n.kt)("span",i({parentName:"span"},{className:"base"}),(0,n.kt)("span",i({parentName:"span"},{className:"strut",style:{height:"0.6444em"}})),(0,n.kt)("span",i({parentName:"span"},{className:"mord"}),"784")))))," \u7684\u5411\u91cf\u8868\u793a\u6bcf\u4e00\u5f20\u56fe\u50cf\u3002\u56e0\u6b64\uff0c\u8f93\u5165\u4e2a\u6570\u4e3a784\uff0c\u8f93\u51fa\u4e2a\u6570\u4e3a10\u3002\u5b9e\u9a8c\u4e2d\uff0c\u6211\u4eec\u8bbe\u8d85\u53c2\u6570\u9690\u85cf\u5355\u5143\u4e2a\u6570\u4e3a256\u3002"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{className:"language-python"}),"num_inputs, num_outputs, num_hiddens = 784, 10, 256\nW1 = tf.Variable(tf.random.normal(shape=(num_inputs, num_hiddens),mean=0, stddev=0.01, dtype=tf.float32))\nb1 = tf.Variable(tf.zeros(num_hiddens, dtype=tf.float32))\nW2 = tf.Variable(tf.random.normal(shape=(num_hiddens, num_outputs),mean=0, stddev=0.01, dtype=tf.float32))\nb2 = tf.Variable(tf.random.normal([num_outputs], stddev=0.1))\n")),(0,n.kt)("h3",i({},{id:"\u5b9a\u4e49\u6fc0\u6d3b\u51fd\u6570"}),"\u5b9a\u4e49\u6fc0\u6d3b\u51fd\u6570"),(0,n.kt)("p",null,"\u8fd9\u91cc\u6211\u4eec\u4f7f\u7528\u57fa\u7840\u7684",(0,n.kt)("inlineCode",{parentName:"p"},"max"),"\u51fd\u6570\u6765\u5b9e\u73b0ReLU\uff0c\u800c\u975e\u76f4\u63a5\u8c03\u7528",(0,n.kt)("inlineCode",{parentName:"p"},"relu"),"\u51fd\u6570\u3002"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{className:"language-python"}),"def relu(x):\n    return tf.math.maximum(x,0)\n")),(0,n.kt)("h3",i({},{id:"\u5b9a\u4e49\u6a21\u578b"}),"\u5b9a\u4e49\u6a21\u578b"),(0,n.kt)("p",null,"\u540csoftmax\u56de\u5f52\u4e00\u6837\uff0c\u6211\u4eec\u901a\u8fc7",(0,n.kt)("inlineCode",{parentName:"p"},"reshape"),"\u51fd\u6570\u5c06\u6bcf\u5f20\u539f\u59cb\u56fe\u50cf\u6539\u6210\u957f\u5ea6\u4e3a",(0,n.kt)("inlineCode",{parentName:"p"},"num_inputs"),"\u7684\u5411\u91cf\u3002\u7136\u540e\u6211\u4eec\u5b9e\u73b0\u4e0a\u4e00\u8282\u4e2d\u591a\u5c42\u611f\u77e5\u673a\u7684\u8ba1\u7b97\u8868\u8fbe\u5f0f\u3002"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{className:"language-python"}),"def net(X):\n    X = tf.reshape(X, shape=[-1, num_inputs])\n    h = relu(tf.matmul(X, W1) + b1)\n    return tf.math.softmax(tf.matmul(h, W2) + b2)\n")),(0,n.kt)("h3",i({},{id:"\u5b9a\u4e49\u635f\u5931\u51fd\u6570"}),"\u5b9a\u4e49\u635f\u5931\u51fd\u6570"),(0,n.kt)("p",null,"\u4e3a\u4e86\u5f97\u5230\u66f4\u597d\u7684\u6570\u503c\u7a33\u5b9a\u6027\uff0c\u6211\u4eec\u76f4\u63a5\u4f7f\u7528Tensorflow\u63d0\u4f9b\u7684\u5305\u62ecsoftmax\u8fd0\u7b97\u548c\u4ea4\u53c9\u71b5\u635f\u5931\u8ba1\u7b97\u7684\u51fd\u6570\u3002"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{className:"language-python"}),"def loss(y_hat,y_true):\n    return tf.losses.sparse_categorical_crossentropy(y_true,y_hat)\n")),(0,n.kt)("h3",i({},{id:"\u8bad\u7ec3\u6a21\u578b"}),"\u8bad\u7ec3\u6a21\u578b"),(0,n.kt)("p",null,"\u8bad\u7ec3\u591a\u5c42\u611f\u77e5\u673a\u7684\u6b65\u9aa4\u548c3.6\u8282\u4e2d\u8bad\u7ec3softmax\u56de\u5f52\u7684\u6b65\u9aa4\u6ca1\u4ec0\u4e48\u533a\u522b\u3002\u6211\u4eec\u76f4\u63a5\u8c03\u7528",(0,n.kt)("inlineCode",{parentName:"p"},"d2l"),"\u5305\u4e2d\u7684",(0,n.kt)("inlineCode",{parentName:"p"},"train_ch3"),"\u51fd\u6570\uff0c\u5b83\u7684\u5b9e\u73b0\u5df2\u7ecf\u57283.6\u8282\u91cc\u4ecb\u7ecd\u8fc7\u3002\u6211\u4eec\u5728\u8fd9\u91cc\u8bbe\u8d85\u53c2\u6570\u8fed\u4ee3\u5468\u671f\u6570\u4e3a5\uff0c\u5b66\u4e60\u7387\u4e3a0.5\u3002"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{className:"language-python"}),"num_epochs, lr = 5, 0.5\nparams = [W1, b1, W2, b2]\nd2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, params, lr)\n")),(0,n.kt)("p",null,"\u8f93\u51fa\uff1a"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{}),"epoch 1, loss 0.8208, train acc 0.693, test acc 0.804\nepoch 2, loss 0.4784, train acc 0.822, test acc 0.832\nepoch 3, loss 0.4192, train acc 0.843, test acc 0.850\nepoch 4, loss 0.3874, train acc 0.857, test acc 0.858\nepoch 5, loss 0.3651, train acc 0.864, test acc 0.860\n")),(0,n.kt)("hr",null),(0,n.kt)("h2",i({},{id:"\u4f7f\u7528\u6846\u67b6\u5b9e\u73b0"}),"\u4f7f\u7528\u6846\u67b6\u5b9e\u73b0"),(0,n.kt)("p",null,"\u4e0b\u9762\u6211\u4eec\u4f7f\u7528Tensorflow\u6765\u5b9e\u73b0\u4e0a\u4e00\u8282\u4e2d\u7684\u591a\u5c42\u611f\u77e5\u673a\u3002\u9996\u5148\u5bfc\u5165\u6240\u9700\u7684\u5305\u6216\u6a21\u5757\u3002"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{className:"language-python"}),"import tensorflow as tf\nfrom tensorflow import keras\nfashion_mnist = keras.datasets.fashion_mnist\n")),(0,n.kt)("h3",i({},{id:"\u5b9a\u4e49\u6a21\u578b-1"}),"\u5b9a\u4e49\u6a21\u578b"),(0,n.kt)("p",null,"\u548csoftmax\u56de\u5f52\u552f\u4e00\u7684\u4e0d\u540c\u5728\u4e8e\uff0c\u6211\u4eec\u591a\u52a0\u4e86\u4e00\u4e2a\u5168\u8fde\u63a5\u5c42\u4f5c\u4e3a\u9690\u85cf\u5c42\u3002\u5b83\u7684\u9690\u85cf\u5355\u5143\u4e2a\u6570\u4e3a256\uff0c\u5e76\u4f7f\u7528ReLU\u51fd\u6570\u4f5c\u4e3a\u6fc0\u6d3b\u51fd\u6570\u3002"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{className:"language-python"}),"model = tf.keras.models.Sequential([\n    tf.keras.layers.Flatten(input_shape=(28, 28)),\n    tf.keras.layers.Dense(256, activation='relu',),\n    tf.keras.layers.Dense(10, activation='softmax')\n])\n")),(0,n.kt)("h3",i({},{id:"\u8bfb\u53d6\u6570\u636e\u5e76\u8bad\u7ec3\u6a21\u578b"}),"\u8bfb\u53d6\u6570\u636e\u5e76\u8bad\u7ec3\u6a21\u578b"),(0,n.kt)("p",null,"\u6211\u4eec\u4f7f\u7528\u4e4b\u524d\u8bad\u7ec3softmax\u56de\u5f52\u51e0\u4e4e\u76f8\u540c\u7684\u6b65\u9aa4\u6765\u8bfb\u53d6\u6570\u636e\u5e76\u8bad\u7ec3\u6a21\u578b\u3002"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{className:"language-python"}),"fashion_mnist = keras.datasets.fashion_mnist\n(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()\nx_train = x_train / 255.0\nx_test = x_test / 255.0\nmodel.compile(optimizer=tf.keras.optimizers.SGD(lr=0.5),\n             loss = 'sparse_categorical_crossentropy',\n             metrics=['accuracy'])\nmodel.fit(x_train, y_train, epochs=5,\n              batch_size=256,\n              validation_data=(x_test, y_test),\n              validation_freq=1)\n")),(0,n.kt)("p",null,"\u8f93\u51fa\uff1a"),(0,n.kt)("pre",null,(0,n.kt)("code",i({parentName:"pre"},{}),"Train on 60000 samples, validate on 10000 samples\nEpoch 1/5\n60000/60000 [==============================] - 2s 33us/sample - loss: 0.7428 - accuracy: 0.7333 - val_loss: 0.5489 - val_accuracy: 0.8049\nEpoch 2/5\n60000/60000 [==============================] - 1s 22us/sample - loss: 0.4774 - accuracy: 0.8247 - val_loss: 0.4823 - val_accuracy: 0.8288\nEpoch 3/5\n60000/60000 [==============================] - 1s 21us/sample - loss: 0.4111 - accuracy: 0.8497 - val_loss: 0.4448 - val_accuracy: 0.8401\nEpoch 4/5\n60000/60000 [==============================] - 1s 21us/sample - loss: 0.3806 - accuracy: 0.8600 - val_loss: 0.5326 - val_accuracy: 0.8132\nEpoch 5/5\n60000/60000 [==============================] - 1s 21us/sample - loss: 0.3603 - accuracy: 0.8681 - val_loss: 0.4217 - val_accuracy: 0.8448\n<tensorflow.python.keras.callbacks.History at 0x7f9868e12310>\n")))}g.isMDXComponent=!0}}]);