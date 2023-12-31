"use strict";(self.webpackChunkml_notebook=self.webpackChunkml_notebook||[]).push([[31],{3905:(e,t,n)=>{n.d(t,{Zo:()=>f,kt:()=>k});var o=n(67294);function r(e,t,n){return t in e?Object.defineProperty(e,t,{value:n,enumerable:!0,configurable:!0,writable:!0}):e[t]=n,e}function a(e,t){var n=Object.keys(e);if(Object.getOwnPropertySymbols){var o=Object.getOwnPropertySymbols(e);t&&(o=o.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),n.push.apply(n,o)}return n}function l(e){for(var t=1;t<arguments.length;t++){var n=null!=arguments[t]?arguments[t]:{};t%2?a(Object(n),!0).forEach((function(t){r(e,t,n[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(n)):a(Object(n)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(n,t))}))}return e}function p(e,t){if(null==e)return{};var n,o,r=function(e,t){if(null==e)return{};var n,o,r={},a=Object.keys(e);for(o=0;o<a.length;o++)n=a[o],t.indexOf(n)>=0||(r[n]=e[n]);return r}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(o=0;o<a.length;o++)n=a[o],t.indexOf(n)>=0||Object.prototype.propertyIsEnumerable.call(e,n)&&(r[n]=e[n])}return r}var i=o.createContext({}),s=function(e){var t=o.useContext(i),n=t;return e&&(n="function"==typeof e?e(t):l(l({},t),e)),n},f=function(e){var t=s(e.components);return o.createElement(i.Provider,{value:t},e.children)},c="mdxType",u={inlineCode:"code",wrapper:function(e){var t=e.children;return o.createElement(o.Fragment,{},t)}},m=o.forwardRef((function(e,t){var n=e.components,r=e.mdxType,a=e.originalType,i=e.parentName,f=p(e,["components","mdxType","originalType","parentName"]),c=s(n),m=r,k=c["".concat(i,".").concat(m)]||c[m]||u[m]||a;return n?o.createElement(k,l(l({ref:t},f),{},{components:n})):o.createElement(k,l({ref:t},f))}));function k(e,t){var n=arguments,r=t&&t.mdxType;if("string"==typeof e||r){var a=n.length,l=new Array(a);l[0]=m;var p={};for(var i in t)hasOwnProperty.call(t,i)&&(p[i]=t[i]);p.originalType=e,p[c]="string"==typeof e?e:r,l[1]=p;for(var s=2;s<a;s++)l[s]=n[s];return o.createElement.apply(null,l)}return o.createElement.apply(null,n)}m.displayName="MDXCreateElement"},51361:(e,t,n)=>{n.r(t),n.d(t,{assets:()=>h,contentTitle:()=>g,default:()=>O,frontMatter:()=>k,metadata:()=>y,toc:()=>v});var o=n(3905),r=Object.defineProperty,a=Object.defineProperties,l=Object.getOwnPropertyDescriptors,p=Object.getOwnPropertySymbols,i=Object.prototype.hasOwnProperty,s=Object.prototype.propertyIsEnumerable,f=(e,t,n)=>t in e?r(e,t,{enumerable:!0,configurable:!0,writable:!0,value:n}):e[t]=n,c=(e,t)=>{for(var n in t||(t={}))i.call(t,n)&&f(e,n,t[n]);if(p)for(var n of p(t))s.call(t,n)&&f(e,n,t[n]);return e},u=(e,t)=>a(e,l(t)),m=(e,t)=>{var n={};for(var o in e)i.call(e,o)&&t.indexOf(o)<0&&(n[o]=e[o]);if(null!=e&&p)for(var o of p(e))t.indexOf(o)<0&&s.call(e,o)&&(n[o]=e[o]);return n};const k={},g="Footnote test",y={unversionedId:"chtest/footnote",id:"chtest/footnote",title:"Footnote test",description:"This is some text with footnote.",source:"@site/docs/chtest/footnote.md",sourceDirName:"chtest",slug:"/chtest/footnote",permalink:"/chtest/footnote",draft:!1,tags:[],version:"current",frontMatter:{}},h={},v=[],b={toc:v},d="wrapper";function O(e){var t=e,{components:n}=t,r=m(t,["components"]);return(0,o.kt)(d,u(c(c({},b),r),{components:n,mdxType:"MDXLayout"}),(0,o.kt)("h1",c({},{id:"footnote-test"}),"Footnote test"),(0,o.kt)("p",null,"This is some text with footnote",(0,o.kt)("sup",c({parentName:"p"},{id:"fnref-1"}),(0,o.kt)("a",c({parentName:"sup"},{href:"#fn-1",className:"footnote-ref"}),"1")),"."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This is another text with another footnote",(0,o.kt)("sup",c({parentName:"p"},{id:"fnref-2"}),(0,o.kt)("a",c({parentName:"sup"},{href:"#fn-2",className:"footnote-ref"}),"2")),"."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"Test long footnote",(0,o.kt)("sup",c({parentName:"p"},{id:"fnref-3"}),(0,o.kt)("a",c({parentName:"sup"},{href:"#fn-3",className:"footnote-ref"}),"3")),"."),(0,o.kt)("p",null,"Test long footnote name",(0,o.kt)("sup",c({parentName:"p"},{id:"fnref-some-footnote"}),(0,o.kt)("a",c({parentName:"sup"},{href:"#fn-some-footnote",className:"footnote-ref"}),"some-footnote"))),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("p",null,"This page is very long."),(0,o.kt)("div",c({},{className:"footnotes"}),(0,o.kt)("hr",{parentName:"div"}),(0,o.kt)("ol",{parentName:"div"},(0,o.kt)("li",c({parentName:"ol"},{id:"fn-1"}),(0,o.kt)("p",{parentName:"li"},"This is the footnote.",(0,o.kt)("a",c({parentName:"p"},{href:"#fnref-1",className:"footnote-backref"}),"\u21a9"))),(0,o.kt)("li",c({parentName:"ol"},{id:"fn-2"}),(0,o.kt)("p",{parentName:"li"},"This is the footnote.",(0,o.kt)("a",c({parentName:"p"},{href:"#fnref-2",className:"footnote-backref"}),"\u21a9"))),(0,o.kt)("li",c({parentName:"ol"},{id:"fn-3"}),(0,o.kt)("p",{parentName:"li"},"Long footnote"),(0,o.kt)("p",{parentName:"li"},"with\nmultiple\nlines.",(0,o.kt)("a",c({parentName:"p"},{href:"#fnref-3",className:"footnote-backref"}),"\u21a9"))),(0,o.kt)("li",c({parentName:"ol"},{id:"fn-some-footnote"}),(0,o.kt)("p",{parentName:"li"},"Footnote name"),(0,o.kt)("p",{parentName:"li"},"can be long.",(0,o.kt)("a",c({parentName:"p"},{href:"#fnref-some-footnote",className:"footnote-backref"}),"\u21a9"))))))}O.isMDXComponent=!0}}]);