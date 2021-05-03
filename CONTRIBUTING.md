# 如何向魔法部日志（目录最后一部分的论文阅读日志）添加新的内容
1. 首先，fork本仓库并且克隆到本地。
2. 打开本仓库的docs/unlimited-paper-works
3. 查看最大的文件编号，每个文件的命名格式是"<编号>名称.md"
4. 新建自己的文件，例如当前的最大编号为n，那么你新建的文件应该是"<n+1>名称.md"。其中名称是你所阅读的论文的名称，仅包含大小写字母、数字和横线。
  例如你阅读的论文名称是：
  ```
  Decoders Matter for Semantic Segmentation: Data-Dependent Decoding Enables Flexible Feature Aggregation
  ```
  那么你应该给文件起名为：
  ```
  <n+1>Decoders-Matter-for-Semantic-Segmentation-Data-Dependent-Decoding-Enables-Flexible-Feature-Aggregation.md
  ```
  也就是说，忽略除了'-'以外的任何符号，并将空格替换成'-'。
5. 按照以下模板进行撰写：
  ```md
  # 论文名称
  
  #### 这篇笔记的写作者是[你的名字](你的github首页地址)。
  
  你对这篇论文的几句话的总结
  
  > 论文的abstract或是某种形式的简介原文
  
  ---
  
  笔记内容...
  
  ```
  其中，md文件内产生的图片应使用相对文件路径。也就是你的图片文件应该在unlimited-paper-works/src/文件名去去掉.md后缀/
  
其他的细节你可以在现存的论文笔记中找到参考。
