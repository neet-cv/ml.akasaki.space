const { getSidebar } = require("../../js/utils");

const indexData = {
    "ch0": "第零章：在开始之前",
    "ch1p1": "第一章上：HelloWorld",
    "ch1p2": "第一章下：深度学习基础",
    "ch2p1": "第二章上：卷积神经网络",
    "ch2p2": "第二章下：经典卷积神经网络",
    "ch3p1": "第三章上：谈一些计算机视觉方向",
    "ch3p2": "第三章下：了解更高级的技术",
    "appendix": "附录：永远是你的好朋友",
    "ch5": "第五章：杂七杂八的手记（暂存）",
    "ch-1": "第-1章：TensorFlow编程策略",
    "ch-2": "第-2章：数字信号处理（DSP）",
    "unlimited-paper-works": "魔法部日志（又名论文阅读日志）",
};

exports.getSidebarIndex = function () {
    return Object.entries(indexData)
        .map(([folder, title]) => ({
            title,
            children: getSidebar(folder)
        }));
};
