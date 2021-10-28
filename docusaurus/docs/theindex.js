const { getSidebar } = require("../../js/utils");

const indexData = {
    "ch0": "第零章：在开始之前",
    "ch1p1": "第一章上：HelloWorld",
    "ch1p2": "第一章下：深度学习基础",
    "ch2p1": "第二章上：卷积神经网络",
    "ch2p2": "第二章下：经典卷积神经网络",
    "ch3p1": "第三章上：谈一些计算机视觉方向",
    "ch3p2": "第三章下：了解更高级的技术",
    "unlimited-paper-works": "魔法部日志（又名论文阅读日志）",
    "appendix-1": "附录1：好朋友们",
    "appendix-2": "附录2：数学是真正的圣经",
    "appendix-3": "附录3：信号和采样的学问（DSP）",
    "appendix-4": "附录4：TensorFlow编程策略",
};

exports.getSidebarIndex = function () {
    return Object.entries(indexData)
        .map(([folder, title]) => ({
            title,
            folder,
        }));
};
