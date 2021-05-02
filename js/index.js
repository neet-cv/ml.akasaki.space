const fs = require("fs");
const path = require("path");
const readLine = require("readline");
/**
 *
 * @param {string} dir
 */
function getPages(dir) {
    return fs.readdirSync(dir).filter((self) => {
        return !fs.statSync(path.join(dir, self)).isDirectory()
    });
}

/**
 * 获取侧边栏
 * @param {string} folder 目录文件名
 * @param {string} title 标题
 */
function getSidebar(folder) {
    let pages = getPages(`docs/${folder}`);
    const sidebar = [];
    pages.sort(function(a, b) {
        //todo 等学了正则再回来改吧....
        return a.match(/\<(.+)\>/g)[0].replace('<', '').replace('>', '') * 1 - b.match(/\<(.+)\>/g)[0].replace('<', '').replace('>', '') * 1
    });
    pages.forEach((md) => {
        const name = md.substring(0, md.length - 3)
        sidebar.push({
            title: name.substring(name.indexOf('>') + 1),
            path: `/${folder}/${md}`,
        });
    });
    return sidebar;
}

module.exports = {
    getSidebar,
};