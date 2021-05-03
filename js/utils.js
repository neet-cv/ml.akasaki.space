const fs = require("fs");
const path = require("path");

/**
 * 扫描dir下面的.md文件
 * @param {string} dir
 */
function getPages(dir) {
    return fs.readdirSync(dir).filter((self) => {
        return !fs.statSync(path.join(dir, self)).isDirectory();
    });
}

/**
 * 获取侧边栏
 * @param {string} folder 目录文件名
 * @param {string} title 标题
 */
function getSidebar(folder) {
    return getPages(`docs/${folder}`)
        .map(path => ({
            index: parseInt(path.match(/^\[(.+)\]/g)[1]),
            path
        }))
        .sort((a, b) => a.index - b.index)
        .map(({ path }) => {
            const title = readMDFileTitle(`docs/${folder}/${path}`);
            return {
                title,
                path: `/${folder}/${path}`,
                collapsable: false,
            };
        });
}

function readMDFileTitle(path) {
    return fs.readFileSync(path, 'utf8').match(/^# (.+?)\n/m)[1];
}

module.exports = {
    getSidebar,
};