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
    console.info('==> Getting pages under ' + folder);
    var pages = getPages(`docs/${folder}`)
        .map(path => {
            console.log(parseFloat(path.match(/\[([\d.]+)\]/)[1]))
            return {
                index: parseFloat(path.match(/\[([\d.]+)\]/)[1]),
                path
            }
        })
        .sort((a, b) => a.index - b.index)
        .map(({ index, path }) => {
            const title = readMDFileTitle(`docs/${folder}/${path}`);
            return {
                index,
                title,
                path: `/${folder}/${path}`,
                collapsable: false,
            };
        });
    pages.forEach(x => console.info('-> page ' + x.index + ': ' + x.path));
    return pages;
}

function readMDFileTitle(path) {
    var match = fs.readFileSync(path, 'utf8').match(/^# (.+?)[\r\n]/m);
    return match ? match[1] : path;
}

module.exports = {
    getSidebar,
};
