(function () {
    var designW1 = 1920; //设计稿宽，我这次的web端设计稿是1920*1080的
    var font_rate = 100;
    var designW2 = 375;//移动端设计稿宽，一般按照iphone6的设备宽度来设计。

    if (document.documentElement.clientWidth > 750) {
        //适配
        document.getElementsByTagName("html")[0].style.fontSize = (document.documentElement.clientWidth) / designW1 * font_rate + "px";

        //监测窗口大小变化
        window.addEventListener("onorientationchange" in window ? "orientationchange" : "resize", function () {
            document.getElementsByTagName("html")[0].style.fontSize = (document.documentElement.clientWidth) / designW1 * font_rate + "px";
        }, false);
    } else {
        //适配
        document.getElementsByTagName("html")[0].style.fontSize = (document.documentElement.clientWidth) / designW2 * font_rate + "px";

        //监测窗口大小变化
        window.addEventListener("onorientationchange" in window ? "orientationchange" : "resize", function () {
            document.getElementsByTagName("html")[0].style.fontSize = (document.documentElement.clientWidth) / designW2 * font_rate + "px";
        }, false);
    }

})();
