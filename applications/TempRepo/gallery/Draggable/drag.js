/**
 * @description - drag-拖动缩放类
 * @author - Candy
 * @creationTime - 2018-12-11
 * https://iiter.cn
 * @修改人 - XXX
 * @修改记录 - XXX
 * @修改时间 - XXX
 *
 */
class Draggable {
    /**
     *
     * @param {*} elem  -拖拽的元素
     * @param {*} dragHandle -拖拽的手柄
     * @param {*} resizeHandle -缩放的手柄
     * @param {*} resizeHandle  -是否按照像素定位(不是则为百分比)
     * @param {*} dragFn - 拖拽完成后的回调
     * @param {*} resizeFn - 缩放完成后的回调
     */
    constructor(container, elem, dragHandle, resizeHandle, isPixel, dragFn, resizeFn) {
        this.container = container;
        this.elem = elem;
        this.dragHandle = dragHandle;
        this.resizeHandle = resizeHandle;
        this.isPixel = isPixel;
        this.dragFn = dragFn;
        this.resizeFn = resizeFn;
        this.dragMinWidth = 50; //最小宽度
        this.dragMinHeight = 50; //最大宽度
        this.init();
        this.drag();
        this.resize();
    }

    init() {
        if (getStyle(this.container, "position") === "") {
            this.container.style.position = "relative";
        }
        if (getStyle(this.elem, "position") === "") {
            this.elem.style.position = "absolute";
        }
    }

    drag() {
        let disX = 0;
        let disY = 0;
        let realL = "";
        let realT = "";
        this.dragHandle = this.dragHandle || this.elem;
        this.dragHandle.style.cursor = "move";
        this.dragHandle.onmousedown = event => {
            var event = event || window.event;
            disX = event.clientX - this.elem.offsetLeft;
            disY = event.clientY - this.elem.offsetTop;

            document.onmousemove = event => {
                var event = event || window.event;
                let iL = event.clientX - disX;
                let iT = event.clientY - disY;
                let maxL = this.elem.parentNode.clientWidth - this.elem.offsetWidth;
                let maxT = this.elem.parentNode.clientHeight - this.elem.offsetHeight;

                iL <= 0 && (iL = 0);
                iT <= 0 && (iT = 0);
                iL >= maxL && (iL = maxL);
                iT >= maxT && (iT = maxT);
                // this.elem.style.left = iL + "px";
                // this.elem.style.top = iT + "px";
                if (this.isPixel) {
                    //按像素
                    realL = iL + "px";
                    realT = iT + "px";
                } else {
                    //按百分比
                    realL = (iL / this.elem.parentNode.clientWidth) * 100 + "%";
                    realT = (iT / this.elem.parentNode.clientHeight) * 100 + "%";
                }

                this.elem.style.left = realL;
                this.elem.style.top = realT;

                return false
            };

            document.onmouseup = () => {
                document.onmousemove = null;
                document.onmouseup = null;
                this.releaseCapture && this.releaseCapture()
                if (this.dragFn) {
                    this.dragFn({
                        left: realL,
                        top: realT
                    });
                }
            };
            this.setCapture && this.setCapture();
            return false
        };

    }

    resize() {
        this.resizeHandle.onmousedown = event => {
            var event = event || window.event;
            let disX = event.clientX - this.resizeHandle.offsetLeft;
            let disY = event.clientY - this.resizeHandle.offsetTop;
            let iW = 0;
            let iH = 0;
            let realW = "";
            let realH = "";
            document.onmousemove = event => {
                var event = event || window.event;

                let iL = event.clientX - disX;
                let iT = event.clientY - disY;
                let maxW = this.elem.parentNode.clientWidth - this.elem.offsetLeft - 2;
                let maxH = this.elem.parentNode.clientHeight - this.elem.offsetTop - 2;
                iW = this.resizeHandle.offsetWidth + iL;
                iH = this.resizeHandle.offsetHeight + iT;

                // 宽
                iW < this.dragMinWidth && (iW = this.dragMinWidth);
                iW > maxW && (iW = maxW);
                // lockX || (this.elem.style.width = iW + "px");
                if (this.isPixel) {
                    realW = iW + "px";
                } else {
                    realW = (iW / this.elem.parentNode.clientWidth) * 100 + "%";
                }
                this.elem.style.width = realW;

                // 高
                iH < this.dragMinHeight && (iH = this.dragMinHeight);
                iH > maxH && (iH = maxH);
                // lockY || (this.elem.style.height = iH + "px");
                if (this.isPixel) {
                    realH = iH + "px"
                } else {
                    realH = (iH / this.elem.parentNode.clientHeight) * 100 + "%"
                }
                this.elem.style.height = realH;
                if ((iW == this.dragMinWidth) || (iH == this.dragMinHeight)) document.onmousemove = null;
                return false;
            };
            document.onmouseup = () => {
                document.onmousemove = null;
                document.onmouseup = null;
                if (this.resizeFn) {
                    this.resizeFn({
                        width: realW,
                        height: realH
                    });
                }
            };
            return false;
        }
    }

}

function getStyle(obj, attr) {
    const style = obj.currentStyle ? obj.currentStyle[attr] : getComputedStyle(obj, false)[attr];
    if (!style || style === "static") {
        return "";
    }
    return style;
}
