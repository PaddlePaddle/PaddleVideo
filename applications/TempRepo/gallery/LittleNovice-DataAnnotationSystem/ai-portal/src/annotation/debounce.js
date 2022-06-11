function _debounce(func, wait, immediate) {
    var timeout

    return function () {
        var context = this
        var args = arguments

        if (timeout) clearTimeout(timeout)
        if (immediate) {
            var callNow = !timeout
            timeout = setTimeout(function () {
                timeout = null
            }, wait)
            if (callNow) func.apply(context, args)
        } else {
            timeout = setTimeout(function () {
                func.apply(context, args)
            }, wait)
        }
    }
}

/**
 * 防抖函数
 * @param wait
 * @param immediate 判断是否是立刻执行
 * @returns {function(*, *, *): *&{value(): void}}
 */
export default function debounce(wait = 1000, immediate = true) {
    return function handleDescriptor(target, key, descriptor) {
        const callback = descriptor.value

        if (typeof callback !== 'function') {
            throw new SyntaxError('Only functions can be debounced')
        }

        var fn = _debounce(callback, wait, immediate)

        return {
            ...descriptor,
            value() {
                var args = arguments
                fn.apply(this, args)
            }
        }
    }
}
