function _throttle(func, wait, leading, trailing) {
    var timeout, context, args
    var previous = 0

    var later = function () {
        previous = leading === false ? 0 : new Date().getTime()
        timeout = null
        func.apply(context, args)
        if (!timeout) context = args = null
    }

    var throttled = function () {
        var now = new Date().getTime()
        if (!previous && leading === false) previous = now
        var remaining = wait - (now - previous)
        context = this
        args = arguments
        if (remaining <= 0 || remaining > wait) {
            if (timeout) {
                clearTimeout(timeout)
                timeout = null
            }
            previous = now
            func.apply(context, args)
            if (!timeout) context = args = null
        } else if (!timeout && trailing !== false) {
            timeout = setTimeout(later, remaining)
        }
    }
    return throttled
}

/**
 *
 * @param wait
 * @param leading 表示是否开启第一次执行
 * @param trailing 表示是否开启停止触发的回调
 * @returns {function(*, *, *): *&{value(): void}}
 */
export default function throttle(wait = 1000, leading = true, trailing = true) {
    return function handleDescriptor(target, key, descriptor) {
        const callback = descriptor.value

        if (typeof callback !== 'function') {
            throw new SyntaxError('Only functions can be debounced')
        }

        var fn = _throttle(callback, wait, leading, trailing)

        return {
            ...descriptor,
            value() {
                var args = arguments
                fn.apply(this, args)
            }
        }
    }
}
