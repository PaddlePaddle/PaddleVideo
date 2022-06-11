/**
 * 保证接口在请求完成时只执行一次
 * @param target
 * @param name
 * @param descriptor
 * @returns {*}
 */
export default function once(target, name, descriptor) {
    const oldValue = descriptor.value
    let isFetch = false
    descriptor.value = async function (...args) {
        if (isFetch) {
            return
        }
        isFetch = true
        try {
            const value = await oldValue.apply(this, args)
            return value
        } finally {
            isFetch = false
        }
    }

    return descriptor
}
