const global = {
    //==================================================
    // 定义 BASE URL
    //==================================================
    API_BASE_URL: process.env.NODE_ENV === 'development' ? 'http://127.0.0.1:9000/api/' : 'http://xstv2.nieqing.net/api/',

    API_FILE_URL: process.env.NODE_ENV === 'development' ? 'http://127.0.0.1/static' : 'http://xstv2.nieqing.net/static',
    //==================================================
    // 客户端不做特殊处理的信息返回
    //==================================================
    CODE_OK: 200,
    CODE_ERROR: 500,

    // ==================================================
    // 成功常量定义
    // ==================================================
    CODE_OK_FILL_FORM: 7000,
    CODE_OK_LOGIN: 8000,
    CODE_OK_LOGOUT: 8001,

    // ==================================================
    // 失败常量定义
    // ==================================================
    CODE_ERROR_DO_OP: 9000,
    CODE_ERROR_NOT_LOGIN: 9001,
    CODE_ERROR_FORM_VALID: 9002,

    PATTERN_MOBILE: /^[1][0-9]{10}$/,
    PATTERN_PASSWORD: /^(?=.*?[a-zA-Z])(?=.*?[0-9]).{8,32}$/,
    PATTERN_JOIN_CODE: /^([a-zA-Z0-9]){4,8}$/
}
export default global
