module.exports = {
    // 申明为主配置，如果配置为true就不会在向外寻找配置文件
    root: true,
    // 指定需要使用的环境提供预定义的全局变量  目前我们使用的export import 就是node环境中的
    env: {
        node: true
    },
    extends: [
        // 最基础的vue3 eslint 规则 还有vue2的规则等
        'plugin:vue/vue3-essential',
        // eslint 核心功能  eslint:all 表示引入所有功能
        'eslint:recommended',
        // eslint格式美化 核心功能
        'plugin:prettier/recommended'],
    parserOptions: {
        // 提供eslint本身不支持的特性
        parser: 'babel-eslint',
        ecmaFeatures: {
            // 支持装饰器 @符号
            legacyDecorators: true
        }
    },
    rules: {
        // console是否会报错
        'no-console': 'off',
        // debugger是否会报错
        'no-debugger': 'off',
        // vue3 html 格式化 https://eslint.vuejs.org/rules/html-indent.html#options
        'vue/html-indent': [
            'error',
            'tab',
            {
                // 属性缩进数
                attribute: 1,
                // 顶级语句的缩进倍数
                baseIndent: 1,
                // 右括号缩进数
                closeBracket: 0,
                //  在多行情况下，属性是否应与第一个属性垂直对齐的条件
                alignAttributesVertically: true,
                // 忽略节点的选择器
                ignores: []
            }
        ],
        // 缩进配置
        indent: [
            1,
            //  使用tab缩进
            'tab',
            {
                // switch语句缩进配置
                SwitchCase: 1
            }
        ],
        // eslint格式美化配置
        'prettier/prettier': [
            // 格式错误抛出eslint错误
            'error',
            {
                // tab宽度
                tabWidth: 4,
                // 使用单引号
                singleQuote: true,
                // 结尾没有分号
                semi: false,
                // 使用tab进行缩进
                useTabs: true,
                // 不换行
                proseWrap: 'never',
                // 多少字符换行
                printWidth: 900000,
                "trailingComma": "none"
            }
        ]
    }
}
