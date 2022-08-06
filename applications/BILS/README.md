# BILS



## BILS介绍

BILS (Baidu Intelligent Labeling System) 是一款支持时间轴打标签的视频标注软件，可被用于视频事件定位 、短视频分类等任务的标注工作。

![img](imgs/UI2.png)

## 安装说明

### 开发版本运行方式

```shell
cd frontend
npm install
npm run dev
```

### 安装包打包方式

#### 1. 更换package.json的内容

如果是Mac：

```json
{
  "name": "BILS",
  "version": "0.0.0",
  "private": true,
  "description": "",
  "repository": {
    "type": "git",
    "url": "https://github.com/ci010/frontend.git"
  },
  "scripts": {
    "dev:docs": "node scripts/dev.docs.js",
    "dev": "node scripts/dev.js",
    "build": "node scripts/build.js",
    "build:dir": "cross-env BUILD_TARGET=dir node scripts/build.js",
    "build:lite": "cross-env BUILD_TARGET=lite node scripts/build.js",
    "build:production": "cross-env BUILD_TARGET=production node scripts/build.js",
    "lint": "eslint --ext .ts,.vue,.js src scripts",
    "lint:fix": "npm run lint -- --fix",
    "postinstall": "node ./scripts/dev.install.js"
  },
  "author": {
    "email": "example@email.com",
    "name": "example"
  },
  "license": "MIT",
  "dependencies": {
    "prettier": "^2.5.1",
    "@ffmpeg-installer/ffmpeg": "^1.1.0",
    "@ffprobe-installer/ffprobe": "^1.4.1",
    "@vueuse/components": "^8.0.1",
    "@vueuse/core": "^7.7.0",
    "dmg-license": "^1.0.11",
    "electron-updater": "^4.3.5",
    "element-plus": "^2.0.4",
    "fluent-ffmpeg": "^2.1.2",
    "mitt": "^3.0.0",
    "pinia": "^2.0.11",
    "rollup": "^2.70.0",
    "sass": "^1.49.9",
    "uuid": "^8.3.2",
    "vue": "^3.2.31",
    "vue-json-viewer": "^3.0.4",
    "vue-router": "^4.0.10",
    "vue3-video-play": "^1.3.1-beta.6"
  },
  "external": [
    "electron-updater",
    "fs-extra",
    "prettier",
    "@ffmpeg-installer/ffmpeg",
    "@ffprobe-installer/ffprobe",
    "@vueuse/components",
    "@vueuse/core",
    "element-plus",
    "fluent-ffmpeg",
    "mitt",
    "pinia",
    "rollup",
    "sass",
    "uuid",
    "vue",
    "vue-json-viewer",
    "vue-router",
    "vue3-video-play"
  ],
  "devDependencies": {
    "@rollup/plugin-alias": "^3.1.2",
    "@rollup/plugin-commonjs": "^19.0.0",
    "@rollup/plugin-json": "^4.1.0",
    "@rollup/plugin-node-resolve": "^13.0.0",
    "@rollup/plugin-typescript": "^8.2.1",
    "@types/fs-extra-promise": "^1.0.10",
    "@types/uuid": "^8.3.4",
    "@vitejs/plugin-vue": "^1.2.3",
    "@vue/compiler-sfc": "^3.0.8",
    "autoprefixer": "^10.4.2",
    "builtin-modules": "^3.1.0",
    "chalk": "^4.1.0",
    "cross-env": "^7.0.2",
    "electron": "^13.1.2",
    "electron-builder": "^23.3.3",
    "esbuild": "^0.12.8",
    "eslint": "^7.9.0",
    "extract-zip": "^1.7.0",
    "fs-extra": "^9.0.1",
    "fs-extra-promise": "^1.0.1",
    "got": "^9.6.0",
    "magic-string": "^0.25.7",
    "postcss": "^8.4.7",
    "rollup": "^2.38.5",
    "tailwindcss": "^3.0.23",
    "tslib": "^1.14.1",
    "typescript": "^4.1.2",
    "unplugin-auto-import": "^0.6.1",
    "unplugin-element-plus": "^0.3.1",
    "unplugin-vue-components": "^0.17.21",
    "vite": "^2.3.7",
    "vue-loader-v16": "^16.0.0-beta.5.4"
  },
  "optionalDependencies": {
    "@types/node": "^14.14.7",
    "@typescript-eslint/eslint-plugin": "^4.7.0",
    "@typescript-eslint/parser": "^4.7.0",
    "eslint": "^7.9.0",
    "eslint-config-standard": "^14.1.1",
    "eslint-plugin-import": "^2.22.1",
    "eslint-plugin-node": "^11.1.0",
    "eslint-plugin-promise": "^4.2.1",
    "eslint-plugin-standard": "^4.0.2",
    "eslint-plugin-vue": "^7.1.0"
  }
}

```

如果是Windows：

```JSON
{
  "name": "BILS",
  "version": "0.0.0",
  "private": true,
  "description": "",
  "repository": {
    "type": "git",
    "url": "https://github.com/ci010/frontend.git"
  },
  "scripts": {
    "dev:docs": "node scripts/dev.docs.js",
    "dev": "node scripts/dev.js",
    "build": "node scripts/build.js",
    "build:dir": "cross-env BUILD_TARGET=dir node scripts/build.js",
    "build:lite": "cross-env BUILD_TARGET=lite node scripts/build.js",
    "build:production": "cross-env BUILD_TARGET=production node scripts/build.js",
    "lint": "eslint --ext .ts,.vue,.js src scripts",
    "lint:fix": "npm run lint -- --fix",
    "postinstall": "node ./scripts/dev.install.js"
  },
  "author": {
    "email": "example@email.com",
    "name": "example"
  },
  "license": "MIT",
  "dependencies": {
    "@ffmpeg-installer/ffmpeg": "^1.1.0",
    "@ffprobe-installer/ffprobe": "^1.4.1",
    "@floating-ui/core": "^0.6.2",
    "@vueuse/components": "^8.0.1",
    "@vueuse/core": "^7.7.0",
    "electron-updater": "^4.3.5",
    "element-plus": "^2.0.4",
    "fluent-ffmpeg": "^2.1.2",
    "mitt": "^3.0.0",
    "pinia": "^2.0.11",
    "prettier": "^2.5.1",
    "rollup": "^2.70.0",
    "sass": "^1.49.9",
    "uuid": "^8.3.2",
    "vue": "^3.2.31",
    "vue-json-viewer": "^3.0.4",
    "vue-router": "^4.0.10",
    "vue3-video-play": "^1.3.1-beta.6",
    "@vue/compiler-dom": "3.2.35"
  },
  "external": [
    "electron-updater",
    "fs-extra",
    "prettier",
    "@ffmpeg-installer/ffmpeg",
    "@ffprobe-installer/ffprobe",
    "@vueuse/components",
    "@vueuse/core",
    "element-plus",
    "fluent-ffmpeg",
    "@floating-ui/core",
    "mitt",
    "pinia",
    "rollup",
    "sass",
    "uuid",
    "vue",
    "vue-json-viewer",
    "vue-router",
    "vue3-video-play",
    "@vue/compiler-dom",
    "vue/compiler-sfc",
    "@vue/runtime-dom",
    "@vue/server-renderer",
    "nanoid",
    "@vueuse/metadata",
    "@vueuse/shared"
  ],
  "devDependencies": {
    "@rollup/plugin-alias": "^3.1.2",
    "@rollup/plugin-commonjs": "^19.0.0",
    "@rollup/plugin-json": "^4.1.0",
    "@rollup/plugin-node-resolve": "^13.0.0",
    "@rollup/plugin-typescript": "^8.2.1",
    "@types/fs-extra-promise": "^1.0.10",
    "@types/uuid": "^8.3.4",
    "@vitejs/plugin-vue": "^1.2.3",
    "@vue/compiler-sfc": "^3.2.35",
    "autoprefixer": "^10.4.2",
    "builtin-modules": "^3.1.0",
    "chalk": "^4.1.0",
    "cross-env": "^7.0.2",
    "electron": "^13.1.2",
    "electron-builder": "^23.3.3",
    "esbuild": "^0.12.8",
    "eslint": "^7.9.0",
    "extract-zip": "^1.7.0",
    "fs-extra": "^9.0.1",
    "fs-extra-promise": "^1.0.1",
    "got": "^9.6.0",
    "magic-string": "^0.25.7",
    "postcss": "^8.4.7",
    "rollup": "^2.38.5",
    "tailwindcss": "^3.0.23",
    "tslib": "^1.14.1",
    "typescript": "^4.1.2",
    "unplugin-auto-import": "^0.6.1",
    "unplugin-element-plus": "^0.3.1",
    "unplugin-vue-components": "^0.17.21",
    "vite": "^2.3.7",
    "vue-loader-v16": "^16.0.0-beta.5.4",
    "@vue/runtime-dom": "3.2.35",
    "@vue/server-renderer": "3.2.35",
    "nanoid": "3.3.4",
    "@vueuse/metadata": "8.5.0"
  },
  "optionalDependencies": {
    "@types/node": "^14.14.7",
    "@typescript-eslint/eslint-plugin": "^4.7.0",
    "@typescript-eslint/parser": "^4.7.0",
    "eslint": "^7.9.0",
    "eslint-config-standard": "^14.1.1",
    "eslint-plugin-import": "^2.22.1",
    "eslint-plugin-node": "^11.1.0",
    "eslint-plugin-promise": "^4.2.1",
    "eslint-plugin-standard": "^4.0.2",
    "eslint-plugin-vue": "^7.1.0"
  }
}

```

#### 2. 运行下面代码生成安装包

```shell
cd frontend
npm install
npm run build:production
```



## 功能使用说明

带有缩略图，打标签支持快捷键（按1,2,3……依次对应从上到下）

![img](imgs/UI1.png)

标签高度自定义，双击改名

<img src="imgs/colorpane.png" alt="img" style="zoom:50%;" /> <img src="imgs/edit.png" alt="img" style="zoom: 50%;" />

缩略图可放大缩小和调整宽度

![img](imgs/timeline1.png)

![img](imgs/timeline2.png)

打开片选菜单，支持全屏打标签

![img](imgs/fullscreen.png)

导出可自定义，可实时预览

![img](imgs/output.png)

两套主题可切换，亮、暗，视频播放支持画中画，镜像，循环，倍速，音量调整，网页全屏

![img](imgs/themes.png)

项目配置，并支持导入配置

![outport](imgs/outport.png)



## 安装常见问题

### 1. rollup库出现版本问题

上传代码时需要一起上传package-lock.json文件，以锁定版本号。因为库在实时更新，可能下一个版本就会出现兼容性问题，所以必须在锁定时锁定版本号。注意，cnpm不会生成package-lock.json文件.

解决方式：用npm i安装完package.json后，卸载rollup，然后再安装指定版本即可。
![img.png](imgs/img.png)

   ```shell
    npm i
    npm uninstall rollup
    npm install rollup@v2.70.0
   ```

如果还是不行，确保node版本是否是v16

### 2. 安装electron镜像问题报错
![img.png](imgs/electron.png)

   ```shell
    npm config set electron_mirror "https://npm.taobao.org/mirrors/electron/"
   ```

### 3. mac下ffmpeg问题报错
![img.png](imgs/ffmpeg.png)

   ```shell
   npm install --save @ffprobe-installer/ffprobe
   npm install --save @ffmpeg-installer/ffmpeg
   ```

### 4. Mac上安装包打包问题

![image-20220806214725154](/Users/ttjygbtj/Library/Application Support/typora-user-images/image-20220806214725154.png)

```shell
npm install electron-builder@23.0.2
```



## 使用常见问题

该版本肯定存在诸多需要改进的地方，开源出来，欢迎各位二次开发！


