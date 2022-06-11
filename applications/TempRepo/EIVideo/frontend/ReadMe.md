
## 打包常见问题

### windows 环境打包
#### electron-builder 不能下载到 electron文件

通过在 build.config.js 中添加配置解决

```
  electronDownload: {
    mirror: 'https://npm.taobao.org/mirrors/electron/'
  }
```


#### 需要拷贝的文件
使用 electron-builder 打包时，因为网络问题，一些软件经常难以下载下来。  
但这些软件打包时又需要用到，为了方便大家使用，我把它放到了这里：
```
tools\electron-builder-Cache.zip
拷贝到本机对应的Cache下面就可以。
C:\Users\Administrator\AppData\Local\electron-builder\Cache
```
