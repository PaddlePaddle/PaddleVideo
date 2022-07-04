import {app, BrowserWindow} from 'electron'
import {Socket} from 'net'
// @ts-ignore
import extensions from 'vue-devtools'
// eslint-disable-next-line import/first
import './index'

app.on('browser-window-created', (event, window) => {
  if (!window.webContents.isDevToolsOpened()) {
    window.webContents.openDevTools()
    window.webContents.session.loadExtension(extensions)
      .catch((e) => {
        console.error('Fail to load vue extension. Please run "npm run postinstall" to install extension, or remove it and try again!')
        console.error(e)
      })
  }
})

const devServer = new Socket({}).connect(3031, '127.0.0.1')
devServer.on('data', () => {
  BrowserWindow.getAllWindows().forEach(w => w.reload())
})
