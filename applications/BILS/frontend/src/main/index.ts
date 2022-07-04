import {app, BrowserWindow, Menu, shell, dialog, nativeImage} from 'electron'
import './dialog'
import {Logger} from './logger'
import {initialize} from './services'
import createBaseWorker from './workers/index?worker'
import indexPreload from '/@preload/index'
import indexHtmlUrl from '/@renderer/index.html'
import logoUrl from '/@static/logo.png'

const template = [
  {
    label: '编辑',
    submenu: [{
      label: '撤销',
      accelerator: 'CmdOrCtrl+Z',
      role: 'undo'
    }, {
      label: '重做',
      accelerator: 'Shift+CmdOrCtrl+Z',
      role: 'redo'
    }, {
      type: 'separator'
    }, {
      label: '剪切',
      accelerator: 'CmdOrCtrl+X',
      role: 'cut'
    }, {
      label: '复制',
      accelerator: 'CmdOrCtrl+C',
      role: 'copy'
    }, {
      label: '粘贴',
      accelerator: 'CmdOrCtrl+V',
      role: 'paste'
    }, {
      label: '全选',
      accelerator: 'CmdOrCtrl+A',
      role: 'selectall'
    }, {
      type: 'separator'
    }, {
      label: '关于',
      click: function (item, focusedWindow) {
        if (focusedWindow) {
          const options = {
            type: 'info',
            title: '关于 EIVideo',
            buttons: ['确定'],
            message: '版本号：内测',
            icon: nativeImage.createFromPath('../../static/logo.png')
          }
          dialog.showMessageBox(focusedWindow, options, function () {
          })
        }
      }
    }]
  },
  {
    label: '查看',
    submenu: [{
      label: '重载',
      accelerator: 'CmdOrCtrl+R',
      click: function (item, focusedWindow) {
        if (focusedWindow) {
          // 重载之后, 刷新并关闭所有的次要窗体
          if (focusedWindow.id === 1) {
            BrowserWindow.getAllWindows().forEach(function (win) {
              if (win.id > 1) {
                win.close()
              }
            })
          }
          focusedWindow.reload()
        }
      }
    }, {
      label: '切换全屏',
      accelerator: (function () {
        if (process.platform === 'darwin') {
          return 'Ctrl+Command+F'
        } else {
          return 'F11'
        }
      })(),
      click: function (item, focusedWindow) {
        if (focusedWindow) {
          focusedWindow.setFullScreen(!focusedWindow.isFullScreen())
        }
      }
    },
      {
        label: '切换开发者工具',
        accelerator: (function () {
          if (process.platform === 'darwin') {
            return 'Alt+Command+I'
          } else {
            return 'Ctrl+Shift+I'
          }
        })(),
        click: function (item, focusedWindow) {
          if (focusedWindow) {
            focusedWindow.toggleDevTools()
          }
        }
      },
    ]
  },
  {
    label: '窗口',
    role: 'window',
    submenu: [{
      label: '最小化',
      accelerator: 'CmdOrCtrl+M',
      role: 'minimize'
    }, {
      label: '关闭',
      accelerator: 'CmdOrCtrl+W',
      role: 'close'
    }, {
      type: 'separator'
    }, {
      label: '重新打开窗口',
      accelerator: 'CmdOrCtrl+Shift+T',
      enabled: false,
      key: 'reopenMenuItem',
      click: function () {
        app.emit('activate')
      }
    }]
  },
  {
    label: '帮助',
    role: 'help',
    submenu: [{
      label: 'Paddle官网',
      click: function () {
        shell.openExternal('https://www.paddlepaddle.org.cn')
      }
    }]
  }
]

async function main() {
  const logger = new Logger()
  logger.initialize(app.getPath('userData'))
  initialize(logger)
  app.whenReady().then(() => {
    const main = createWindow()
    const ses = main.webContents.session
    ses.loadExtension('/Users/niunaicaomeixiaobinggan/Library/Application Support/Google/Chrome/Default/Extensions/ljjemllljcmogpfapbkkighbhhppjdbg/6.0.0.21_0')
    // const [x, y] = main.getPosition()
    // const side = createSecondWindow()
    // side.setPosition(x + 800 + 5, y)
  })
  // thread_worker example
  createBaseWorker({workerData: 'worker world'}).on('message', (message) => {
    logger.log(`Message from worker: ${message}`)
  }).postMessage('')
}

function addUpdateMenuItems(items, position) {
  if (process.mas) return

  const version = app.getVersion()
  let updateItems = [{
    label: `Version ${version}`,
    enabled: false
  }, {
    label: '正在检查更新',
    enabled: false,
    key: 'checkingForUpdate'
  }, {
    label: '检查更新',
    visible: false,
    key: 'checkForUpdate',
    click: function () {
      require('electron').autoUpdater.checkForUpdates()
    }
  }, {
    label: '重启并安装更新',
    enabled: true,
    visible: false,
    key: 'restartToUpdate',
    click: function () {
      require('electron').autoUpdater.quitAndInstall()
    }
  }]

  items.splice.apply(items, [position, 0].concat(updateItems))
}

function createWindow() {

  if (process.platform === 'darwin') {
    const name = app.getName()
    template.unshift({
      label: name,
      submenu: [{
        label: `关于 ${name}`,
        role: 'about'
      }, {
        type: 'separator'
      }, {
        label: '服务',
        role: 'services',
        submenu: []
      }, {
        type: 'separator'
      }, {
        label: `隐藏 ${name}`,
        accelerator: 'Command+H',
        role: 'hide'
      }, {
        label: '隐藏其它',
        accelerator: 'Command+Alt+H',
        role: 'hideothers'
      }, {
        label: '显示全部',
        role: 'unhide'
      }, {
        type: 'separator'
      }, {
        label: '退出',
        accelerator: 'Command+Q',
        click: function () {
          app.quit()
        }
      }]
    })

    // 窗口菜单.
    template[3].submenu.push({
      type: 'separator'
    }, {
      label: '前置所有',
      role: 'front'
    })

    addUpdateMenuItems(template[0].submenu, 1)
  }

  if (process.platform === 'win32') {
    const helpMenu = template[template.length - 1].submenu
    addUpdateMenuItems(helpMenu, 0)
  }
  // Create the browser window.
  const mainWindow = new BrowserWindow({
    height: 1200,
    width: 1600,
    webPreferences: {
      preload: indexPreload,
      contextIsolation: true,
      nodeIntegration: false,
      webSecurity: false
    },
    icon: logoUrl
  })
  // const mainMenu = Menu.buildFromTemplate(null)
  // Menu.setApplicationMenu(mainMenu);
  mainWindow.loadURL(indexHtmlUrl)
  return mainWindow
}

function findReopenMenuItem() {
  const menu = Menu.getApplicationMenu()
  if (!menu) return

  let reopenMenuItem
  menu.items.forEach(function (item) {
    if (item.submenu) {
      item.submenu.items.forEach(function (item) {
        if (item.key === 'reopenMenuItem') {
          reopenMenuItem = item
        }
      })
    }
  })
  return reopenMenuItem
}

// ensure app start as single instance
if (!app.requestSingleInstanceLock()) {
  app.quit()
}

app.on('window-all-closed', () => {
  let reopenMenuItem = findReopenMenuItem()
  if (reopenMenuItem) reopenMenuItem.enabled = true
  if (process.platform !== 'darwin') {
    app.quit()
  }
})
app.on('ready', function () {
  const menu = Menu.buildFromTemplate(template)
  Menu.setApplicationMenu(menu)
})

app.on('browser-window-created', function () {
  let reopenMenuItem = findReopenMenuItem()
  if (reopenMenuItem) reopenMenuItem.enabled = false
})
process.nextTick(main)
