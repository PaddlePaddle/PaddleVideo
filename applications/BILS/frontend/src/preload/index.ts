import {clipboard, contextBridge, Dialog, ipcRenderer, IpcRenderer, shell} from 'electron'

console.log('hello world 1st preload!')

/**
 * Wrapper of ipc renderer.
 *
 * So the `contextIsolation: true` won't prevent you to use method inherit from EventEmitter,
 * lile `ipcRenderer.on`
 */
const _ipcRenderer: IpcRenderer = {
  invoke: (channel, ...args) => ipcRenderer.invoke(channel, ...args),
  on: (channel, listener) => {
    ipcRenderer.on(channel, listener)
    return _ipcRenderer
  },
  once: (channel, listener) => {
    ipcRenderer.once(channel, listener)
    return _ipcRenderer
  },
  postMessage: (channel, message, transfers) => ipcRenderer.postMessage(channel, message, transfers),
  removeAllListeners: (channel) => {
    ipcRenderer.removeAllListeners(channel)
    return _ipcRenderer
  },
  removeListener: (channel, listener) => {
    ipcRenderer.removeListener(channel, listener)
    return _ipcRenderer
  },
  send: (channel, ...args) => ipcRenderer.send(channel, ...args),
  sendSync: (channel, ...args) => ipcRenderer.sendSync(channel, ...args),
  sendTo: (id, channel, ...args) => ipcRenderer.sendTo(id, channel, ...args),
  sendToHost: (channel, ...args) => ipcRenderer.sendToHost(channel, args),
  // event emitter methods
  setMaxListeners: (n) => {
    ipcRenderer.setMaxListeners(n)
    return _ipcRenderer
  },
  getMaxListeners: () => ipcRenderer.getMaxListeners(),
  listeners: (e) => ipcRenderer.listeners(e),
  rawListeners: (e) => ipcRenderer.rawListeners(e),
  emit: (e, ...args) => ipcRenderer.emit(e, ...args),
  listenerCount: (e) => ipcRenderer.listenerCount(e),
  addListener: (e, l) => {
    ipcRenderer.addListener(e, l)
    return _ipcRenderer
  },
  off: (e, l) => {
    ipcRenderer.off(e, l)
    return _ipcRenderer
  },

  prependListener: (e, l) => {
    ipcRenderer.prependListener(e, l)
    return _ipcRenderer
  },
  prependOnceListener: (e, l) => {
    ipcRenderer.prependOnceListener(e, l)
    return _ipcRenderer
  },
  eventNames: () => ipcRenderer.eventNames()
}

const api = {
  shell,
  clipboard,
  ipcRenderer: _ipcRenderer,
  dialog: {
    showCertificateTrustDialog(...options: any[]) {
      return ipcRenderer.invoke('dialog:showCertificateTrustDialog', ...options)
    },
    showErrorBox(...options: any[]) {
      return ipcRenderer.invoke('dialog:showErrorBox', ...options)
    },
    showMessageBox(...options: any[]) {
      return ipcRenderer.invoke('dialog:showMessageBox', ...options)
    },
    showOpenDialog(...options: any[]) {
      return ipcRenderer.invoke('dialog:showOpenDialog', ...options)
    },
    showSaveDialog(...options: any[]) {
      return ipcRenderer.invoke('dialog:showSaveDialog', ...options)
    }
  } as Pick<Dialog, 'showCertificateTrustDialog' | 'showErrorBox' | 'showMessageBox' | 'showOpenDialog' | 'showSaveDialog'>
}

try {
  contextBridge.exposeInMainWorld('electron', api)
} catch {
  (window as any).electron = api
}
